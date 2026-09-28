# Business Entity Resolution: end-to-end reproduction (team Aspiring Alphas)

Regenerates `output/matching_results.tsv` (leaderboard **0.986509**) and `output/candidate_pairs.tsv` from the raw
challenge data. No external data, APIs, geocoding or lookups are used. Every model is open (MIT / Apache-2.0),
and the total size of all shipped models is **≈ 6.22 B parameters** (< 8 B).

```
dataset/  ->  1. blocking + LightGBM   ->  handoff/full_out/        (candidates, LightGBM OOF + test probs)
          ->  2. xlm-roberta CEs      ->  handoff/ce_*_out/        (src/xlmr_ce/README.md)
          ->  3. Qwen3-4B CE          ->  handoff/ce_qwen_*_out/   (src/qwen_ce/README.md)
          ->  4. LightGBM stacker + decision  ->  output/matching_results.tsv
```

| Model | Params | Licence | Role |
|---|---|---|---|
| intfloat/multilingual-e5-small (fine-tuned) | 0.12 B | MIT | embedding blocking |
| LightGBM ranker (73 pair features) | negligible | MIT | candidate scoring |
| xlm-roberta-base cross-encoder | 0.28 B | MIT | stacker feature `ce_prob` |
| xlm-roberta-large cross-encoder (+ France self-trained copy) | 0.56 B + 0.56 B | MIT | `ce_large_prob` (France: swapped) |
| xlm-roberta-large continued on full data (`ce_full`) | 0.56 B | MIT | `ce_full_prob` |
| Qwen3-4B-Base cross-encoder, two LoRA r=32 adapters on one base (India/US + France self-trained) | 4.02 B + 2 × 0.07 B | Apache-2.0 | `qwen_prob` |
| LightGBM stacker | negligible | MIT | final pair probability |

## 1. Environment
- Stages 1 and 4 (CPU + one GPU for the e5 encoder): `pip install -r requirements.txt` (pinned; Python 3.12/3.13).
- Stage 2 (xlm-roberta): `src/xlmr_ce/requirements-gpu.txt` (Python 3.12.14, torch 2.6.0+cu124, transformers 5.6.0).
- Stage 3 (Qwen3): `src/qwen_ce/requirements-gpu.txt` (Python 3.10.19, torch 2.9.1+cu128, transformers 5.5.0, peft 0.18.1).
- Hardware used: NVIDIA H100 80 GB (PBS cluster) for stages 1–3, 8 CPU threads per job; stage 4 on CPU.
  Every script caps its threads to the reservation (`N_CPUS`, `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, `MKL_NUM_THREADS`).
- Pretrained weights are downloaded once from Hugging Face; jobs then run with `HF_HUB_OFFLINE=1`.

Data layout (paths in `configs/*.yaml`):
```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```
Folds everywhere: `src/evaluate.fold_of(s1_id, 5)` = crc32(s1_id) % 5; **fold 0 is held out** from every model.

## 2. Stage 1: normalisation, blocking, LightGBM (all 2.2 M train S1) -> `handoff/full_out/`
PBS versions of these commands are in `scripts/hpc/` (`base_stages.sh`, `run_full.sh`, `pack_full.sh`).
```
python -m src.run --stage normalize --split train
python -m src.run --stage normalize --split test
python -m src.finetune_embed --pairs 400000 --max_steps 2500 --out artefacts/embed_ft    # contrastive e5 fine-tune
python -m src.run --stage embed --split train
python -m src.run --stage embed --split test
python -m src.run --stage block --split train      # rare-token IDF pass + embedding kNN, union, 25 candidates/S1
python -m src.run --stage block --split test       # -> the candidate set of output/candidate_pairs.tsv
python -m src.run --stage features --split train --config configs/full.yaml   # 73 pair features, every train S1
python -m src.run --stage rank     --split train --config configs/full.yaml   # grouped 5-fold LightGBM, stage 1 + 2 (OOF)
python -m src.run --stage features --split test  --config configs/full.yaml
python -m src.run --stage rank     --split test  --config configs/full.yaml
python -m src.run --stage decide   --split test  --config configs/full.yaml   # also writes candidate_pairs.tsv
python scripts/hpc/pack_full.py      # -> handoff/full_out/{oof_train_full_part*, probs_model_full_part*, candidate_pairs.tsv.gz.part*}
```
Blocking recall on train: 0.9920 (oracle F0.5 ceiling 0.9975). Test: 42,788,807 candidate pairs for 1,732,544 S1
(24.7 per S1), which is exactly the set in `output/candidate_pairs.tsv` and the set the final stacker scores.
LightGBM (step 3) OOF macro F0.5 0.9805.

The cross-encoders are first trained on a 500k-S1 development sample of the same pipeline (`configs/v2.yaml`),
exported as `handoff/ce/{train,test}_pairs_part*.parquet` (s1_id, pool_id, lgbm_prob, label, fold).

## 3. Stage 2: xlm-roberta cross-encoders -> `handoff/ce_out`, `ce_large_out`, `ce_full_out`, `ce_france_out`
See **`src/xlmr_ce/README.md`**: base and large (`ce_rescore.py`), France self-training r1 (`ce_france.py`),
large continued on 4 M full-data pairs (`ce_full.py`), and scores for the step-3 pairs outside the development
files (`ce_extra.py`). It lists the exact commands, data, held-out metrics and checkpoint md5s.

## 4. Stage 3: Qwen3-4B cross-encoder -> `handoff/ce_qwen_frself_out`
See **`src/qwen_ce/README.md`**:
- a LoRA cross-encoder on Qwen3-4B-Base trained on 1.2 M hard (contested) pairs; it scores India/US;
- a second adapter, self-trained on France: 3-teacher agreement pseudo-labels + same-name "sibling" hard
  negatives, continued from the first adapter, lr 1e-5, 1 epoch; it scores France only.

`handoff/ce_qwen_frself_out` = the first adapter's India/US scores + the France self-trained adapter's France scores.

## 5. Stage 4: stacker + decision -> `output/matching_results.tsv`
Held-out check (India/US fold 0, 434,574 S1, decision tuned on one half and scored on the other), expect 0.9903:
```
python -m src.stack_eval --pairs "handoff/full_out/oof_train_full_part*.parquet" \
  --extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=handoff/ce_qwen_frself_out \
  --fill qwen=ce_full --compare_extras --only_all --tag final
```
Final submission:
```
python -m src.stack_submit --features extra \
  --extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=handoff/ce_qwen_frself_out \
  --fill qwen=ce_full --pairs "handoff/full_out/oof_train_full_part*.parquet" \
  --lgbm_test "handoff/full_out/probs_model_full_part*.parquet" --swap ce_large=handoff/ce_france_out \
  --shift france:-1.25 --alpha 1.5 --out output/final
```
What it does:
- The stacker is a LightGBM on fold-0 labelled pairs. Its features are:
  - the LightGBM prob and every CE prob (`ce_prob` from `handoff/ce_out`);
  - per-S1 context (rank / gap / max / second / sum / count ≥ 0.5 of each score);
  - pool-record competition (how many S1 claim the record, best other S1, margin);
  - an empty-address flag.
- `--fill qwen=ce_full`: pairs without a Qwen score take ce_full's, identically in train and test.
- `--swap ce_large=handoff/ce_france_out`: on France (a country absent from train), the ce_large column is replaced
  by the France self-trained copy.
- `--shift france:-1.25`: a −1.25 logit shift on France pairs, correcting over-confidence on an unseen country.
  The shift was chosen on the leaderboard: −1 → 0.986201, −1.25 → 0.986232, −1.5 → 0.98615 on the v11 base.
- Decision (`src/decide.py`):
  - one-to-one: a pool record goes to at most one S1;
  - then per S1, the prefix of candidates that maximises expected F0.5 (alpha 1.5), empty allowed.
- `utils/validate_submission.py` then checks the output format.

## 6. Determinism
Seeds are fixed (42 unless stated). Folds are a crc32 hash of the S1 id. LightGBM with a different thread count
or platform changes a few hundred France decisions: e.g. the laptop and HPC builds of the same recipe differ by
~650 France pairs, and India/US decisions are identical. The submitted `output/matching_results.tsv` is the exact
file that scored 0.986509 (md5 `8b68aed84c39980832ca283d1f050748`).

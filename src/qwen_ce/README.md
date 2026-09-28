# Qwen3-4B cross-encoder (the `qwen` stacker column)

A pairwise cross-encoder that scores a (S1 record, pool record) pair as "same business" or not. Its probability
`ce_prob` is one extra feature column (`qwen`) in the final LightGBM stacker (`src/stack_submit.py`).
It is trained on labelled **train** pairs only. The France adapter uses train labels plus pseudo-labels on the
unlabelled France **test** text. No test labels, no external data, no other teams' code.

## Models in the final submission (Apache-2.0)

| piece | params | used for | weights |
|---|---|---|---|
| `Qwen/Qwen3-4B-Base` (Hugging Face) | 4.02B | shared base for both adapters | download to `$BER_MODELS/Qwen3-4B-Base` |
| v11 LoRA adapter (r=32 on q,k,v,o,gate,up,down) + score head | 0.066B | India / US pairs | `qwen3_v11_adapter`, adapter_model.safetensors md5 `80576b1bf568f12b8e0ea59e653cc822` |
| France self-trained LoRA adapter (v11 continued 1 epoch) | 0.066B | France pairs only | `qwen3_frself_adapter`, adapter_model.safetensors md5 `ff0c2b44ede8c3f5db70c4570ba30cd3` |

Only one copy of the 4B base is used, so the whole submission is ~6.22B parameters: the v8 stack is 2.08B, the base
4.02B and the two adapters 0.13B. That is under the 8B cap.

## Environment (GPU)

- Python 3.10.19, CUDA 12.8, one GPU with >= 60 GB free (we used H100 80 GB, one per job).
- `pip install -r requirements-gpu.txt`
- The CPU-only steps (`prep*.py`, `pack*.py`) also run with the root `requirements.txt` plus `scikit-learn`.

## Paths

`paths.py` holds every default location. Run all scripts **from this folder** (`src/qwen_ce`).

| env var | default | content |
|---|---|---|
| `QWEN_CE_WORK` | this folder | `inputs/`, `out*/`, `logs/` |
| `BER_DATA` | `<package root>/dataset` | `train/train_source{1,2,3}.tsv`, `train/train_ground_truth.tsv`, `test/test_source{1,2,3}.tsv` |
| `BER_MODELS` | `~/models` | `Qwen3-4B-Base/` |
| `BER_HANDOFF` | `<package root>/handoff` | stage-1 outputs, and the score folders the stacker reads |

## Files

| file | purpose |
|---|---|
| `link_inputs.sh` | fills `inputs/` from the stage-1 outputs: `handoff/full_out` (LightGBM full-data OOF + test probs), `handoff/ce_full_out` (xlm-r ce_full scores), `handoff/ce` (train/test candidate pairs) |
| `prep.py` | `inputs/train.parquet` (1.2M train pairs, fold != 0), `fold0.parquet` (held-out fold 0), `test.parquet` (1.46M contested test pairs) |
| `prep_france.py` | `inputs/france_all.parquet`: every France test candidate with LightGBM prob >= 0.001 (1,675,848 pairs) |
| `prompts.py` | pair -> token ids (`plain` template: `name \| address` of both records) |
| `qwen_ce.py` | trains the v11 LoRA CE for 1 epoch, then scores fold 0 and the contested test pairs |
| `qwen_score.py` | scores any pair list with a trained adapter |
| `package.py` | packs the v11 scores into `handoff/ce_qwen_out` (the stacker's CE format) |
| `prep_selftrain.py`, `prep_selftrain_sib.py` | France self-training set: 3-teacher one-to-one pseudo-labels + sibling hard negatives + India/US replay |
| `qwen_dann.py` | continues an adapter. `--max_lambda 0` gives plain training with no domain head, which is what the final used |
| `gate_frcanon.py` | offline gate for the France adapter: India/US AUC must not drop by more than 0.001 |
| `pack_frcanon.py` | `handoff/ce_qwen_frself_out`: the v11 files with only the France scores replaced |
| `data/gate_*_50k.parquet` | the two 50k-row gate pair lists (ids only). See the note in step 3 |
| `pbs/*.pbs` | the PBS jobs we ran (1 GPU each). They pick a free GPU on a shared node and call the commands below |

## Reproduce

```bash
cd src/qwen_ce
./link_inputs.sh                 # needs handoff/full_out, handoff/ce_full_out, handoff/ce (stage 1, see the package README)
python prep.py
python prep_france.py

# 1. v11 adapter: 1 epoch over 1.2M train pairs (5.4 h training + 1.9 h scoring on one H100)       [pbs/run_train.pbs]
python qwen_ce.py --out out                                     # -> out/adapter, out/qwen_oof_fold0.parquet, out/qwen_test_contested.parquet
python qwen_score.py --adapter out/adapter --pairs inputs/france_all.parquet --split test --stem ce_test_france --out out_scores
python package.py all                                           # -> handoff/ce_qwen_out (fold-0 OOF + contested test)
cp out_scores/ce_test_france_part*.parquet out_scores/ce_test_france.DONE ../../handoff/ce_qwen_out/

# 2. France self-training set (CPU)
python prep_selftrain.py         # 108k France pseudo pairs (lgbm_full, ce_full and v11 Qwen all agree; one-to-one) + 150k IN/US replay
python prep_selftrain_sib.py     # 42k sibling hard negatives replace easy negatives -> inputs/selftrain_fr.parquet (300k pairs)

# 3. France adapter: continue v11 for 1 epoch, lr 1e-5, no domain loss (48 min + 52 min France rescore)          [pbs/run_frself.pbs]
python qwen_dann.py --adapter out/adapter --max_lambda 0 --src_file inputs/selftrain_fr.parquet --target gate_fr_50k.parquet \
    --lr 1e-5 --bs 64 --accum 2 --out out_frself
python qwen_score.py --adapter out_frself/adapter --pairs inputs/france_all.parquet --split test --stem ce_test_france --out out_frself_scores
python pack_frcanon.py out_frself_scores ../../handoff ce_qwen_frself_out   # the v11 files with only the France scores replaced
```

Then run the stacker from the package root with `--extra qwen=handoff/ce_qwen_frself_out --shift france:-1.25`. The full
command is in the package README.

Scoring only: with the released adapters, skip training and point `qwen_score.py --adapter` at
`qwen3_v11_adapter` or `qwen3_frself_adapter`.

Note on `--target` in step 3: with `--max_lambda 0` the target pairs are never passed through the model. The script
still draws random indices into them, so we keep the same 50k-row file to keep the random stream identical to our run.

Training recipe (v11): 1 epoch, batch 64, AdamW lr 1e-4, linear schedule with 3% warm-up, max length 128 tokens,
bf16 autocast, seed 7. Score head: a linear layer on the last token's hidden state.
France adapter: seed 11, lr 1e-5, batch 64 x accum 2.

## Results

| | number |
|---|---|
| stacker held-out macro F0.5 (India/US fold 0), v8 -> v11 | 0.9899 -> 0.9903 |
| v11 Qwen fold-0 AUC on contested pairs | 0.9025 |
| v8 stack (no Qwen) -> + Qwen (v11), LB | 0.985143 -> 0.986201 |
| + France self-trained adapter (France pairs only), shift -1.25, LB (FINAL) | **0.986509** |

Tried and not used (in the Documentation): Qwen3.5-4B, Qwen3-Reranker-4B, bge-reranker-v2-m3, second/third Qwen3
seeds, test-time French text canonicalisation, and a triangle-consistency filter. Each lost on held-out data or on the LB.

# xlm-roberta cross-encoders (`src/xlmr_ce/`)

Four pairwise cross-encoders feed the final LightGBM stacker (`src/stack_submit.py`). Each scores a
(Source-1 record, candidate pool record) pair given as text `"business_name | business_address"`, and outputs
P(same business). All four are MIT-licensed xlm-roberta models (no external data, no API):

| Stacker column | Checkpoint (HPC) | Base | Params | How it was trained |
|---|---|---|---|---|
| `ce_prob` (base) | `artefacts/ce_base/final` | FacebookAI/xlm-roberta-base | 278 M | `ce_rescore.py`, bs 256, lr 2e-5 |
| `ce_large_prob` | `artefacts/ce_large/final` | FacebookAI/xlm-roberta-large | 560 M | `ce_rescore.py`, bs 128, lr 1e-5 |
| `ce_full_prob` | `artefacts/ce_full_large/final` | ce_large, continued | 560 M | `ce_full.py`, 4 M full-data pairs, lr 5e-6 |
| `ce_large_prob` on France only (`--swap`) | `artefacts/ce_france/final` | ce_large, continued | 560 M | `ce_france.py`, France self-training |

model.safetensors md5: base `0c182b869f01c249ca3b417872524b0d` · large `4750cdca081f1fd0286f0e71a51abdb3` ·
full `05f17f21e5022753132f506b5a01a1f8` · france `2f85641f4eb034a02a1d4b8d718d0e60`.

Pinned environment: `src/xlmr_ce/requirements-gpu.txt` (Python 3.12.14, torch 2.6.0+cu124, transformers 5.6.0).

Common recipe (all four): `AutoModelForSequenceClassification(num_labels=1)`, BCE-with-logits, bf16 autocast on an
NVIDIA H100 80 GB, AdamW (weight decay 0.01), 5% linear warmup + linear decay, 1 epoch, grad-clip 1.0, max_len 128,
seed 42, texts tokenised as a pair (S1 text, candidate text).

## Inputs these scripts read
- `dataset/{train,test}/*_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv` (competition data)
- `handoff/ce/{train,test}_pairs_part*.parquet`: candidate pairs of the 500k-S1 development sample, with the
  LightGBM probability `lgbm_prob`, `label` and `fold` (branch `ce-results`; built from the blocking + LightGBM
  pipeline in `src/run.py`)
- `artefacts/train/oof.parquet` (step-3 LightGBM out-of-fold probs on all 2.2 M train S1; for ce_full use the v1
  file `oof_model_full_v1.parquet`, which is what ce_full was trained on) and `artefacts/test/probs_model_full.parquet`
  (step-3 LightGBM test probs), both from `src/run.py` with `configs/full.yaml`
- folds: `src/evaluate.fold_of(s1_id, 5)` (crc32 % 5); fold 0 is held out everywhere

## Reproduce (in this order; each is one PBS job, 1 GPU, 8 cores)
```
# 1) base and large on the development sample (fold != 0 rows with lgbm_prob >= 0.001 + 10% of the rest)
qsub -v MODEL=FacebookAI/xlm-roberta-base,TAG=base,OUT=handoff/ce_out -o logs/ce_base.log src/xlmr_ce/jobs/ce.sh
qsub -v MODEL=FacebookAI/xlm-roberta-large,TAG=large,OUT=handoff/ce_large_out,EXTRA="--bs 128 --lr 1e-5" \
     -o logs/ce_large.log src/xlmr_ce/jobs/ce.sh
#    -> 3,451,055 train pairs (1,373,809 positive); base 13,481 steps / 42 min, large 26,962 steps / 49 min
# 2) France self-training r1 (starts from ce_large)
qsub src/xlmr_ce/jobs/ce_france.sh
# 3) ce_full: ce_large continued on 4 M full-data step-3 pairs (fold != 0), lr 5e-6, bs 128
qsub src/xlmr_ce/jobs/ce_full.sh
# 4) score the step-3 candidate pairs the development pair files do not cover, for base / large / france
qsub src/xlmr_ce/jobs/ce_extra.sh
```
Each script writes `<out>/ce_oof_fold0_part*.parquet` and `<out>/ce_test_part*.parquet` (or `ce_test_unseen_part*`
for France), columns `s1_id, pool_id, ce_prob`, plus `fold0_metrics.json` and `.DONE` markers. `ce_extra.py` adds
`ce_extra_{fold0,test}_part*.parquet` to the same folders. These folders are the `--extra` / `--swap` inputs of
`src/stack_eval.py` and `src/stack_submit.py`.

`jobs/pick_gpu.sh` pins `CUDA_VISIBLE_DEVICES` to the least-used idle GPU (PBS on this cluster does not isolate GPUs);
all scripts cap CPU threads to the PBS reservation (`N_CPUS`, `OMP_NUM_THREADS`, ...) and run with `HF_HUB_OFFLINE=1`
(models pre-downloaded to the HF cache).

## France self-training r1 (`ce_france.py`)
France appears only in test (an unseen country). Pseudo-labels come from our own models on the France candidate pairs:
blend = sigmoid(0.2·logit(lgbm) + 0.4·logit(ce_base) + 0.4·logit(ce_large)), with one-to-one per pool record.
- positive: blend >= 0.97, every model >= 0.8, the pair is the pool record's best S1, and the runner-up S1 <= 0.5
- negative: every model <= 0.2, decoys first (the pool record is a confident positive of ANOTHER S1), then random
Run: 600,000 pseudo-positives, 56,551 decoy + 745,038 random negatives, mixed 1:1 with 1,401,589 labelled train
pairs (fold != 0) -> 2,803,178 pairs, 21,900 steps, 72 min. Labelled fold-0 sanity check: AUC 0.99838 / logloss 0.0462
(ce_large 0.99839 / 0.0457), so India/US are not harmed.

## Held-out results (fold 0, never trained on)
| Model | Pair AUC | Pair logloss | Rows |
|---|---|---|---|
| LightGBM (reference) | 0.99497 | 0.07768 | 684,947 handoff fold-0 pairs |
| ce base | 0.99817 | 0.04916 | same |
| ce large | 0.99839 | 0.04570 | same |
| ce france r1 | 0.99838 | 0.04617 | same |
| ce_full | 0.99809 | 0.04916 | 2,635,403 step-3 fold-0 pairs (step-3 LightGBM there: 0.99629 / 0.06402) |

Entity-level macro F0.5 on the 99,964 development fold-0 S1 (`src/blend_eval.py`, decision tuned on one half and
scored on the other): LightGBM 0.9723, ce_large alone 0.9812, ce_full alone 0.9824, 0.5/0.5 logit blend with
LightGBM 0.9871 (large) / 0.9875 (full). The final stacker combines all of them with the Qwen3 cross-encoder
(`src/qwen_ce/`), held-out 0.9903 on 434,574 fold-0 S1.

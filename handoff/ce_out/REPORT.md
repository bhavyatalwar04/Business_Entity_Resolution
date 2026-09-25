# Cross-encoder results (HPC, NVIDIA H100 80GB)

## Model and training
- `FacebookAI/xlm-roberta-base` (MIT, 278M), `AutoModelForSequenceClassification`, num_labels=1, BCEWithLogits
- bf16 autocast, max_len 128, batch 256, lr 2e-5 (AdamW, wd 0.01), 5% warmup + linear decay, 1 epoch, grad clip 1.0
- Train rows: fold != 0 only; all rows with lgbm_prob >= 0.001 + 10% of the rest (seed 42)
  -> **3,451,055 pairs** (1,373,809 positive), 13,481 steps, **42.2 min** training
- Text: `business_name | business_address` from the challenge TSVs (S1 text, candidate text)
- Checkpoint (kept on HPC): `artefacts/ce_base/final` (1.1 GB)
- Code: `src/ce_rescore.py`, PBS script `jobs/ce.sh`

## Fold-0 held-out (rows with lgbm_prob >= 0.001; 684,947 pairs, 342,905 positive)

| | AUC | logloss |
|---|---|---|
| ce_prob | **0.99817** | **0.04916** |
| lgbm_prob | 0.99497 | 0.07768 |

## Files
- `ce_oof_fold0_part*.parquet`: s1_id, pool_id, ce_prob (float32), sorted by s1_id, zstd
- `ce_test_part*.parquet`: all 13,047,424 test pairs (added in a follow-up commit)

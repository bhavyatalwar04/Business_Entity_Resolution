# Cross-encoder results: xlm-roberta-large (HPC, NVIDIA H100 80GB)

- `FacebookAI/xlm-roberta-large` (MIT, 560M), num_labels=1, BCEWithLogits; bf16, max_len 128, batch 128,
  lr 1e-5 (AdamW, wd 0.01), 5% warmup + linear decay, 1 epoch, grad clip 1.0
- Train rows: fold != 0; all rows with lgbm_prob >= 0.001 + 10% of the rest (seed 42), same as ce_out
  -> **3,451,055 pairs** (1,373,809 positive), 26,962 steps, **49.2 min**
- Code: `src/ce_rescore.py` (branch ce-results), checkpoint on HPC: `artefacts/ce_large/final`

## Fold-0 held-out (rows with lgbm_prob >= 0.001; 684,947 pairs, 342,905 positive)

| | AUC | logloss |
|---|---|---|
| ce_large | **0.99839** | **0.04570** |
| lgbm_prob (v2) | 0.99497 | 0.07768 |
| ce base (ce_out, for reference) | 0.99817 | 0.04916 |

## Files
- `ce_oof_fold0_part*.parquet`: s1_id, pool_id, ce_prob (float32), sorted by s1_id, zstd
- `ce_test_part*.parquet`: pending (follow-up commit)

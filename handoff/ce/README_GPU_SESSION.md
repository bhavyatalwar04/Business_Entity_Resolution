# Task for the GPU session: cross-encoder re-scorer

Goal: a multilingual **cross-encoder** that reads (S1 record, candidate record) together and outputs
P(match). It will be blended with our LightGBM probabilities to fix the main error type: **decoys**
— same/similar generic name + same city, but a different street, an extra descriptor word
("Développement", "Distribution", "Holding", "International", "Ecole"), a different legal form
(SARL vs SNC/SA), or a different house number (30 vs 34, 18730 vs 1873).

## Rules (challenge constraints — do not break)
- Model must be **MIT or Apache-2.0** and **≤ 8B params**. Use `xlm-roberta-base` (MIT, 278M).
  Alternative if time allows: `xlm-roberta-large` (MIT, 560M). Do NOT use Llama/Qwen-with-restrictive-license/APIs.
- **No external data, APIs, geocoding or lookups.** Only the challenge TSVs + these files.

## Inputs
1. Challenge data (same download as ours) → `dataset/train/*.tsv`, `dataset/test/*.tsv`
   (read with `sep="\t", dtype=str, keep_default_na=False, quoting=3`).
2. This folder (`handoff/ce/` in the repo, branch `main`):
   - `train_pairs_part*.parquet` — 12,371,240 rows: `s1_id, pool_id, lgbm_prob, label (0/1), fold (0-4)`.
     These are ALL blocking candidates of 500k training S1s; `lgbm_prob` is our out-of-fold LightGBM prob.
   - `test_pairs_part*.parquet` — 13,047,424 rows: `s1_id, pool_id, lgbm_prob` (test candidates with lgbm_prob ≥ 0.001;
     everything else is a certain non-match and need not be scored).

## Text of a record
`f"{business_name} | {business_address}"` (raw strings from the TSVs; S2/S3 ids are looked up in
source2/source3 of the same split). Input to the model = pair (S1 text, candidate text), max_len 128.

## Training
- Use only rows with **fold != 0** (fold 0 = held-out S1s for blending/tuning — never train on them).
- Sample: all positives with `lgbm_prob >= 0.001` + all negatives with `lgbm_prob >= 0.001`
  (these are the hard ones) + 10% of remaining negatives. Expect ~3–4M pairs. Shuffle.
- `AutoModelForSequenceClassification` (num_labels=1, BCEWithLogits) or 2-class CE; fp16/bf16;
  lr 2e-5, warmup 5%, linear decay, batch 64–128, **1 epoch** (2 if fast). Log loss every ~1k steps.
- Save the checkpoint (keep it; we may need it for the final zip — MIT model weights are fine to ship).

## Inference (write these outputs)
1. `ce_oof_fold0` — score **all fold-0 rows with lgbm_prob ≥ 0.001** from train_pairs.
   Report AUC and logloss vs `label`, and the same metrics for `lgbm_prob` on those rows (to compare).
2. `ce_test` — score **all 13,047,424 test pairs**.
Columns: `s1_id, pool_id, ce_prob` (float32). Sort by s1_id. Write as zstd parquet parts **< 90 MB each**.

## Hand back
Push to branch **`ce-results`** (not main) in `handoff/ce_out/`:
`ce_oof_fold0_part*.parquet`, `ce_test_part*.parquet`, and `REPORT.md` (model, #train pairs, epochs,
time, fold-0 AUC/logloss for ce vs lgbm). Commit as **arnav1803**, no AI co-author lines.
Also keep the checkpoint dir locally (report its path/size).

## Priority / time budget
Deadline is 27 Sep 23:59 IST. Deliver `ce_oof_fold0` first (so blending can be tuned), then `ce_test`.
If throughput is low, reduce training to ~1.5M pairs rather than skipping test inference.

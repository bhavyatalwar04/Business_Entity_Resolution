# Methodology draft (fill numbers as experiments land)

## EDA facts
- Train: S1 2,206,821 / S2 5,034,616 / S3 5,285,603. Test: S1 1,732,544 (India 810k, US 663k, France 259k).
- Singletons 5.6%. Matches per S1: mostly 2–6 (mean ≈ 3.5), max 11. 7.64M true pairs, 48% from S2.
- One-to-one holds exactly (no S2/S3 id under two S1s). Every true pair shares the country label.
- ~26% of S2/S3 records match no S1 (distractors). ~9% of S2 names are in Devanagari/Bengali
  script — phonetic transliterations of English names ("redd veNcrs praaivett limittedd").
- Name noise: legal words anywhere, junk marks, "(ID: n)", domain-style names, fka/aka aliases,
  OCR digit swaps. Address noise: St→"SAINT", reordering, state codes vs names vs native script,
  dropped components, mutated house numbers.

## Blocking
- Pass 1 — rare-token IDF cosine per country over name phonetic-skeleton tokens, whole no-space
  name, and address tokens (tokens with pool df > 3000 dropped). Token-only recall@20 = 83.2%,
  @50 = 87.3% (20k validation S1 vs full 10.3M pool).
- Pass 2 — multilingual-e5-small (MIT) fine-tuned with in-batch InfoNCE on 400k training pairs,
  batches grouped by city/state for hard negatives; exact GPU kNN per country. Recall@5/10/20/30 = 90.7 / 98.5 / 99.2 / 99.4%
  (20k validation S1 vs full pool) — the fine-tuned encoder carries blocking.
- Union (token top-12 + embedding top-20) capped at 25 per S1 by best rank across passes:
  ~99.1% recall at ~25 candidates per S1 (union@10 per pass: 98.8% at 16/S1; @15: 99.2% at 25/S1).

## Matching
- ~55 features (see src/features.py) + LightGBM, folds grouped by S1 hash.
- Stage 2 (per-S1 probability context) kept only if OOF improves.
- Decision: one-to-one + threshold/expected-F0.5 set selection tuned on OOF.

## Results log
| experiment | blocking recall | oracle F0.5 | OOF F0.5 |
|---|---|---|---|
| token pass only, 5k S1 smoke | 0.833 | 0.927 | 0.880 |

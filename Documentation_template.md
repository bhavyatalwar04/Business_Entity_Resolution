# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Aspiring Alphas  
**Team Members:** Arnav Sharma, Bhavya Talwar, [others]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve each Source 1 business against ~10M Source 2/3 records with a four-stage pipeline:
(1) **blocking**: the union of a rare-token IDF pass and an exact GPU kNN over a *fine-tuned*
multilingual-e5-small encoder, which keeps **99.2 %** of true matches in ~25 candidates per entity;
(2) a **two-stage LightGBM** pair model over 73 engineered features, trained on all 2.2M training
entities; (3) **cross-encoder re-scoring** of the plausible pairs with four fine-tuned XLM-RoBERTa
models and a LoRA-fine-tuned **Qwen3-4B-Base** classifier; (4) a **LightGBM stacker** over all scores
and their per-entity / per-candidate competition context, followed by a metric-aware decision layer
(one-to-one assignment plus expected-F0.5 set selection, empty set allowed).
Held-out macro F0.5 on 434,574 training entities never seen by any model: **0.9903** (India 0.9906,
US 0.9900). Public leaderboard: **0.986509**. France, which is absent from training, is handled by the
same country-agnostic code plus measured unseen-country adjustments: transductive self-training on
unlabelled French test records (of one xlm-r cross-encoder and of the Qwen3-4B adapter, with
same-name "sibling" hard negatives), and a logit shift for over-confidence.
Total model size **≈ 6.22B parameters** (limit 8B); all models are MIT or Apache-2.0.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the training split (`src/eda.py`):

| Fact | Value | Consequence for the design |
|---|---|---|
| Records | S1 2.21M · S2 5.03M · S3 5.29M (test: 1.73M · 4.89M · 5.08M) | all-pairs is impossible → blocking; memory-lean code for a 16 GB laptop |
| Singletons | 5.6 % of S1 | "predict empty" matters but is rare |
| Matches per S1 | mostly 2–6, mean ≈ 3.5 (max 11) | recall matters as much as the singleton decision |
| One-to-one | **0** S2/S3 ids appear under two S1s | each S2/S3 record may be assigned to at most one S1 |
| Country | 100 % of true pairs share the country label | block within each country value (open set) |
| Distractors | ~26 % of S2/S3 records match no S1 | the matcher must reject look-alikes |
| Scripts | ~9 % of S2 names in Devanagari/Bengali | phonetic transliterations of *English* names |
| Test countries | India 810k, US 663k, **France 259k (unseen)** | France is ~15 % of the metric; it must work without labels |

Noise observed in true pairs (examples from the data):
- **Names:** legal words anywhere (`Private Anand Foundation Ltd`), junk (`***`, `<<`, `(ID: 54473)`,
  `[Ltd]`), domain-style names (`UROLOGYSTRATEGICHEALTH.COM`, `@GIBSONGALLARDO`), aliases
  (`Yumabrixveo fka Allsun Allen, DO`, `… a/k/a Caum, LLC`, `name | www.x.com`), OCR digit swaps
  (`N0SE`, `BAILEY6ULF`), transliterations (`राम मार्केटिंग प्राइवेट लिमिटेड` → "raam maarketting praaivett limittedd").
- **Addresses:** `ST` rendered as `SAINT`, component reordering, state codes vs names vs native script
  (`MH` / `Maharashtra` / `महाराष्ट्र`), dropped components, mutated house numbers (`165` vs `16`).
- **Decoys:** the hardest negatives are near-duplicates (same name and street, a different house
  number) and true copies of *another* S1 (often with an empty address).

### 2.2 Solution Strategy
**Approach Type:** blocking + learned pair model + cross-encoder re-scoring + stacked, metric-aware decision.  
**Core Innovation:** a contrastively fine-tuned multilingual bi-encoder for blocking; cross-encoders
(including a 4B LLM classifier) that read both records jointly; a stacker that sees every score *in
competition* (how a pair ranks among its S1's candidates and among the S1s competing for the same
pool record); a decision layer that optimises the exact per-entity F0.5 under the one-to-one
constraint; and an unseen-country protocol in which every France decision was validated on a
leave-one-country-out proxy or measured directly.

```
raw TSVs → normalise → blocking: token-IDF ∪ fine-tuned-e5 kNN (per country, ≤25 candidates/S1)
  → 73 pair features → LightGBM stage 1 → + per-S1 probability context → LightGBM stage 2   (all pairs)
  → pairs with LightGBM prob ≥ 0.001: cross-encoders (xlm-r base, xlm-r large, xlm-r large full-data,
    xlm-r large France self-trained; Qwen3-4B LoRA on contested pairs + all France pairs)
  → stacker (LightGBM, 65 features: scores, logits, blends, per-S1 and per-candidate competition)
  → France logit shift −1.25 → one-to-one + expected-F0.5 set selection → matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

**Normalisation (`src/normalize.py`, hand-written, country-agnostic).** Unidecode transliteration,
lowercase, `&`→`and`, merged initials (`M.G.`→`mg`), junk/ID-tag removal, alias split
(`fka / aka / dba / |`), URL stems, OCR digit repair inside words. Name views: *clean*, *core*
(legal words removed: pvt/ltd/llc/inc/corp/sarl/sas/sa/eurl/…, also by phonetic skeleton so
"praaivett limittedd" is removed), *skeleton* (phonetic consonant key: bh/w→v, ph→f, sh→s, …,
drop non-initial vowels, squeeze repeats; "sebhen"/"seven" → `svn`), *no-space* (for domain-style
names) and *alias*. Addresses are mapped to canonical **short** forms in both directions
(road→rd, street/str/saint→st, rue→r, avenue/av→ave, boulevard/bd→blvd, nagar→ngr, chemin→ch, …),
and postal codes, house numbers and landmark tokens ("near/opp …") are extracted.

**Blocking keys used** (all computed within each country value; an unseen label on the pool side
falls back to the whole pool):
1. **Rare-token pass**: IDF-weighted cosine over `n:` name-skeleton tokens, `s:` whole no-space
   name and `a:` address tokens; tokens with pool document frequency > 3000 are dropped (keeps the
   sparse product sparse). Top-12 per S1.
2. **Dense pass**: `intfloat/multilingual-e5-small` (MIT) **fine-tuned** on 400k training pairs
   (only non-validation S1s) with symmetric InfoNCE; batches are built from S1s in the same
   city/state so in-batch negatives are hard. Records are encoded as `name | address`; exact
   inner-product kNN on GPU, chunked over queries and pool (pool embeddings are a disk memmap).
   Top-20 per S1.
3. **Union**, then keep the 25 best per S1 by best rank across passes.

**Candidate pairs generated:** 42,789,509 on test (24.7 per S1; France 24.9, India 24.7, US 24.6);
54,593,626 on train. Reduction ratio ≈ 1 − 24.7 / 3.9M (same-country pool) ≈ **99.9994 %**.

**How we ensured true matches were not lost:** recall was measured on held-out S1s against the
*full* pool (distractor density matters) after every change (`src/exp_blocking.py`):

| Pass (20k validation S1 vs full 10.3M pool) | @5 | @10 | @20 | @30 |
|---|---|---|---|---|
| rare-token IDF | 70.6 % | 78.6 % | 83.2 % | 85.3 % |
| fine-tuned e5 kNN | 90.7 % | 98.5 % | 99.2 % | 99.4 % |
| union (best rank) | 93.4 % | 98.8 % | 99.4 % | 99.5 % |

Final configuration on all training S1: **recall 99.0 % (India) / 99.3 % (US)**. For France (no
labels) we checked that the candidate volume matches the labelled countries: 24.9 candidates and 3.61
LightGBM-likely pairs per S1, against 24.7 / 3.47 for India. So France's remaining error is in
scoring, not blocking. Every final match is a subset of its candidates (enforced in code).

---

## 4. Matching Model

**Pair features (73, `src/features.py`):**
- **Name:** Levenshtein ratio, partial ratio, token-sort and token-set ratio, Jaro-Winkler on the core
  name; ratio on the clean name; ratio / token-set / Jaccard / overlap on phonetic skeletons; ratio and
  partial ratio on the no-space name; exact-core and no-space equality; alias token-set similarity.
- **Address:** ratio, partial, token-set, token-sort, token Jaccard/overlap; house-number equality;
  number-set Jaccard / overlap / conflict; postal-code Jaccard; landmark overlap; missing flags.
- **Decoy features:** signals for the near-duplicate pattern (same name and street with a different
  number, inserted or missing name words, empty pool address).
- **Blocking:** token-pass score & rank, embedding cosine & rank, best rank.
- **Context:** rank and gap-to-best of the candidate among its S1's candidates, and, computed over **all**
  S1s exactly as at test time, how many S1s compete for the candidate and this S1's rank among them.
- **Meta:** source (S2/S3), string lengths. Country is never one-hot encoded.

**LightGBM pair model (MIT).** 5 folds grouped by S1 (crc32 of the S1 id) on **all 2.2M training S1**
(54.6M pairs). Stage 2 adds per-S1 probability context from out-of-fold stage-1 probabilities. OOF
macro F0.5: stage 1 0.9794, stage 2 **0.9805** (300k S1).

**Cross-encoders** (read `name | address` of both records jointly; trained on out-of-fold data only):

| Model | Params | Licence | Training | Scores |
|---|---|---|---|---|
| xlm-roberta-base | 278M | MIT | pairs of a 500k-S1 sample | all LightGBM-plausible pairs |
| xlm-roberta-large | 560M | MIT | same sample | same |
| xlm-roberta-large "full" | 560M | MIT | continued on 4M pairs from all 2.2M S1 | same |
| xlm-roberta-large "France r1" | 560M | MIT | continued with transductive self-training on unlabelled French test records (600k confident pseudo-positives, 56k one-to-one decoys, 745k easy negatives, 1:1 labelled replay) | replaces the large model's score on France pairs |
| Qwen3-4B-Base + LoRA (r=32, α=64) | 4.02B + 0.07B | Apache-2.0 | 1 epoch on 1.2M hard (contested) training pairs; sequence-classification head | contested pairs (all countries) and **all** France pairs; elsewhere the column takes the xlm-r-large "full" score, identically on train and test |
| Qwen3-4B LoRA "France self-train" | +0.07B (same base) | Apache-2.0 | the adapter above continued 1 epoch (lr 1e-5) on 300k pairs: 108k French 3-teacher pseudo-labels (one-to-one positives, decoy/easy negatives), 42k **sibling hard negatives** (a pool record confidently owned by S1-A paired with a same-name S1-B in the same city at another address), 150k labelled India/US replay | replaces the Qwen score on France pairs only |

**Stacker (`src/stack_submit.py`, LightGBM).** 65 features: every model score and its logit, blends,
per-S1 context for each score (rank, gap to best, max, second, mass, count > 0.5), candidate-side
competition (number of S1s competing for the pool record, whether this S1 is the top one, margin),
empty-address flag, meta. Trained on the fold-0 pairs of all 2.2M training S1 (2.64M pairs), the only
pairs with out-of-fold cross-encoder scores, with an early-stopping slice, then refit.

**Decision layer.** (1) **One-to-one:** each S2/S3 record is kept only for its highest-probability S1.
(2) **Set selection:** per S1, candidates sorted by probability; choose the prefix (possibly empty)
maximising approximate expected F0.5 `1.25·Σp_top-k / (0.25·Σp + k)` against `α·Π(1−p)` for the empty
set, α = 1.5, tuned on held-out data. (3) **Unseen country:** on countries absent from training
(France), stacker logits are shifted by **−1.25** before the decision (see 5.2).

---

## 5. Results & Error Analysis

### 5.1 Scores
- **Held-out macro F0.5: 0.9903** (India 0.9906, US 0.9900) on the 434,574 fold-0 training S1s, scored
  by models that never saw them, with the exact formula of the problem statement (singletons included).
  The decision is tuned on one half and scored on the other, both ways.
- **Public leaderboard: 0.986509.** Since India/US are measured offline, the leaderboard gives
  France directly: LB ≈ 0.850·(India/US) + 0.150·(France), so France ≈ **0.965**.

| Build | Held-out | LB | Implied France |
|---|---|---|---|
| LightGBM + xlm-r base blend (v3) | 0.9865 | 0.980683 | — |
| + stacker (v4) | 0.9871 | 0.982057 | — |
| + xlm-r large (v5) | 0.9877 | 0.983546 | — |
| LightGBM on all 2.2M S1 + full-data CE + France r1 (v8) | 0.9900 | 0.985143 | 0.958 |
| + Qwen3-4B LoRA (v11) | 0.9903 | 0.986201 | 0.963 |
| v11 with France shift −1.25 | 0.9903 | 0.986232 | 0.963 |
| **+ France self-trained Qwen adapter on France (final)** | 0.9903 | **0.986509** | 0.965 |

### 5.2 Generalisation to the unseen country (France)
- **Proxy.** We trained models on India only and scored the US as if unseen. An India-trained
  stacker stays well-ranked on the US (AUC 0.992) but becomes over-confident in the middle score bands
  (said 0.63 → true 0.44; said 0.89 → true 0.57), while its top band stays reliable (p ≥ 0.95 → 99 %
  precision). The best logit shift on the proxy was strictly negative.
- **France shift, measured on the leaderboard with India/US held fixed:** −1: 0.986201 · **−1.25: 0.986232**
  · −1.5: 0.98615. We kept −1.25.
- **What moved France:** a stronger cross-encoder did. Qwen3-4B raised France from ~0.958 to ~0.963
  (+0.0054) while India/US rose only +0.0003. It is less extreme on French text than the
  xlm-r models (17 % of France pairs scored 0.05–0.95, against 12 % for xlm-r-large "full").

### 5.3 Error analysis
- **False positives (wrong merges):** near-duplicate decoys dominate: the same or near-same name at the
  same street with a slightly different house number (`… 18730 Little Lane` → `… 1873 Little Ln`;
  `Flat No 243` → `Flat No 247`), generic public names in another city, a different name at the exact
  same address, and name-only records with an empty address. Of the high-scoring wrong pairs on
  training data, 56 % are planted decoys and 44 % are true copies of *another* S1.
- **False negatives (missed matches):** the mirror image: true matches whose house number was also
  mutated (`14298` vs `14306`, `6138` vs `138`), a different trade name at the same address, empty
  addresses with noisy names, native-script names. F0.5 deliberately trades some recall for precision.
- **France specifically:** 52.6 % of French S1 share a core name with another S1, concentrated in ~15
  cities, and 6.2 % differ from a sibling only by street. The models output 3.18 France matches per S1
  against 3.37 for India/US, i.e. they are cautious exactly where same-name siblings make F0.5 costly.

### 5.4 What we tested and dropped (all measured before or on the leaderboard)
| Idea | Result |
|---|---|
| Qwen3.5-4B instead of Qwen3-4B | India/US equal (held-out 0.9902), France −0.0034 (LB 0.985613) |
| Two Qwen3-4B LoRA adapters averaged | India/US equal, France −0.0028 (LB 0.985777) |
| Domain-adversarial training (DANN, gradient reversal) | proxy US AUC 0.9538 vs 0.9575 control; −0.012 on xlm-r-large |
| Qwen3-Reranker-4B, bge-reranker-v2-m3, xlm-roberta-xl | no held-out gain over Qwen3-4B / xlm-r |
| Mixing Qwen's raw score into France, temperature re-calibration | proxy: monotone loss / +0.0001 |
| French text canonicalisation, S2↔S3 triangle-consistency filter, per-pair rules | all hurt held-out or proxy |
| Synthetic French pairs | AUC 0.75 / 0.63 against 0.97 for real pairs |
| France self-training of Qwen3 at lr 3e-5 (the lr 1e-5 run is in the final) | India/US contested AUC −0.0022 (gate fail) |

---

## 6. Conclusion
Blocking with a fine-tuned dense retriever (99 % recall at ~25 candidates), a feature-rich LightGBM on
all training data, cross-encoder re-scoring and a competition-aware stacker take held-out macro F0.5 to
0.990. The unseen country is the real difficulty: India/US saturate near 0.990, while France reached
~0.965 (Qwen3-4B +0.0054, then France self-training with sibling hard negatives +0.0021). The biggest lessons: measure blocking against the full pool; exploit the one-to-one and
same-country structure; judge unseen-country changes by leaderboard-implied France, not by held-out
scores, because India/US gains did not predict France (Qwen3.5 was better on India/US and worse on
France); and validate every adaptation trick on a leave-one-country-out proxy before spending a
submission. Next steps would be French-aware pair data and listwise ("select among candidates")
scoring for same-name siblings.

---

## Appendix

### A. Code Artefacts
Repository `Business_Entity_Resolution` (branch `main`; build notes for the final file on branch
`subs-final`, `BUILD_NOTES.md`; Qwen training code on branch `ce-qwen`, folder `qwen_ce/`).
- Blocking, features, LightGBM: `python -m src.run --stage <stage> --split <train|test>`, configs in `configs/`.
- Cross-encoders: `src/ce_rescore.py`, `jobs/ce_full.py`, `jobs/ce_france.py`; Qwen: `qwen_ce/qwen_ce.py`, `qwen_ce/qwen_score.py`.
- Held-out evaluation of the stacker: `python -m src.stack_eval --pairs "handoff/full_out/oof_train_full_part*.parquet" --extra ce_large=… --extra ce_full=… --extra qwen=… --fill qwen=ce_full --compare_extras --only_all`.
- Final file: `python -m src.stack_submit --features extra --extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=handoff/ce_qwen_frself_out --fill qwen=ce_full --pairs "handoff/full_out/oof_train_full_part*.parquet" --lgbm_test "handoff/full_out/probs_model_full_part*.parquet" --swap ce_large=handoff/ce_france_out --shift france:-1.25 --alpha 1.5 --out output/final`.
- Unseen-country proxy: `src/proxy_eval.py`, `src/ce_dann_local.py`. Validator: `utils/validate_submission.py`.

**Models and parameter count:** multilingual-e5-small 118M (MIT) + xlm-roberta-base 278M (MIT) +
3 × xlm-roberta-large 560M (MIT) + Qwen3-4B-Base 4.02B (Apache-2.0) + 2 LoRA adapters 0.13B ≈ **6.22B** (< 8B).
LightGBM (MIT) models are negligible. No external data, APIs, geocoding or lookups; all pseudo-labels
come from our own models on the provided test candidates.

### B. Submission log
All uploads with held-out, LB and implied France are in `logs/submissions.csv`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.

# Blend submissions: v2 LightGBM + cross-encoder (xlm-roberta-base)

Blend = sigmoid(0.5 * logit(ce_prob) + 0.5 * logit(lgbm_prob)) on the handoff test pairs (lgbm_prob >= 0.001).
Weight and decision chosen on fold 0 (99,964 held-out S1): each weight tuned on one half and scored on the
other, both ways round. Decision: expected-F0.5 prefix, alpha 1.5, one-to-one.

| w_ce | fold-0 macro F0.5 (honest) |
|---|---|
| 0.0 (LightGBM v2) | 0.97232 |
| 0.3 | 0.98506 |
| **0.5** | **0.98653** |
| 0.6 | 0.98619 |
| 0.7 | 0.98566 |
| 0.85 | 0.98379 |
| 1.0 (CE only) | 0.98035 |

## Variants (differ only for S1 whose country never appears in training = France, 259,452 S1)
Unseen-country probabilities shifted in logit space before the decision.

| variant | shift | France empty | France matches/S1 | S1 differing from v2 (sets): all / France |
|---|---|---|---|---|
| base | 0 | 5.25% | 3.306 | 17.86% / 30.39% |
| strict | -1 logit | 5.66% | 3.187 | 18.59% / 35.26% |
| loose | +1 logit | 4.87% | 3.419 | 17.76% / 29.76% |

US/India are identical across variants (empty 5.8%, 3.34 matches/S1). All three pass utils/validate_submission.py.
Unzip before submitting: `gunzip -k handoff/blend_out/<variant>/matching_results.tsv.gz`.
Code: `src/blend_submit.py`, PBS script `jobs/blend.sh`.

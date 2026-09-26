# France probes on top of v8 (0.985143)

Finding: France has 14,175 pool records that ce_full scores >= 0.95 but v8 leaves unmatched (US: 917). The step-3
LightGBM disagrees on them (median prob 0.37 vs 0.03 in the US). 36.8% of those with legal forms on both sides have a
legal-form conflict (SARL vs SAS ...), while legal-form conflicts occur in only 0.0% (India) / 0.4% (US) of TRUE training
matches, so LightGBM learned "conflict => decoy" from US/India. France may instead swap legal forms as noise.

Both files = v8 + extra France pairs; only pool records unmatched in v8, one S1 each (its highest-CE S1), so one-to-one holds.
Both pass utils/validate_submission.py. US/India rows are identical to v8.

| file | added pairs | France S1 touched |
|---|---|---|
| v8i_fr_cehigh | all 14,175 CE>=0.95 rejected France pairs | 13,347 (5.1%) |
| v8h_fr_legal  | the 2,581 of them with a legal-form conflict | 2,553 (1.0%) |

Reading: v8i > v8 => trust the CE on confident France pairs (a France recall gap). v8i < v8 => those are decoys.
france_cehigh_rejected_pairs.parquet: s1_id, pool_id, ce_prob, legal_conflict (the added pairs). Built by build_probe.py.

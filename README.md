# Business Entity Resolution — Amazon ML Challenge 2026

For each Source 1 business, find all matching Source 2 / Source 3 records.
Metric: macro F0.5 per S1 entity (singletons included).

## Layout

```
configs/default.yaml   k values, thresholds, model names, seeds
src/
  run.py               CLI: python -m src.run --stage all --split test
  io_utils.py          read_tsv(sep='\t'), write_tsv, id checks
  normalize.py         clean_name(), clean_address(), extract_postcode()
  blocking.py          tfidf_block(), embed_block(), token_block(), union_candidates()
  features.py          build_pair_features()
  cross_encoder.py     build_pairs_text(), train_ce(), predict_ce()
  ranker.py            train_lgbm(), predict_lgbm(), calibrate()
  decide.py            one_to_one(), s2_s3_boost(), choose_set()
  evaluate.py          f05(), macro_f05(), blocking_recall(), loco_eval()
  export.py            write matching_results.tsv + candidate_pairs.tsv
dashboard/app.py       Streamlit review dashboard (not scored)
dataset/train, test    challenge TSVs (gitignored)
utils/                 validate_submission.py from the student resource
artefacts/<split>/     cached stage outputs (gitignored)
output/                matching_results.tsv, candidate_pairs.tsv
logs/                  experiments.csv, submissions.csv
```

## Setup

```
pip install -r requirements.txt
```

Put the challenge files in `dataset/train/` and `dataset/test/`, and the
provided `validate_submission.py` in `utils/`.

## Run

```
python -m src.run --stage all --split test
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

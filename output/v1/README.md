# Submission v1 outputs (2026-09-25, commit 399cc48)

Local OOF macro F0.5 = 0.9699 · validator PASS (incl. --check-ids) · 5.4% empty rows · 3.38 matches per S1.

- `matching_results.tsv.gz` — the leaderboard file (gunzip before uploading)
- `candidate_pairs.tsv.gz.part00..02` — candidate set, split to stay under GitHub's 100 MB limit

Restore both TSVs:

```
python output/v1/restore.py
```

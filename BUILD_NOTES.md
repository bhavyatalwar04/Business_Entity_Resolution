# Submission build notes (branch `subs-final`)

Best leaderboard file: **`output/v16_v11_s-1.25/matching_results.tsv.gz`**, LB **0.986232** (27 Sep 2026, 19:2x IST).
md5 of the uncompressed TSV: `e8ba1a0fee1a186c2193a24570397aaa`.

It is the v11 recipe (LB 0.986201) with only the France decision moved: logit shift −1.25 instead of −1 on France
(unseen-country) pairs. India/US decisions are identical to the v11 recipe.

## Models (all MIT / Apache-2.0; total ≈ 6.16 B parameters)
| Component | Params | Source |
|---|---|---|
| multilingual-e5-small (fine-tuned blocker) | 0.12 B | `src/embed.py`, `src/finetune_embed.py` |
| xlm-roberta-base CE | 0.28 B | `handoff/ce_out` (branch `ce-results`) |
| xlm-roberta-large CE (`ce_large`) | 0.56 B | branch `ce-large` |
| xlm-roberta-large CE continued on full data (`ce_full`) | 0.56 B | branch `ce-full` |
| xlm-roberta-large France self-trained r1 (`ce_france`, swapped in for France) | 0.56 B | branch `ce-france` |
| Qwen3-4B-Base cross-encoder, LoRA r=32 (`qwen`) | 4.02 B + 0.07 B | branch `ce-qwen`, `qwen_ce/` (README there) |
| LightGBM ranker + LightGBM stacker | negligible | `src/run.py`, `src/stack_submit.py` |

## Exact build (run from the repo root, CPU only)
Inputs, checked out from their branches into `handoff/` (not tracked on `main`):
```
git fetch origin full-results ce-large ce-full ce-france ce-results ce-qwen
for bd in full-results:full_out ce-large:ce_large_out ce-full:ce_full_out ce-france:ce_france_out ce-results:ce_out ce-qwen:ce_qwen_out; do
  git checkout origin/${bd%%:*} -- handoff/${bd##*:} && git reset -q handoff/${bd##*:}; done
rm -f handoff/ce_qwen_out/*_rest*    # v11 coverage: Qwen3 fold-0 without the remainder file
```
Held-out check (India/US fold-0, cross-fitted halves) — expect 0.9903:
```
python -m src.stack_eval --pairs "handoff/full_out/oof_train_full_part*.parquet" \
  --extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=handoff/ce_qwen_out \
  --fill qwen=ce_full --compare_extras --only_all --tag v11
```
Submission (writes one folder per France shift; `-1.25` is the best):
```
python -m src.stack_submit --features extra \
  --extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=handoff/ce_qwen_out \
  --fill qwen=ce_full --pairs "handoff/full_out/oof_train_full_part*.parquet" \
  --lgbm_test "handoff/full_out/probs_model_full_part*.parquet" --swap ce_large=handoff/ce_france_out \
  --shift "france:-1.5,-1.25,-0.75,-0.5" --alpha 1.5 --out output/v16_v11
```
`scripts/hpc_build.sh` wraps the same steps (8 threads); `scripts/fr_diff.py` compares a file to a reference by country.
Threads: `N_CPUS=OMP_NUM_THREADS=RAYON_NUM_THREADS=MKL_NUM_THREADS=6..8`. LightGBM results vary slightly with thread
count / platform (~650 France pairs between the laptop and HPC builds of v11); India/US decisions were identical.

## Leaderboard log (27 Sep)
| File | Change vs v11 | LB |
|---|---|---|
| v11 (26 Sep) | — | 0.986201 |
| v13_q35_s-0.25 | Qwen3.5-4B replaces Qwen3-4B | 0.985613 |
| v14_ens2_s-0.5 | Qwen3 v11 + adapter #2 logit-averaged | 0.985777 |
| **v16_v11_s-1.25** | France shift −1 → −1.25 | **0.986232** |

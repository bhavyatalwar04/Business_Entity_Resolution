# Cross-encoder models (hpc-server part of the final submission)

Pairwise cross-encoders that score a (S1 record, pool record) pair as "same business" or not. Each outputs `ce_prob` per pair; the stacker (laptop) uses them as features next to LightGBM and the xlm-r CEs.
All models are fine-tuned on labelled **train** pairs only (fold != 0 for the held-out fold-0 scores). No test labels, no external data, no other teams' code.

## Models (all Apache-2.0)

| handoff dir | base model | params shipped | fine-tuning |
|---|---|---|---|
| `ce_qwen_out` | Qwen/Qwen3-4B-Base | 4.02B + LoRA 0.06B | LoRA r=32 (q,k,v,o,gate,up,down) + score head |
| `ce_q35_out` | Qwen/Qwen3.5-4B-Base, text-only copy | 4.206B + LoRA 0.06B | as above + DeltaNet projections (in_proj_qkv, in_proj_z, out_proj) |
| `ce_rr4b_out` | Qwen/Qwen3-Reranker-4B | 4.02B + LoRA 0.06B | as Qwen3; score head initialised from the reranker's yes/no LM rows |
| `ce_bge_out` | BAAI/bge-reranker-v2-m3 | 0.568B | full fine-tune |

Only ONE 4B family ships in the final build (8B total budget across all shipped models); see the final submission notes for which.

## Environment

- Python env with torch 2.9.1, transformers 5.5, peft 0.18.1 (`~/.conda/envs/minor-project`).
- Qwen3.5 only: `fla-core` + `flash-linear-attention` 0.5.2, and Triton >= 3.7.1 on H100 (Triton 3.5.1 gives wrong
  DeltaNet gradients on Hopper; we install 3.7.1 into `pylib_tri/` and put it on `PYTHONPATH` for Qwen3.5 jobs only).
- Base models are downloaded from Hugging Face into `~/models/`. Qwen3.5 checkpoints ship as vision-language models;
  `make_text_only.py` writes the text-only copy we use (vision tower and MTP head dropped).

## Files

| file | purpose |
|---|---|
| `prep.py` | builds `inputs/train.parquet` (1.2M train pairs, fold != 0), `fold0.parquet`, `test.parquet` (contested test pairs) |
| `prompts.py` | pair -> token ids: `plain` (Qwen3 / Qwen3.5), `reranker_short` (Qwen3-Reranker), `pair` (bge) |
| `qwen_ce.py` | trains one CE for 1 epoch (LoRA or `--full_ft`), then scores fold-0 and the contested test pairs |
| `qwen_score.py` | scores any pair list (e.g. all France test pairs) with a trained model |
| `make_text_only.py` | Qwen3.5 VLM checkpoint -> text-only checkpoint |
| `pack_model.py`, `deliver_model.sh` | pack scores into the handoff format (+ fold-0 comparison vs Qwen3) and push |
| `run_train.pbs`, `run_rr.pbs`, `run_qwen_score.pbs` | PBS jobs (1 H100 each model) |

## Reproduce

```bash
python prep.py
# Qwen3-4B (v11)
python qwen_ce.py --out out
python qwen_score.py --adapter out/adapter --pairs inputs/france_all.parquet --split test --stem ce_test_france --out out_scores
# Qwen3.5-4B
python make_text_only.py ~/models/Qwen3.5-4B-Base ~/models/Qwen3.5-4B-Base-text
PYTHONPATH=pylib_tri python qwen_ce.py --model ~/models/Qwen3.5-4B-Base-text --out out_q35 --accum 2 \
    --target_modules q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj+in_proj_qkv+in_proj_z+out_proj
# Qwen3-Reranker-4B
python qwen_ce.py --model ~/models/Qwen3-Reranker-4B --prompt reranker_short --init_yesno --max_len 160 --accum 2 --out out_rr4b
# bge-reranker-v2-m3
python qwen_ce.py --model ~/models/bge-reranker-v2-m3 --prompt pair --full_ft --lr 2e-5 --save_every 5000 --out out_bge
```
Each `qwen_score.py` call takes the same `--model` / `--prompt` / `--max_len` as its training run.

Recipe for every model: 1 epoch over the 1.2M train pairs, batch 64, AdamW (LoRA lr 1e-4, full fine-tune 2e-5), linear
schedule with 3% warm-up, max length 128 tokens (160 for the reranker template), bf16 autocast, seed 7.

## Status at the 27 Sep 2026 freeze

Final submission family: **v11** = v8 stack + ONE Qwen3-4B-Base LoRA adapter (`ce_qwen_out`, adapter in
`final_models/qwen3_v11_adapter`). Total shipped params ~6.16B (v8 2.08B + Qwen3-4B 4.02B + LoRA 0.06B), under 8B.
LB 0.986201. Any later file that beats it is v11 with a France-only change, documented in the build notes.

**FINAL: `output/v15_frself_s-1.25` (laptop build, unzipped md5 8b68aed84c39980832ca283d1f050748), LB 0.986509**
(27 Sep 21:03 IST, +0.000308 over v11; implied France 0.9629 -> 0.9650; India/US byte-identical to v11).
Same as v11 except that the Qwen feature on **France pairs only** comes from a France self-trained adapter, with France
shift `france:-1.25` (count-matched). India/US Qwen scores still come from the v11 adapter.
Shipped models: v8 stack (2.08B) + Qwen3-4B-Base (4.02B, one shared copy) + v11 LoRA (0.06B, `final_models/qwen3_v11_adapter`)
+ France self-train LoRA (0.06B, `final_models/qwen3_frself_adapter`, adapter_model.safetensors md5
ff0c2b44ede8c3f5db70c4570ba30cd3) = ~6.22B, under 8B.

France self-train adapter, reproduce (all from train labels + unlabelled test text; no test labels, no external data):
```bash
python prep_selftrain.py       # 3-teacher (lgbm_full, ce_full, v11 Qwen) one-to-one France pseudo-labels + equal IN/US labelled replay
python prep_selftrain_sib.py   # + sibling hard negatives -> inputs/selftrain_fr.parquet
# continue the v11 adapter 1 epoch, plain source-only training (domain head off), lr 1e-5  [run_frself.pbs]
python qwen_dann.py --adapter out/adapter --max_lambda 0 --src_file inputs/selftrain_fr.parquet --target gate_fr_50k.parquet \
    --lr 1e-5 --bs 64 --accum 2 --out out_frself
# score every France test pair  [run_frselfsc.pbs]
python qwen_score.py --adapter out_frself/adapter --pairs inputs/france_all.parquet --split test --stem ce_test_france --out out_frself_scores
python pack_frcanon.py out_frself_scores handoff ce_qwen_frself_out   # v11 Qwen files with France scores replaced
```
Stacker/decision: the v11 recipe with `--extra qwen=handoff/ce_qwen_frself_out --shift france:-1.25`
(`subs-final` `BUILD_NOTES.md`). Note: on our internal gate this adapter missed by 0.0001 (IN/US contested AUC -0.0011);
it is used for France pairs only, so India/US output is unaffected.

Previous best (now superseded): **`subs-final` `output/v16_v11_s-1.25`, LB 0.986232** (27 Sep 19:27 IST, +0.000031 over v11).
It is the v11 recipe unchanged except for the France decision shift: exactly the same models and weights,
stacker, and India/US output as v11. Build config (`output/v16_v11_s-1.25/blend.json`):
stacker `stack_extra` with extra CEs `ce_large` (swapped for `handoff/ce_france_out` on France), `ce_full`, and
`qwen` = `handoff/ce_qwen_out` (adapter `final_models/qwen3_v11_adapter`); decision `expf`, alpha 1.5,
one-to-one; shift `france:-1.25` (v11 used `france:-1`). Built on the HPC. The unzipped `matching_results.tsv`
has md5 e8ba1a0fee1a186c2193a24570397aaa and 1,732,544 rows. Shipped params are the same as v11 (~6.16B).
Stacker and decision commands for this file are in `BUILD_NOTES.md` on branch `subs-final` (commit 2807e9c).

Tested on 26-27 Sep and NOT used (measured, with the number that decided it):

| experiment | code here | result |
|---|---|---|
| Qwen3.5-4B (text-only) replacing Qwen3 | `make_text_only.py`, `run_train.pbs` | LB 0.985613 (France -0.0034) |
| Qwen3.5 adapter #2 | `run_second.pbs` | contested AUC 0.8987 < 0.9025 |
| Qwen3-Reranker-4B (yes/no head init) | `prompts.py` reranker_short, `run_rr.pbs` | contested 0.9027 = Qwen3 level |
| bge-reranker-v2-m3 (full fine-tune) | `run_rr.pbs` | stacker held-out 0.9902 < 0.9903 |
| Qwen3 adapter #2 (train_seed2) ensemble | `run_second.pbs` | LB 0.985777 (France -0.0028) |
| Qwen3 adapter #3 (train_seed3) | `prep_seed3.py` | contested 0.8680; hurts every ensemble |
| France text canonicalisation (fr_canon.py v1/v2/keep) | `run_frcanon.pbs`, `gate_frcanon.py` | all variants FAIL the gate |
| TransClean triangle S2xS3 consistency | `run_tri.pbs`, `tri_sim.py` | stacker-level: no threshold beats no-filter |
| France self-training (3-teacher pseudo-labels + sibling negatives) | `prep_selftrain*.py`, `qwen_dann.py --max_lambda 0`, `run_frself*.pbs` | lr 1e-5: gate borderline (IN/US contested -0.0011); lr 3e-5: FAIL (-0.0022) |

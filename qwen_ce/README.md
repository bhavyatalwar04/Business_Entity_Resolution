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

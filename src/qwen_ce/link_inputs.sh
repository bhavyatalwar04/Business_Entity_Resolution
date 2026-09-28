#!/bin/bash
# Fill inputs/ from the stage-1 outputs in <package root>/handoff (run from src/qwen_ce):
#   handoff/full_out    : LightGBM full-data model, oof_train_full_part* (train OOF) + probs_model_full_part* (test)
#   handoff/ce_full_out : xlm-roberta-large CE continued on full data (fold-0 OOF + test scores)
#   handoff/ce          : train_pairs_part* / test_pairs_part* (candidate pairs with lgbm_prob and fold)
# then copy the two 50k-row gate files (data/) that run_frself.pbs uses.
set -e
H=${BER_HANDOFF:-../../handoff}
mkdir -p inputs/full inputs/ce_full
ln -sf $(realpath $H/full_out)/oof_train_full_part*.parquet $(realpath $H/full_out)/probs_model_full_part*.parquet inputs/full/
ln -sf $(realpath $H/ce_full_out)/ce_*_part*.parquet inputs/ce_full/
ln -sf $(realpath $H/ce)/train_pairs_part*.parquet $(realpath $H/ce)/test_pairs_part*.parquet inputs/
cp data/gate_fr_50k.parquet data/gate_fold0_50k.parquet inputs/
ls inputs inputs/full inputs/ce_full | head -40

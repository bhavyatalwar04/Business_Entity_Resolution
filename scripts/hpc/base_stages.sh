#!/bin/bash
#PBS -N base_stages
#PBS -j oe
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
cd $PBS_O_WORKDIR
exec > logs/base_stages.log 2>&1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
module load cuda
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
set -e
source src/xlmr_ce/jobs/pick_gpu.sh
run() { echo "=== $(date '+%F %T') $*"; "$@"; }
run python -m src.finetune_embed --pairs 400000 --max_steps 2500 --out artefacts/embed_ft
run python -m src.run --stage embed --split train
run python -m src.run --stage embed --split test
run python -m src.run --stage block --split train
run python -m src.run --stage block --split test
echo "=== $(date '+%F %T') ALL DONE"

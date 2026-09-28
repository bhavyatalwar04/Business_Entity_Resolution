#!/bin/bash
#PBS -N run_full
#PBS -j oe
#PBS -q workq
#PBS -l select=1:ncpus=16:mem=96gb
#PBS -l walltime=24:00:00
cd $PBS_O_WORKDIR
exec > logs/full.log 2>&1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
set -e
run() { echo "=== $(date '+%F %T') $*"; "$@"; }
run python -m src.run --stage features --split train --config configs/full.yaml
run python -m src.run --stage rank     --split train --config configs/full.yaml
run python -m src.run --stage features --split test  --config configs/full.yaml
run python -m src.run --stage rank     --split test  --config configs/full.yaml
run python -m src.run --stage decide   --split test  --config configs/full.yaml
echo "=== $(date '+%F %T') ALL DONE"

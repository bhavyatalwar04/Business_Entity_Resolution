#!/bin/bash
#PBS -N ce_full_lg
#PBS -j oe
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=48gb
#PBS -l walltime=10:00:00
cd $PBS_O_WORKDIR
exec > logs/ce_full.log 2>&1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
export N_CPUS=${NCPUS:-8} OMP_NUM_THREADS=${NCPUS:-8} RAYON_NUM_THREADS=${NCPUS:-8} MKL_NUM_THREADS=${NCPUS:-8}
module load cuda
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
source src/xlmr_ce/jobs/pick_gpu.sh
PYTHONPATH=. python -m src.xlmr_ce.ce_full --model artefacts/ce_large/final --lr 5e-6 --bs 128 --cap 4000000 --ckpt artefacts/ce_full_large --out handoff/ce_full_out

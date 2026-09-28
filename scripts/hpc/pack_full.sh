#!/bin/bash
#PBS -N pack_full
#PBS -j oe
#PBS -q workq
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=02:00:00
# Packs step-3 outputs into < 90 MB pieces under handoff/full_out/ for the full-results push.
cd $PBS_O_WORKDIR
exec > logs/pack_full.log 2>&1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
export N_CPUS=${NCPUS:-8} OMP_NUM_THREADS=${NCPUS:-8} RAYON_NUM_THREADS=${NCPUS:-8} MKL_NUM_THREADS=${NCPUS:-8}
set -e
PYTHONPATH=. python scripts/hpc/pack_full.py
ls -la handoff/full_out

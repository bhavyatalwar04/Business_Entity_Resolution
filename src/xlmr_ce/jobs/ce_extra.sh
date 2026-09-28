#!/bin/bash
#PBS -N ce_extra
#PBS -j oe
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=48gb
#PBS -l walltime=06:00:00
cd $PBS_O_WORKDIR
exec > logs/ce_extra.log 2>&1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
export N_CPUS=${NCPUS:-8} OMP_NUM_THREADS=${NCPUS:-8} RAYON_NUM_THREADS=${NCPUS:-8} MKL_NUM_THREADS=${NCPUS:-8}
module load cuda
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
source src/xlmr_ce/jobs/pick_gpu.sh
PYTHONPATH=. python -m src.xlmr_ce.ce_extra --ckpts base=artefacts/ce_base/final:handoff/ce_out large=artefacts/ce_large/final:handoff/ce_large_out france=artefacts/ce_france/final:handoff/ce_france_out

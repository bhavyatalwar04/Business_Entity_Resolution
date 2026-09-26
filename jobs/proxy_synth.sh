#!/bin/bash
#PBS -N proxy_synth
#PBS -j oe
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=04:00:00
cd $PBS_O_WORKDIR
exec > logs/proxy_synth.log 2>&1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
export N_CPUS=${NCPUS:-8} OMP_NUM_THREADS=${NCPUS:-8} RAYON_NUM_THREADS=${NCPUS:-8} MKL_NUM_THREADS=${NCPUS:-8}
module load cuda
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
source jobs/pick_gpu.sh
PYTHONPATH=. python jobs/proxy_synth.py --out handoff/proxy_synth_out --ckpt artefacts/proxy_synth

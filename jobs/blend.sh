#!/bin/bash
#PBS -N blend
#PBS -j oe
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=04:00:00
cd $PBS_O_WORKDIR
exec > logs/blend.log 2>&1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
export PYTHONUNBUFFERED=1
python -m src.blend_submit

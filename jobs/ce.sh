#!/bin/bash
#PBS -N ce
#PBS -j oe
#PBS -q gpu
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
# qsub -v MODEL=FacebookAI/xlm-roberta-base,TAG=base,OUT=handoff/ce_out -o logs/ce_base.log jobs/ce.sh
cd $PBS_O_WORKDIR
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
module load cuda
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
nvidia-smi -L | head -1
python -m src.ce_rescore --model ${MODEL} --ckpt artefacts/ce_${TAG} --out ${OUT} ${EXTRA}

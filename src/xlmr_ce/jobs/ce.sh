#!/bin/bash
#PBS -N ce
#PBS -j oe
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
# qsub -v MODEL=FacebookAI/xlm-roberta-base,TAG=base,OUT=handoff/ce_out -o logs/ce_base.log jobs/ce.sh
cd $PBS_O_WORKDIR
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ber
export N_CPUS=${NCPUS:-8} OMP_NUM_THREADS=${NCPUS:-8} RAYON_NUM_THREADS=${NCPUS:-8} MKL_NUM_THREADS=${NCPUS:-8}
module load cuda
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
source src/xlmr_ce/jobs/pick_gpu.sh
python -m src.xlmr_ce.ce_rescore --model ${MODEL} --ckpt artefacts/ce_${TAG} --out ${OUT} ${EXTRA}

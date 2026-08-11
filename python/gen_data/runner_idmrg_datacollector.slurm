#!/bin/bash
#SBATCH --job-name=idmrg_corrs
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --partition=cmsp,long
#SBATCH --time=04:00:00
#SBATCH --output=logs/idmrg_corrs_%A_%a.out
#SBATCH --mail-user=ashankar@ictp.it
#SBATCH --mail-type=ALL


mkdir -p logs

TASK_COUNT=8   

module load python/3.11.9 #check what the module is on argo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
source ../.venv/bin/activate


echo "Running task ${SLURM_ARRAY_TASK_ID} of ${TASK_COUNT} on $(hostname)"

python driver_idmrg_datacollector.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --task-count "${TASK_COUNT}"

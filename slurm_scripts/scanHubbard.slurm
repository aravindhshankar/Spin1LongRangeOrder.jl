#!/bin/bash
#
#SBATCH --job-name=fixedN_scanHubbard
#SBATCH --partition=cmsp,long
# SBATCH --array=1-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=ashankar@ictp.it
#SBATCH --mail-type=ALL
#SBATCH --export=PATH

mkdir -p logs

module load mkl
module load intel

export JULIA_NUM_THREADS=1

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OPENBLAS_NUM_THREADS=1
echo "Starting sumulation with id ${SLURM_ARRAY_TASK_ID}"
echo "Number of CPUs = ${SLURM_CPUS_PER_TASK}"

#should submit from slurm_scripts directory
cd ..
echo "Should now be in root directory : $(pwd)"
# julia --project=. scripts/scanFmHubbard.jl
julia --project=. scripts/fixedNpScanHubbard.jl

echo "FINISHED, exiting slurm"

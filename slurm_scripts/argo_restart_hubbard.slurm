#!/bin/bash
#SBATCH --job-name=restartHubbard
#SBATCH --output=logs/resHub_%A_%a.out
#SBATCH --array=1-9
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=cmsp,long
#SBATCH --mem=30G
#SBATCH --mail-user=ashankar@ictp.it
#SBATCH --mail-type=ALL
#SBATCH --export=PATH

# Load any required modules here
module load mkl
module load intel
echo "Module Load steps compelete"

export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OPENBLAS_NUM_THREADS=1
echo "Starting sumulation with id ${SLURM_ARRAY_TASK_ID}"
echo "Number of CPUs = ${SLURM_CPUS_PER_TASK}"

#should submit from slurm_scripts directory
cd ..
echo "Should now be in root directory : $(pwd)"
julia --project=. scripts/HubbardImprovePrecision.jl

echo "FINISHED, exiting slurm"

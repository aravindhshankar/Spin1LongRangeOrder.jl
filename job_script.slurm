#!/bin/bash
#
#SBATCH --job-name=testDMRG
#SBATCH --partition=cmsp,long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

module load mkl
module load intel

export JULIA_NUM_THREADS=1
# MKL uses the 6 CPUs you allocated
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

julia --project=. example/eval.jl

echo "FINISHED, exiting slurm"

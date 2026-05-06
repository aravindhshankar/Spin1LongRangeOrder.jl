#!/bin/bash
#
#SBATCH --job-name=testDMRG
#SBATCH --partition=cmsp,long
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=ashankar@ictp.it
#SBATCH --mail-type=ALL

mkdir -p logs

module load mkl
module load intel

export JULIA_NUM_THREADS=1
# MKL uses the 6 CPUs you allocated
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ..
julia --project=. scripts/scang2.jl

echo "FINISHED, exiting slurm"

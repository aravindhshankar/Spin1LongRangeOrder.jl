#!/bin/bash
#
#SBATCH --job-name=scan_g2_fixed_g1_N
#SBATCH --partition=cmsp,long
#SBATCH --array=1-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=ashankar@ictp.it
#SBATCH --mail-type=ALL

mkdir -p logs

module load mkl
module load intel

export JULIA_NUM_THREADS=1

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OPENBLAS_NUM_THREADS=1

cd ..
julia --project=. scripts/scan_g2_fixed_g1_N.jl

echo "FINISHED, exiting slurm"

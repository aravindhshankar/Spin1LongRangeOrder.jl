#!/bin/bash
#
#SBATCH --job-name=testDMRG
#SBATCH --partition=cmsp,long
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=12:00:00
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
julia --project=. scripts/scan_g1_g2.jl

echo "FINISHED, exiting slurm"

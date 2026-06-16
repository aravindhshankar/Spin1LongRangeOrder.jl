#!/bin/bash
#
#SBATCH --job-name=testIDMRG
#SBATCH --partition=cmsp,long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

module load mkl
module load intel
module load python #check what the module is on argo

# export MKL_NUM_THREADS=4 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source .venv/bin/activate #make venv first
python scanner.py

echo "FINISHED, exiting slurm"

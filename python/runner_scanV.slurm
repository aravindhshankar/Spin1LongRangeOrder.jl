#!/bin/bash
#
#SBATCH --job-name=chi20IDMRG
#SBATCH --partition=cmsp,long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#
#SBATCH --mem=2G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

mkdir -p logs

module load python/3.11.9 #check what the module is on argo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source .venv/bin/activate #make venv first

echo "------------------iDMRG started--------------------"
python3 scanner_scanV_serial.py
echo "------------------iDMRG ended--------------------"
echo "FINISHED, exiting slurm"

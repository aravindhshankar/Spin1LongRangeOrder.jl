#!/bin/bash
#
#SBATCH --job-name=scanreIDMRG
#SBATCH --partition=cmsp,long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3
#SBATCH --mem=4G
#SBATCH --time=23:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashankar@ictp.it

mkdir -p logs

module load python/3.11.9 #check what the module is on argo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source .venv/bin/activate #make venv first

echo "------------------iDMRG started--------------------"
python3 scanner_scanV_serial.py
echo "------------------iDMRG ended--------------------"
echo "FINISHED, exiting slurm"

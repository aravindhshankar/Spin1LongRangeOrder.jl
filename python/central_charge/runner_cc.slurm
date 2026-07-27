#!/bin/bash
#
#SBATCH --job-name=centralCharge
#SBATCH --partition=cmsp,long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-1
#SBATCH --mem=5G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.out

mkdir -p logs

module load python/3.11.9 #check what the module is on argo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ../.venv/bin/activate # venv is in python root 

#files=(
#    "freeFermion_cc.py"
#    "central_charge_Hubbard.py"
#)
#echo " running file ${files[$SLURM_ARRAY_TASK_ID]}"
echo "------------------iDMRG started--------------------"
#python3 "${files[$SLURM_ARRAY_TASK_ID]}"
python3 fm_central_charge.py
echo "------------------iDMRG ended--------------------"
echo "FINISHED, exiting slurm"

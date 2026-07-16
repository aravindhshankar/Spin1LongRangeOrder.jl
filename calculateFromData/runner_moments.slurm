#!/bin/bash
#SBATCH --job-name=momentsBinder
#SBATCH --array=0-7      
#SBATCH --cpus-per-task=6           
#SBATCH --mem=12G
#SBATCH --partition=cmsp,long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashankar@ictp.it
#SBATCH --time=2:00:00
#SBATCH --output=logs/moments_%A_%a.out

export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd .. #should be in project root now
julia --project=. calculateFromData/driver_moments.jl

#!/bin/bash
#SBATCH --job-name=corrmats
#SBATCH --array=0-4                 # exactly 5 tasks
#SBATCH --cpus-per-task=24          # 6 threads * 4 MKL threads 
#SBATCH --mem=16G
#SBATCH --partition=cmsp,long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashankar@ictp.it
#SBATCH --time=10:00:00              #each task now does nfiles/5 runs
#SBATCH --output=logs/corr_%A_%a.out


export JULIA_NUM_THREADS=6
export MKL_NUM_THREADS=4

cd .. #should be in project root now
julia --project=. calculateFromData/driver.jl
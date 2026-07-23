#!/bin/bash
#SBATCH --job-name=corrmats
#SBATCH --array=0-4                 # exactly 5 tasks
#SBATCH --cpus-per-task=10          # 5 threads * 2 MKL threads 
#SBATCH --mem=8G
#SBATCH --partition=cmsp,long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashankar@ictp.it
#SBATCH --time=1:00:00              #each task now does nfiles/5 runs
#SBATCH --output=logs/corr_%A_%a.out


export JULIA_NUM_THREADS=5
export MKL_NUM_THREADS=2

cd .. #should be in project root now
julia --project=. calculateFromData/driver.jl
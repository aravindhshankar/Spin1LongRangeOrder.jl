#!/bin/bash
#SBATCH --job-name=dyn_corr
#SBATCH --output=logs/dyn_corr_%A_%a.out
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cmsp,long

mkdir -p logs

module load mkl
module load intel
echo "Module Load steps compelete"

PARAMS_FILE=${PARAMS_FILE:-job_params.txt}
TF=${TF:-1.0}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")
read -r N t U Vpp Vpm OP <<< "$LINE"


export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

cd .. # go to project root
echo "Should now be in root directory : $(pwd)"

julia --project=. julia_tebd/run_dynamical_correlations.jl "$N" "$t" "$U" "$Vpp" "$Vpm" "$OP" "$TF"

echo "FINISHED, exiting slurm"
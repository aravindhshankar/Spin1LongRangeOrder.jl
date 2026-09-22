#!/bin/bash
#SBATCH --job-name=dyn_corr
#SBATCH --output=logs/dyn_corr_%A_%a.out
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

# MKL keeps its own internal scratch-buffer pool and does not return it to
# the OS by default; over a long-running TEBD loop that pool alone can
# account for several GB of RSS that never shrinks. Disabling it makes MKL
# malloc/free normally so memory is actually released between steps.
export MKL_DISABLE_FAST_MM=1

cd .. # go to project root
echo "Should now be in root directory : $(pwd)"

# Julia's GC paces itself off perceived system memory pressure, which a
# SLURM cgroup memory cap doesn't create until the limit is nearly hit --
# so raising --mem alone doesn't make the GC collect any sooner. Pointing
# --heap-size-hint below the cgroup limit tells Julia to start collecting
# well before that, instead of ballooning RSS until the OOM killer fires.
HEAP_HINT_GB=$(( ${SLURM_MEM_PER_NODE:-65536} * 85 / 100 / 1024 ))
julia --project=. --heap-size-hint=${HEAP_HINT_GB}G julia_tebd/run_dynamical_correlations.jl "$N" "$t" "$U" "$Vpp" "$Vpm" "$OP" "$TF"

echo "FINISHED, exiting slurm"
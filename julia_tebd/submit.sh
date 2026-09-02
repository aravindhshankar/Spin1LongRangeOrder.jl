#!/bin/bash
# Usage: ./submit.sh [params_file] [tf] [max_concurrent]
PARAMS_FILE=${1:-job_params.txt}
TF=${2:-1.0}
MAX_CONCURRENT=${3:-10}

NLINES=$(wc -l < "$PARAMS_FILE")
sbatch --array=1-${NLINES}%${MAX_CONCURRENT} --export=PARAMS_FILE="$PARAMS_FILE",TF="$TF" submit_dyn_corr.slurm

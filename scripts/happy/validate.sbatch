#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=happy_validate
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/happy_validate_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/happy_validate_%A_%a.err

#==============================================================================
# Score happy against the pulse recording, one run per array task
#==============================================================================
# Waveform agreement + BOLD variance (validate.py --fmriprep), for runs that
# happy.sbatch has processed in `pure` mode. Each task writes one row to
# <rows_dir>/<run>.tsv; collect afterwards with
#   validate.py --collect <rows_dir> --out <table.tsv>
#
# Usage:
#   sbatch --array=1-N validate.sbatch <runs.txt> <happy_out_dir> <rows_dir>
#
#   runs.txt       the list given to happy.sbatch
#   happy_out_dir  happy.sbatch's <out_dir> (reads <out_dir>/pure, and
#                  <out_dir>/ref when present)
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

RUNS=${1:?usage: validate.sbatch <runs.txt> <happy_out_dir> <rows_dir>}
HAPPY=${2:?usage: validate.sbatch <runs.txt> <happy_out_dir> <rows_dir>}
ROWS=${3:?usage: validate.sbatch <runs.txt> <happy_out_dir> <rows_dir>}

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "$REPO_ROOT/scripts/load_config.sh"
set -e

ENV=/gpfs/projects/hulacon/shared/envs/rapidtide
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
REL=$(sed -n "${TASK}p" "$RUNS")
[[ -n "$REL" ]] || { echo "no line $TASK in $RUNS" >&2; exit 2; }
BASE=$(basename "${REL%_bold.nii.gz}")

REF_ARGS=()
[[ -d "$HAPPY/ref" ]] && REF_ARGS=(--ref "$HAPPY/ref")

echo "Job ${SLURM_JOB_ID}_${TASK} on $(hostname), started $(date -Is): $BASE"
"$ENV/bin/python" "$REPO_ROOT/scripts/happy/validate.py" \
  --bids "$BIDS_DIR" --happy "$HAPPY/pure" "${REF_ARGS[@]}" \
  --fmriprep "$OUTPUT_DIR/fmriprep" \
  --runs "$RUNS" --index "$TASK" --out "$ROWS/$BASE.tsv"
echo "finished $(date -Is)"

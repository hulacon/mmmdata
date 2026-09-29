#!/bin/bash
#SBATCH --job-name=loc_splithalf
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/loc_splithalf_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/loc_splithalf_%A_%a.err

#==============================================================================
# Per-run + split-half localizer maps on fsaverage6 (scripts/functional_space/
# localizer_splithalf.py). One array task per (subject, model) line of UNITS.
#
# Usage:
#   sbatch --array=0-5 --export=ALL,UNITS="sub-##:floc sub-##:motor ..." localizer_splithalf.sbatch
# Sizing: fsaverage6 is ~82k vertices, below the 2 mm MNI grid the
# glm_contrast_maps.sbatch envelope (16 G / 1 h) was measured on.
#==============================================================================

set -euo pipefail
: "${UNITS:?set UNITS (space-separated subject:model pairs) via --export}"

# shellcheck disable=SC2206
UNIT_ARR=(${UNITS})
UNIT="${UNIT_ARR[${SLURM_ARRAY_TASK_ID}]}"
SUBJECT="${UNIT%%:*}"
MODEL="${UNIT##*:}"

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}: ${SUBJECT} ${MODEL} — $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
# No `module load` after this line: it would replace the venv's python.
PY="${REPO_ROOT}/.venv/bin/python"

"${PY}" "${REPO_ROOT}/scripts/functional_space/localizer_splithalf.py" --subject "${SUBJECT}" --model "${MODEL}"

echo "End: $(date)"

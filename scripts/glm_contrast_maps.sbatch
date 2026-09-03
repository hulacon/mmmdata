#!/bin/bash
#SBATCH --job-name=glm_contrast_maps
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glm_contrast_maps_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glm_contrast_maps_%j.err

#==============================================================================
# Condition-level GLM contrast maps — one subject, one BIDS Stats Model
#==============================================================================
# Design record: mmmdata-agents/docs/workbench/glm-strategy/. Runner:
# scripts/glm_contrast_maps.py; models in models/model-*_smdl.json.
#
# Sizing is a guess until the first fit (cluster-reentry R15): a 6-run fLoc
# subject is ~1,250 volumes at 2 mm MNI; nilearn holds one run at a time with
# minimize_memory off, so peak is one run's data plus its residuals. Revise
# from seff after the pilot.
#
# Usage (SUBJECT and MODEL are required; SESSIONS and EXTRA are optional):
#   sbatch --export=ALL,SUBJECT=sub-03,MODEL=motor,SESSIONS="ses-30" glm_contrast_maps.sbatch
#   sbatch --export=ALL,SUBJECT=sub-03,MODEL=floc glm_contrast_maps.sbatch
#   sbatch --export=ALL,SUBJECT=sub-03,MODEL=floc,EXTRA="--noise-model ols" glm_contrast_maps.sbatch
#==============================================================================

set -euo pipefail

: "${SUBJECT:?set SUBJECT (e.g. sub-03) via --export}"
: "${MODEL:?set MODEL (e.g. floc, motor, auditory) via --export}"
SESSIONS="${SESSIONS:-}"
EXTRA="${EXTRA:-}"

echo "=========================================="
echo "GLM contrast maps — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Model: ${MODEL}"
echo "Sessions: ${SESSIONS:-<all>}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

SESSION_ARGS=()
if [[ -n "${SESSIONS}" ]]; then
    # shellcheck disable=SC2206
    SESSION_ARGS=(--sessions ${SESSIONS})
fi

# shellcheck disable=SC2086
python "${REPO_ROOT}/scripts/glm_contrast_maps.py" \
    --subject "${SUBJECT}" \
    --model "${MODEL}" \
    "${SESSION_ARGS[@]}" \
    ${EXTRA}

echo ""
echo "End Time: $(date)"

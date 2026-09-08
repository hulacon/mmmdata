#!/bin/bash
#SBATCH --job-name=glm_bakeoff
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glm_bakeoff_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glm_bakeoff_%A_%a.err

#==============================================================================
# GLM estimator bake-off — one cell per array task
#==============================================================================
# Design record: mmmdata-agents/docs/workbench/glm-strategy/. Runner:
# scripts/glm_bakeoff.py; the units file is written by `glm_bakeoff.py plan`
# into <derivatives>/glm_bakeoff/units.txt (one "sub-XX cell-id" per line).
#
# Sizing is a guess until the pilot: the production contrast-map runner
# measured 3.7-5.3 G / 1-2 min for fLoc and motor with nilearn; TBencoding is
# 42 runs of 210 volumes, and the remlfit engine (ARMA(1,1) per voxel, 8 OpenMP
# threads) is the slow one — a voxelwise-HRF x remlfit x TBencoding cell runs
# 42 x 20 3dREMLfit calls. 8 h is the envelope for that corner; revise from
# seff after the pilot units.
#
# Usage:
#   python scripts/glm_bakeoff.py plan            # freeze harness, write units.txt
#   N=$(wc -l < /gpfs/projects/hulacon/shared/mmmdata/derivatives/glm_bakeoff/units.txt)
#   sbatch --array=1-${N}%24 scripts/glm_bakeoff.sbatch
#   sbatch --array=5 --export=ALL,EXTRA="--force" scripts/glm_bakeoff.sbatch
#==============================================================================

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array; each task fits one unit of units.txt}"
EXTRA="${EXTRA:-}"

echo "=========================================="
echo "GLM bake-off — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"
module load afni/24.1.22 2>/dev/null || true
command -v 3dREMLfit >/dev/null || export PATH="/packages/afni/24.1.22:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TMPDIR="${TMPDIR:-/tmp}"

# shellcheck disable=SC2086
python "${REPO_ROOT}/scripts/glm_bakeoff.py" fit --unit "${SLURM_ARRAY_TASK_ID}" ${EXTRA}

echo ""
echo "End Time: $(date)"

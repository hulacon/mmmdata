#!/bin/bash
#SBATCH --job-name=glm_bakeoff_prep
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glm_bakeoff_prep_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glm_bakeoff_prep_%j.err

#==============================================================================
# GLM bake-off prep — one subject: HRFindex NIfTI + glmsingle-betas arm
#==============================================================================
# Loads the subject's encoding GLMsingle TYPED output whole (a pickled dict,
# ~12 GB on disk: betasmd is 97x115x97x2562 float32) — hence 64 G, not the
# array's 32 G. Writes the HRFindex NIfTI the per-voxel HRF arm reads and the
# first-vs-later Welch maps + scores for the glmsingle-betas cell.
#
# Usage:
#   sbatch --export=ALL,SUBJECT=sub-03 scripts/glm_bakeoff_prep.sbatch
#==============================================================================

set -euo pipefail

: "${SUBJECT:?set SUBJECT (e.g. sub-03) via --export}"

echo "=========================================="
echo "GLM bake-off prep — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python "${REPO_ROOT}/scripts/glm_bakeoff.py" prep --subject "${SUBJECT}"

echo ""
echo "End Time: $(date)"

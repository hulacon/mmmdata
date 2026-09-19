#!/bin/bash
#SBATCH --job-name=prf_postfit
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/prf_postfit_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/prf_postfit_%A_%a.err

#==============================================================================
# After a pooled pRF fit lands: project to fsnative + rebuild viewer bundles —
# one array task per SUBJECT (both polarities)
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/prf-retinotopy/
#   Step 4 of the analyzePRF rerun (cluster-reentry M10). Runs after
#   fit_prf_analyzeprf.sbatch, whose assembler has already moved the previous
#   fsnative maps aside with the volumes, so the projection here writes fresh
#   files; the viewer bundles under <unit>/viz/ are regenerated in place (they
#   are views of the maps, not results, and the stale-bundle reap covers the
#   rest).
#
# Array task mapping: 1: sub-03  2: sub-04  3: sub-05  4: sub-06  5: sub-07
#
# Usage:
#   sbatch --array=1-5 --dependency=afterok:<fit array job id> prf_postfit.sbatch
#   sbatch --array=2 prf_postfit.sbatch     # one subject, e.g. after a refit
#==============================================================================

set -u
SUBJECTS=(03 04 05 06 07)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "=========================================="
echo "pRF post-fit (project + viewer) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}   Array Task ID: ${SLURM_ARRAY_TASK_ID}   Node: ${SLURMD_NODENAME}"
echo "Subject: sub-${SUBJECT}   Unit: pooled (derivatives/prf)"
echo "Start Time: $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

echo ""; echo "--- project to fsnative (both polarities)"
python "${REPO_ROOT}/scripts/project_prf_fsnative.py" --subject "${SUBJECT}" --polarity both || exit 1

echo ""; echo "--- viewer bundles (volume + both hemispheres)"
python "${REPO_ROOT}/scripts/build_brain_viewer.py" prf --subject "${SUBJECT}" --mode both --hemi both
EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="
exit ${EXIT_STATUS}

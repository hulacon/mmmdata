#!/bin/bash
#SBATCH --job-name=ps_similarity
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/ps_similarity_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/ps_similarity_%j.err

#==============================================================================
# Pattern-similarity Phase 6 — similarity computation (full grid, single job)
#==============================================================================
# Requires Phase 3-5 caches under derivatives/pattern_similarity/cache/.
# Plan: docs/doc/pattern-similarity-plan.md, Phase 6.
#==============================================================================

echo "=========================================="
echo "ps_similarity — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/pattern_similarity/similarity.py

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}

#!/bin/bash
#SBATCH --job-name=glmsingle_tb_pooled
#SBATCH --partition=memorylong
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tb_pooled_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tb_pooled_%A_%a.err

#==============================================================================
# GLMsingle TB — pooled arm (one fit: enc + ret-image + ret-word subgroups)
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/retrieval-modeling/ (3x2 redesign,
# 2026-08-26). Siloed arms run separately: glmsingle_tb_siloed.sbatch.
#
# Sizing: ~98 runs x ~150-207 vols = ~17k volumes and ~3,000 design columns —
# roughly 2x the 24.1.1 encoding fit that peaked at 200 GB / 18.3 h on 8 CPUs
# (job 43272306). Extrapolated ~400 GB / ~36 h, hence memorylong at
# 500 G / 72 h. UNMEASURED until the first pooled fit; revise from seff before
# submitting the remaining subjects.
#
# Array task mapping:
#   1: sub-03/pooled   2: sub-04/pooled   3: sub-05/pooled
#
# Usage:
#   sbatch --array=1 glmsingle_tb_pooled.sbatch      # pooled pilot: sub-03
#   sbatch --array=2-3 glmsingle_tb_pooled.sbatch    # after seff review
#==============================================================================

SUBJECTS=(sub-03 sub-04 sub-05)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}

echo "=========================================="
echo "GLMsingle TB (pooled) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/glmsingle_tb.py \
    --subject ${SUBJECT} \
    --arm pooled

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
echo "=========================================="

exit ${EXIT_STATUS}

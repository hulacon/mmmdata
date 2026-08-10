#!/bin/bash
#SBATCH --job-name=ps_rawtr
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/ps_rawtr_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/ps_rawtr_%A_%a.err
#SBATCH --array=1-6

#==============================================================================
# Pattern-similarity Phase 3 — rawTR ROI pattern extraction (Model 8)
#==============================================================================
# 62 runs per array task (42 TB + 20 NAT), one (subject x pipeline) each.
# Plan: docs/doc/pattern-similarity-plan.md, Phase 3.
# Array task mapping:
#   1: sub-03/original   2: sub-03/nordic
#   3: sub-04/original   4: sub-04/nordic
#   5: sub-05/original   6: sub-05/nordic
#
# Usage:
#   sbatch extract_rawtr.sbatch
#==============================================================================

SUBJECTS=(sub-03 sub-03 sub-04 sub-04 sub-05 sub-05)
PIPELINES=(original nordic original nordic original nordic)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
PIPELINE=${PIPELINES[$IDX]}

echo "=========================================="
echo "ps_rawtr — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Pipeline: ${PIPELINE}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/pattern_similarity/extract_rawtr.py \
    --sub ${SUBJECT} \
    --pipeline ${PIPELINE}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}

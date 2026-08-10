#!/bin/bash
#SBATCH --job-name=glmsingle_nat
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_nat_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_nat_%A_%a.err
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190
#SBATCH --array=1-6

#==============================================================================
# GLMsingle NATencoding (4.5-s movie chunks) — SLURM Array Job
#==============================================================================
# Pattern-similarity analysis Phase 2 (docs/doc/pattern-similarity-plan.md).
# Masked-2D data layout; sizing reference: full-grid TB fit peaked at 200 GB
# / 18.3 h on 8 CPUs (job 43272306).
# Array task mapping:
#   1: sub-03/original   2: sub-03/nordic
#   3: sub-04/original   4: sub-04/nordic
#   5: sub-05/original   6: sub-05/nordic
#
# Usage:
#   sbatch glmsingle_natencoding.sbatch
#==============================================================================

SUBJECTS=(sub-03 sub-03 sub-04 sub-04 sub-05 sub-05)
PIPELINES=(original nordic original nordic original nordic)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
PIPELINE=${PIPELINES[$IDX]}

echo "=========================================="
echo "GLMsingle NAT — SLURM Job Info"
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

python ${REPO_ROOT}/scripts/glmsingle_natencoding.py \
    --subject ${SUBJECT} \
    --pipeline ${PIPELINE}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}

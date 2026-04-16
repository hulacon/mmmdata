#!/bin/bash
#SBATCH --job-name=glmsingle_nordic
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_nordic_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_nordic_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190
#SBATCH --array=1-6

#==============================================================================
# GLMsingle NORDIC Pilot — SLURM Array Job
#==============================================================================
# Runs GLMsingle on TBencoding data (ses-04) for each subject/pipeline.
# Array task mapping:
#   1: sub-03/original   2: sub-03/nordic
#   3: sub-04/original   4: sub-04/nordic
#   5: sub-05/original   6: sub-05/nordic
#
# Usage:
#   sbatch nordic_glmsingle_pilot.sbatch
#==============================================================================

SUBJECTS=(sub-03 sub-03 sub-04 sub-04 sub-05 sub-05)
PIPELINES=(original nordic original nordic original nordic)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
PIPELINE=${PIPELINES[$IDX]}

echo "=========================================="
echo "GLMsingle NORDIC Pilot — SLURM Job Info"
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

python ${REPO_ROOT}/scripts/nordic_glmsingle_pilot.py \
    --subject ${SUBJECT} \
    --pipeline ${PIPELINE}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}

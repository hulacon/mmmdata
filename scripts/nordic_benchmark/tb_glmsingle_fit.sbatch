#!/bin/bash
#SBATCH --job-name=tb_glmsingle
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_glmsingle_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_glmsingle_%A_%a.err
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190
#SBATCH --array=1-6

# TB GLMsingle benchmark fit — multi-session, NSD shared1000 union design.
# Array task mapping:
#   1: sub-03/original   2: sub-03/nordic
#   3: sub-04/original   4: sub-04/nordic
#   5: sub-05/original   6: sub-05/nordic
#
# For the pilot first-fit, submit only task 1:
#   sbatch --array=1 tb_glmsingle_fit.sbatch

SUBJECTS=(sub-03 sub-03 sub-04 sub-04 sub-05 sub-05)
PIPELINES=(original nordic original nordic original nordic)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
PIPELINE=${PIPELINES[$IDX]}

echo "=========================================="
echo "TB GLMsingle benchmark fit"
echo "=========================================="
echo "Job ID:         ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID:  ${SLURM_ARRAY_TASK_ID}"
echo "Node:           ${SLURMD_NODENAME}"
echo "Subject:        ${SUBJECT}"
echo "Pipeline:       ${PIPELINE}"
echo "Start time:     $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

python -u "${REPO_ROOT}/scripts/nordic_benchmark/tb_glmsingle_fit.py" \
    --sub "${SUBJECT}" \
    --pipeline "${PIPELINE}"

EXIT=$?

echo "=========================================="
echo "End time:       $(date)"
echo "Exit status:    ${EXIT}"
echo "=========================================="

exit ${EXIT}

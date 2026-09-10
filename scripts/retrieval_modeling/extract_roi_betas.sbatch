#!/bin/bash
#SBATCH --job-name=tb_roi_betas
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_roi_betas_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_roi_betas_%A_%a.err

#==============================================================================
# ROI pattern caches for derivatives/glmsingle_tb (all 3 beta types per arm)
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/retrieval-modeling/ (Settles-when 1).
# Sizing: one pickled beta dict is up to ~30 G (pooled TYPEB); types are loaded
# one at a time and freed. Revise from seff after the first array.
#
# Array task mapping:
#   1: sub-03/enc   2: sub-03/ret-image   3: sub-03/ret-word   4: sub-03/pooled
#   5: sub-04/enc   6: sub-04/ret-image   7: sub-04/ret-word   8: sub-04/pooled
#   9: sub-05/enc  10: sub-05/ret-image  11: sub-05/ret-word  12: sub-05/pooled
#
# Usage:
#   sbatch --array=4 extract_roi_betas.sbatch      # pilot: sub-03/pooled (largest)
#   sbatch --array=1-12 extract_roi_betas.sbatch
#==============================================================================

SUBJECTS=(sub-03 sub-03 sub-03 sub-03 sub-04 sub-04 sub-04 sub-04 sub-05 sub-05 sub-05 sub-05)
ARMS=(enc ret-image ret-word pooled enc ret-image ret-word pooled enc ret-image ret-word pooled)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
ARM=${ARMS[$IDX]}

echo "=========================================="
echo "TB ROI betas — Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}   Arm: ${ARM}"
echo "Start Time: $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/retrieval_modeling/extract_roi_betas.py \
    --subject ${SUBJECT} --arm ${ARM} "$@"

EXIT_STATUS=$?
echo ""
echo "End Time: $(date)   Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
exit ${EXIT_STATUS}

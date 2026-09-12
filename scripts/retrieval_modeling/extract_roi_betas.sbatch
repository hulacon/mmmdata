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
# Sizing: measured 2026-09-10 (47191365) at <16 G / <2 min per cell; the
# TB+FIN ret fits are ~8 G per beta type on disk. 80 G is generous.
#
# Array task mapping (5 arms per subject; `pooled` fits were deleted 2026-09-11
# and are not extractable; `ret-*` = TB + FIN since 2026-09-11, `*-tbonly` =
# the retained TB-only fits whose caches were renamed in place):
#   sub-03:  1 enc   2 ret-image   3 ret-word   4 ret-image-tbonly   5 ret-word-tbonly
#   sub-04:  6 enc   7 ret-image   8 ret-word   9 ret-image-tbonly  10 ret-word-tbonly
#   sub-05: 11 enc  12 ret-image  13 ret-word  14 ret-image-tbonly  15 ret-word-tbonly
#
# Usage:
#   sbatch --array=2,3,7,8,12,13 extract_roi_betas.sbatch   # the six TB+FIN ret arms
#   sbatch --array=1-15 extract_roi_betas.sbatch
#==============================================================================

ARM_SET=(enc ret-image ret-word ret-image-tbonly ret-word-tbonly)
SUBJECTS=(sub-03 sub-03 sub-03 sub-03 sub-03 sub-04 sub-04 sub-04 sub-04 sub-04 sub-05 sub-05 sub-05 sub-05 sub-05)
ARMS=("${ARM_SET[@]}" "${ARM_SET[@]}" "${ARM_SET[@]}")

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

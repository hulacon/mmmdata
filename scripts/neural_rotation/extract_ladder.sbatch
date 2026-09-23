#!/bin/bash
#SBATCH --job-name=nr_extract
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nr_extract_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nr_extract_%A_%a.err

#==============================================================================
# Neural-rotation pilot: ROI-ladder pattern caches (TYPED) per subject x arm
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/neural-rotation-pilot/ (step 0/1).
# Same job as retrieval_modeling/extract_roi_betas.sbatch with
# --roi-set ladder: every ladder row (rungs i-iii) + the subject's rung-(iv)
# pRF masks, written beside the six-ROI caches as
#   <sub>_arm-<arm>_set-ladder_desc-typed_roipatterns.npz
# Sizing: the TYPED enc beta file is ~11 G on disk (sub-03) and the ladder
# slice ~2 G; 80 G / 3 h is the six-ROI job's envelope, kept.
#
# Array task mapping (TB-only retrieval arms, the pilot's inputs):
#   sub-03:  1 enc   2 ret-word-tbonly   3 ret-image-tbonly
#   sub-04:  4 enc   5 ret-word-tbonly   6 ret-image-tbonly
#   sub-05:  7 enc   8 ret-word-tbonly   9 ret-image-tbonly
#
# Usage:
#   sbatch --array=1-3 extract_ladder.sbatch        # sub-03 first
#   sbatch --array=4-9 extract_ladder.sbatch
#==============================================================================

ARM_SET=(enc ret-word-tbonly ret-image-tbonly)
SUBJECTS=(sub-03 sub-03 sub-03 sub-04 sub-04 sub-04 sub-05 sub-05 sub-05)
ARMS=("${ARM_SET[@]}" "${ARM_SET[@]}" "${ARM_SET[@]}")

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
ARM=${ARMS[$IDX]}

echo "=========================================="
echo "NR ladder caches — Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}   Arm: ${ARM}"
echo "Start Time: $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/retrieval_modeling/extract_roi_betas.py \
    --subject ${SUBJECT} --arm ${ARM} --types D --roi-set ladder "$@"

EXIT_STATUS=$?
echo ""
echo "End Time: $(date)   Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
exit ${EXIT_STATUS}

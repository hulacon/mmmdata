#!/bin/bash
#SBATCH --job-name=glmsingle_operator_report
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --array=0-14
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_operator_report_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_operator_report_%A_%a.err

#==============================================================================
# Rebuild each GLMsingle TB fit's per-voxel operators and write the operator
# report under <arm>/operator/ (see glmsingle_operator_report.py). One unit =
# one subject x arm; loads the TYPEB, TYPEC and TYPED pickles in turn (each
# 10-16 G for a long fit) plus masked TYPEC betas twice: 120 G.
# Reference grid = the encoding arm's hrfindex dseg; brain mask = intersection
# of the subject's TBencoding run-01 masks (the glmsingle_export_hrf rule).
#
# Usage:
#   sbatch scripts/glmsingle_operator_report.sbatch
#   sbatch --array=0 scripts/glmsingle_operator_report.sbatch   # one unit
#==============================================================================

set -euo pipefail

SUBJECTS=(sub-03 sub-04 sub-05)
ARMS=(enc ret-image ret-word ret-image-tbonly ret-word-tbonly)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID % 3))]}
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}  node ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}  Arm: ${ARM}"
echo "Start Time: $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

REF="${OUTPUT_DIR}/glmsingle_tb/${SUBJECT}/enc/${SUBJECT}_task-TBencoding_space-MNI152NLin2009cAsym_res-2_desc-hrfindex_dseg.nii.gz"
test -f "${REF}" || { echo "ERROR: reference grid ${REF} missing"; exit 1; }
MASKS=$(ls "${OUTPUT_DIR}"/fmriprep/"${SUBJECT}"/ses-*/func/"${SUBJECT}"_ses-*_task-TBencoding_run-01_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz)
test -n "${MASKS}" || { echo "ERROR: no TBencoding run-01 brain masks for ${SUBJECT}"; exit 1; }

# shellcheck disable=SC2086
python "${REPO_ROOT}/scripts/glmsingle_operator_report.py" \
    --subject "${SUBJECT}" --arm "${ARM}" --reference "${REF}" --mask ${MASKS}

echo "End Time: $(date)"

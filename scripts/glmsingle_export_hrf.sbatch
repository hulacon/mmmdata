#!/bin/bash
#SBATCH --job-name=glmsingle_export_hrf
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --array=0-8
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_export_hrf_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_export_hrf_%A_%a.err

#==============================================================================
# Export HRFindex / FitHRFR2 / R2 / meanvol NIfTIs from the siloed GLMsingle
# TB arms. One unit = one subject x arm; loads the TYPEB pickle whole (~16 G),
# hence 48 G. Reference grid = the encoding arm's existing hrfindex dseg.
#
# Usage:
#   sbatch scripts/glmsingle_export_hrf.sbatch
#   sbatch --array=3 scripts/glmsingle_export_hrf.sbatch   # one unit
#==============================================================================

set -euo pipefail

SUBJECTS=(sub-03 sub-04 sub-05)
ARMS=(enc ret-image-tbonly ret-word-tbonly)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID / 3))]}
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID % 3))]}

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}  node ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}  Arm: ${ARM}"
echo "Start Time: $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

REF="${OUTPUT_DIR}/glmsingle_tb/${SUBJECT}/enc/${SUBJECT}_task-TBencoding_space-MNI152NLin2009cAsym_res-2_desc-hrfindex_dseg.nii.gz"
test -f "${REF}" || { echo "ERROR: reference grid ${REF} missing"; exit 1; }

python "${REPO_ROOT}/scripts/glmsingle_export_hrf.py" \
    --subject "${SUBJECT}" --arm "${ARM}" --reference "${REF}"

echo "End Time: $(date)"

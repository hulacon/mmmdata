#!/bin/bash
#SBATCH --job-name=tb_lss
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_lss_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_lss_%A_%a.err

#==============================================================================
# Plain LSS single-trial betas (lss_roi_betas.py) on the ROI voxels of the
# pattern6 cache, optionally plus named ladder ROIs.
#==============================================================================
# One array task per subject: 1 = sub-03, 2 = sub-04, 3 = sub-05.
#   ARM         fit arm (default enc)
#   OUT_DIR     cache root to write under (required); the npz lands at
#               ${OUT_DIR}/cache/glmsingle_tb/<sub>/<arm>/<sub>_arm-<arm>_desc-lss_roipatterns.npz
#   LADDER_ROIS ladder ROI names to add (optional; read from the TYPED ladder cache)
# Usage:
#   OUT_DIR=<dir> LADDER_ROIS="VTC OccipitalPole" sbatch --array=1-3 lss_roi_betas.sbatch
#==============================================================================

SUBJECTS=(sub-03 sub-04 sub-05)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
export REPO_ROOT="${REPO_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${VENV_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}/.venv/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

ARM=${ARM:-enc}
: "${OUT_DIR:?set OUT_DIR}"
FIT=${BIDS_DIR}/derivatives/glmsingle_tb/${SUBJECT}/${ARM}
CACHE=${BIDS_DIR}/derivatives/pattern_similarity/cache/glmsingle_tb/${SUBJECT}/${ARM}
TEMPLATES=(--template ${CACHE}/${SUBJECT}_arm-${ARM}_desc-typed_roipatterns.npz)
ROI_ARGS=()
if [ -n "${LADDER_ROIS}" ]; then
    TEMPLATES+=(--template ${CACHE}/${SUBJECT}_arm-${ARM}_set-ladder_desc-typed_roipatterns.npz)
    ROI_ARGS=(--rois EVC EAC Hippocampus AG Precuneus mPFC ${LADDER_ROIS})
fi
echo "LSS — ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}: ${SUBJECT} ${ARM}  $(date)"
python ${REPO_ROOT}/scripts/tb_item_signal/lss_roi_betas.py --fit-dir ${FIT} "${TEMPLATES[@]}" "${ROI_ARGS[@]}" \
    --out ${OUT_DIR}/cache/glmsingle_tb/${SUBJECT}/${ARM}/${SUBJECT}_arm-${ARM}_desc-lss_roipatterns.npz
S=$?; echo "End $(date) status $S  (seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"; exit $S

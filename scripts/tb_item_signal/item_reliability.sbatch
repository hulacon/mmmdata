#!/bin/bash
#SBATCH --job-name=tb_item_rel
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_item_rel_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_item_rel_%A_%a.err

#==============================================================================
# Cross-run item reliability (item_reliability.py) over TB ROI caches.
#==============================================================================
# One array task per subject: 1 = sub-03, 2 = sub-04, 3 = sub-05.
# Environment knobs (all optional):
#   ARM         fit arm                           (default enc)
#   ROI_SET     pattern6 | ladder | networks      (default pattern6)
#   TYPES       beta types, space-separated       (default "b c d"; "lss" = lss_roi_betas.py output)
#   CACHE_ROOT  root holding cache/glmsingle_tb/  (default derivatives/pattern_similarity)
#   OUT_DIR     where the TSVs go                 (required)
#   ROIS        ROI names, space-separated        (default: every ROI in the cache)
#   N_NULL      shuffle draws                     (default 200)
#
# Usage:
#   OUT_DIR=<dir> sbatch --array=1-3 item_reliability.sbatch
#   OUT_DIR=<dir> ROI_SET=ladder TYPES=d ROIS="VTC OccipitalPole" sbatch --array=1-3 item_reliability.sbatch
#==============================================================================

SUBJECTS=(sub-03 sub-04 sub-05)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

export REPO_ROOT="${REPO_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${VENV_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}/.venv/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

ARM=${ARM:-enc}
ROI_SET=${ROI_SET:-pattern6}
TYPES=${TYPES:-"b c d"}
CACHE_ROOT=${CACHE_ROOT:-${BIDS_DIR}/derivatives/pattern_similarity}
N_NULL=${N_NULL:-200}
: "${OUT_DIR:?set OUT_DIR to the directory the TSVs should go to}"
SET_TAG=""; [ "${ROI_SET}" != "pattern6" ] && SET_TAG="_set-${ROI_SET}"

echo "Item reliability — ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}: ${SUBJECT} ${ARM} ${ROI_SET} types=[${TYPES}]  $(date)"
STATUS=0
for T in ${TYPES}; do
    DESC=type${T}; [ "${T}" = "lss" ] && DESC=lss   # lss_roi_betas.py names its cache desc-lss
    CACHE=${CACHE_ROOT}/cache/glmsingle_tb/${SUBJECT}/${ARM}/${SUBJECT}_arm-${ARM}${SET_TAG}_desc-${DESC}_roipatterns.npz
    OUT=${OUT_DIR}/${SUBJECT}_arm-${ARM}${SET_TAG}_desc-${DESC}_itemrel.tsv
    python ${REPO_ROOT}/scripts/tb_item_signal/item_reliability.py \
        --cache "${CACHE}" --out "${OUT}" --n-null ${N_NULL} ${ROIS:+--rois ${ROIS}} || STATUS=$?
done
echo "End $(date)  status ${STATUS}   (seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
exit ${STATUS}

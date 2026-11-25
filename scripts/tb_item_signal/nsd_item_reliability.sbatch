#!/bin/bash
#SBATCH --job-name=tb_nsd_rel
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_nsd_rel_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_nsd_rel_%A_%a.err

#==============================================================================
# NSD shared1000 -> ROI cache (nsd_shared_cache.py) -> item_reliability.py,
# under the pair scopes that match MMMData (within-session) and NSD's native
# design (cross-session). One array task per NSD subject (1..8).
#   NSD_SHARED  dir with subj##/<space>/{shared_betas_z.npy,trials.tsv,rois.npz} (required)
#   NSD_ROOT    NSD nsddata root holding ppdata/subj##/behav/responses.tsv (required)
#   SPACE       func1pt8mm | fsaverage6 (default func1pt8mm)
#   ROIS        rois.npz names (default "streams_early streams_ventral streams_lateral hippocampus")
#   OUT_DIR     output dir (required)
#   N_NULL      shuffle draws (default 200)
#==============================================================================
S=$(printf "subj%02d" ${SLURM_ARRAY_TASK_ID})
export REPO_ROOT="${REPO_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${VENV_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}/.venv/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${NSD_SHARED:?}" "${NSD_ROOT:?}" "${OUT_DIR:?}"
SPACE=${SPACE:-func1pt8mm}
ROIS=${ROIS:-"streams_early streams_ventral streams_lateral hippocampus"}
N_NULL=${N_NULL:-200}
CACHE=${OUT_DIR}/cache/${S}_space-${SPACE}_nsd_roipatterns.npz
D=${REPO_ROOT}/scripts/tb_item_signal
echo "NSD item reliability — ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}: ${S} ${SPACE}  $(date)"
[ -f "${CACHE}" ] || python ${D}/nsd_shared_cache.py --shared-dir ${NSD_SHARED}/${S}/${SPACE} \
    --responses ${NSD_ROOT}/ppdata/${S}/behav/responses.tsv --rois ${ROIS} --out ${CACHE} || exit 1
STATUS=0
for P in within-session cross-session; do
    python ${D}/item_reliability.py --cache ${CACHE} --pairs ${P} --candidates session --n-null ${N_NULL} \
        --out ${OUT_DIR}/${S}_space-${SPACE}_pairs-${P}_itemrel.tsv || STATUS=$?
done
echo "End $(date) status ${STATUS}  (seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"; exit ${STATUS}

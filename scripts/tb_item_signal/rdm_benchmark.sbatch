#!/bin/bash
#SBATCH --job-name=tb_rdm_bench
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_rdm_bench_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_rdm_bench_%A_%a.err

#==============================================================================
# rdm_benchmark.py: target vs reference ROI caches, second-order (RSA).
# One array task per entry of MASK_ONS (default "run session"; index from 1).
#   TARGETS     space-separated target cache paths (required)
#   REFERENCES  space-separated reference cache paths, >= 2 (required)
#   ROIS        ROI names present in every cache (required)
#   OUT_DIR     output dir (required); writes <TAG>_mask-<m>_rdm.tsv + _matrix.tsv
#   TAG         output prefix (default rdmbench)
#   MASK_ONS    masks, one per array task (default "run session")
#   N_BOOT / N_PERM  (default 1000 / 1000);  EXTRA  more rdm_benchmark.py args
#==============================================================================
export REPO_ROOT="${REPO_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${VENV_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}/.venv/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${TARGETS:?}" "${REFERENCES:?}" "${ROIS:?}" "${OUT_DIR:?}"
read -ra MS <<< "${MASK_ONS:-run session}"
M=${MS[$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))]}
: "${M:?no MASK_ONS entry for array task ${SLURM_ARRAY_TASK_ID}}"
TAG=${TAG:-rdmbench}
echo "RDM benchmark — ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}: mask ${M}  $(date)"
python ${REPO_ROOT}/scripts/tb_item_signal/rdm_benchmark.py \
    --target ${TARGETS} --reference ${REFERENCES} --rois ${ROIS} --mask-on ${M} \
    --n-boot ${N_BOOT:-1000} --n-perm ${N_PERM:-1000} ${EXTRA} \
    --out ${OUT_DIR}/${TAG}_mask-${M}_rdm.tsv --out-matrix ${OUT_DIR}/${TAG}_mask-${M}_matrix.tsv
STATUS=$?
echo "End $(date) status ${STATUS}  (seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"; exit ${STATUS}

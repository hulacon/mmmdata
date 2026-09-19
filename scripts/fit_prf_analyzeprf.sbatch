#!/bin/bash
#SBATCH --job-name=fit_prf_analyzeprf
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fit_prf_analyzeprf_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fit_prf_analyzeprf_%A_%a.err

#==============================================================================
# pRF fit by analyzePRF itself (MATLAB), T1w space, both sessions pooled —
# one array task per SUBJECT x POLARITY
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/prf-retinotopy/
#   DECIDED 2026-09-16: pRF data are handled EXACTLY as analyzePRF handles
#   them, and the backend is analyzePRF (verified viable on compute nodes
#   2026-09-18). Supersedes fit_prf_pooled.sbatch for the released product;
#   fit_prf.py keeps the per-session test-retest units and the exporter.
#
# Three steps, resumable at each:
#   1. prf_analyzeprf_export.py   the pooled pseudo-runs + fit mask, built by
#                                 fit_prf.py's functions (skipped if present;
#                                 shared by both polarities of a subject)
#   2. prf_analyzeprf_fit.m       NSD's verbatim analyzePRF call, in chunks
#                                 (existing chunk files are skipped, so a
#                                 requeued task continues)
#   3. prf_analyzeprf_assemble.py maps + sidecar under --out-root; existing
#                                 files of that suffix are MOVED to
#                                 --archive-root first, never deleted
#
# Cost (measured 2026-09-18, sub-04, 300 voxels): ~2 core-s per voxel ->
# ~5.8 h on 28 cores for a 292k-voxel mask. 12 h leaves 2x headroom.
#
# Array task mapping (subject-major, positive then negative polarity):
#   1: sub-03 pos   2: sub-04 pos   3: sub-05 pos   4: sub-06 pos   5: sub-07 pos
#   6: sub-03 neg   7: sub-04 neg   8: sub-05 neg   9: sub-06 neg  10: sub-07 neg
#
# Usage:
#   # pilot: one subject-polarity into a side tree, product untouched
#   sbatch --array=2 --export=ALL,PRF_OUT_ROOT=/gpfs/.../work/.../prf_pilot fit_prf_analyzeprf.sbatch
#   # the product (archives the Python maps to derivatives/prf_pythoncss)
#   sbatch --array=1-10 fit_prf_analyzeprf.sbatch
#
# Environment overrides:
#   PRF_OUT_ROOT      where maps land (default $OUTPUT_DIR/prf)
#   PRF_ARCHIVE_ROOT  where existing maps move (default $OUTPUT_DIR/prf_pythoncss;
#                     set to "" to refuse instead of archiving)
#   PRF_CHUNKSIZE     voxels per chunk file (default 50000)
#==============================================================================

set -u

SUBJECTS=(03 04 05 06 07 03 04 05 06 07)
NEGATE=( 0  0  0  0  0  1  1  1  1  1)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
IS_NEG=${NEGATE[$IDX]}
POLARITY=prf; [ "${IS_NEG}" = "1" ] && POLARITY=negprf

echo "=========================================="
echo "pRF fit by analyzePRF (T1w, 6 runs pooled) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}   Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}   CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Subject: sub-${SUBJECT}   Sessions: ses-02 + ses-03   Polarity: ${POLARITY}"
echo "Start Time: $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"
module load matlab/R2026a

: "${ANALYZEPRF_DIR:?config/base.toml [paths] analyzeprf_dir is not set}"
: "${KNKUTILS_DIR:?config/base.toml [paths] knkutils_dir is not set}"
export ANALYZEPRF_DIR KNKUTILS_DIR
APRF_COMMIT=$(git -C "${ANALYZEPRF_DIR}" rev-parse --short HEAD 2>/dev/null || echo unrecorded)
KNK_COMMIT=$(git -C "${KNKUTILS_DIR}" rev-parse --short HEAD 2>/dev/null || echo unrecorded)
echo "analyzePRF ${APRF_COMMIT} at ${ANALYZEPRF_DIR}; knkutils ${KNK_COMMIT} at ${KNKUTILS_DIR}"

OUT_ROOT="${PRF_OUT_ROOT:-${OUTPUT_DIR}/prf}"
ARCHIVE_ROOT="${PRF_ARCHIVE_ROOT-${OUTPUT_DIR}/prf_pythoncss}"
CHUNKSIZE="${PRF_CHUNKSIZE:-50000}"
WORK="${WORK_DIR}/prf_analyzeprf/sub-${SUBJECT}"
mkdir -p "${WORK}"
echo "work: ${WORK}   out: ${OUT_ROOT}   archive: ${ARCHIVE_ROOT:-<refuse>}"

echo ""; echo "--- step 1: export"
python "${REPO_ROOT}/scripts/prf_analyzeprf_export.py" \
    --subject "${SUBJECT}" --sessions 02 03 --space T1w --out-dir "${WORK}" || exit 1

echo ""; echo "--- step 2: analyzePRF"
MAT="${WORK}/sub-${SUBJECT}_task-prf_space-T1w_desc-pooled_analyzeprfinput.mat"
CHUNKS="${WORK}/${POLARITY}"
matlab -batch "addpath('${REPO_ROOT}/scripts'); prf_analyzeprf_fit('${MAT}', '${CHUNKS}', ${IS_NEG}, ${SLURM_CPUS_PER_TASK}, ${CHUNKSIZE})" || exit 1

echo ""; echo "--- step 3: assemble"
ARCHIVE_FLAG=()
[ -n "${ARCHIVE_ROOT}" ] && ARCHIVE_FLAG=(--archive-root "${ARCHIVE_ROOT}")
python "${REPO_ROOT}/scripts/prf_analyzeprf_assemble.py" \
    --export-dir "${WORK}" --chunk-dir "${CHUNKS}" \
    --subject "${SUBJECT}" --space T1w --polarity "${POLARITY}" \
    --out-root "${OUT_ROOT}" "${ARCHIVE_FLAG[@]}" \
    --analyzeprf-commit "${APRF_COMMIT}" --knkutils-commit "${KNK_COMMIT}"
EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
echo "=========================================="
exit ${EXIT_STATUS}

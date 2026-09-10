#!/bin/bash
#SBATCH --job-name=fit_prf_pooled
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fit_prf_pooled_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fit_prf_pooled_%A_%a.err

#==============================================================================
# CSS pRF fit, T1w space, both sessions pooled — one array task per SUBJECT
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/prf-retinotopy/
#   (the final product, DECIDED 2026-09-09; per-session `func` fits stay on
#   disk as the test-retest evidence and are produced by fit_prf.sbatch.)
#
# Why this exists and the per-session array does not suffice: until the T1w
# backfill landed, the two pRF sessions had no common grid (native affines
# differ by 5.7-14.5 mm), so six runs could not enter one fit. In space-T1w
# they do -- bit-identically for sub-04/05/06/07, and for sub-03 after one
# resample this script's driver performs and records, its ses-02 having been
# acquired at a smaller FOV.
#
# Pooling follows NSD (cvnlab/nsddatapaper main/glm_prf.m, read 2026-09-09):
# AVERAGE the three runs of each setnum, then fit the two pseudo-runs jointly.
# 400 TRs, not 1200. Our ses-02 (93,94,93) and ses-03 (94,93,94) order puts
# both sessions in both pseudo-runs, so session and stimulus type come out
# partially crossed rather than confounded.
#
# Sizing: the pooled sub-03 pilot (47160475, 2026-09-10) measured 6:24 wall
# and 6.7 G peak RSS on 28 cores -- grid search dominated, refinement of
# 15,727 voxels took 1.5 min. 1 h / 16 G leaves ~9x / 2.4x headroom for the
# larger masks (up to ~297k voxels) and the negative arm, which refines more
# voxels but runs the same grid stage. sub-03 additionally holds one
# resampled 4D volume in memory and is still the peak-RSS case.
#
# Array task mapping (subject-major, positive then negative polarity):
#   1: sub-03 pos   2: sub-04 pos   3: sub-05 pos   4: sub-06 pos   5: sub-07 pos
#   6: sub-03 neg   7: sub-04 neg   8: sub-05 neg   9: sub-06 neg  10: sub-07 neg
#
# Usage:
#   sbatch --array=1 fit_prf_pooled.sbatch          # pilot on sub-03 (the
#                                                   # resample case) before the rest
#   sbatch --array=2-10 fit_prf_pooled.sbatch       # after seff review
#
#   # confound sensitivity arm -- one subject, three cells, `*prf` suffix marked
#   # so it can never overwrite the released product
#   sbatch --job-name=fit_prf_conf --export=ALL,PRF_CONFOUNDS=motion6  --array=1 fit_prf_pooled.sbatch
#   sbatch --job-name=fit_prf_conf --export=ALL,PRF_CONFOUNDS=acompcor --array=1 fit_prf_pooled.sbatch
#==============================================================================

SUBJECTS=(03 04 05 06 07 03 04 05 06 07)
NEGATE=( 0  0  0  0  0  1  1  1  1  1)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
IS_NEG=${NEGATE[$IDX]}

echo "=========================================="
echo "pRF fit (T1w, 6 runs pooled) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: sub-${SUBJECT}   Sessions: ses-02 + ses-03   Negate: ${IS_NEG}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

# One BLAS thread per worker. joblib already owns all 28 cores -- the grid
# search runs 28 threads (numpy-bound, GIL released) and the refinement runs 28
# loky processes, so any BLAS threading on top of that is pure oversubscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

NEGATE_FLAG=""
if [ "${IS_NEG}" = "1" ]; then
    NEGATE_FLAG="--negate"
    echo "Negative-pRF variant: fitting sign-flipped BOLD (negprf suffix)"
fi

CONFOUNDS="${PRF_CONFOUNDS:-none}"
echo "Confound model: ${CONFOUNDS}"
if [ "${CONFOUNDS}" != "none" ]; then
    echo "  NOTE: sensitivity arm, not the released product. The default is no"
    echo "        confound regression, matching analyzePRF, NSD, Klink and Steel."
fi

# --grid-session is left at its default (the last session given, ses-03), which
# is the study-standard geometry: sub-03's ses-02 is the odd FOV out and is the
# session that gets resampled.
python ${REPO_ROOT}/scripts/fit_prf.py \
    --subject ${SUBJECT} \
    --sessions 02 03 \
    --space T1w \
    --pool average \
    --confounds ${CONFOUNDS} \
    --jobs ${SLURM_CPUS_PER_TASK} \
    ${NEGATE_FLAG}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
echo "=========================================="

exit ${EXIT_STATUS}

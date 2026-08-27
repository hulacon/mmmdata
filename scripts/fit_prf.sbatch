#!/bin/bash
#SBATCH --job-name=fit_prf
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fit_prf_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fit_prf_%A_%a.err

#==============================================================================
# CSS pRF fit in the native functional volume — one array task per SESSION
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/prf-retinotopy/
#   (fit space DECIDED 2026-08-26: native volume, projected to fsnative after).
#
# One session = one fit unit, and that is measured rather than stylistic: within
# a session all three pRF runs share the native grid exactly (max |affine diff|
# 0.000, every subject), across sessions they never do (5.7-14.5 mm). Three runs
# x 200 TRs = 600 TRs = 15 min of retinotopy. Per-session fits are also the
# test-retest units the charter's Settles-when 3 asks for, so this array gives
# both legs at once.
#
# Sizing, revised from the pilot's seff (array task 1, COMPLETED exit 0):
#   wall            3:54 (the pre-submission ~5.5 h refinement estimate carried
#                   login-node BLAS pessimism; only ~2% of the mask clears the
#                   grid-R2 gate for refinement, not the full mask)
#   peak memory     8.64 GB of 64 G
# 2 h / 24 G gives ~30x time and ~2.5x memory headroom over the measured pilot,
# covering mask-size variation across subjects.
#
# Array task mapping (subject-major, ses-02 then ses-03):
#   1: sub-03/ses-02   2: sub-03/ses-03   3: sub-04/ses-02   4: sub-04/ses-03
#   5: sub-05/ses-02   6: sub-05/ses-03   7: sub-06/ses-02   8: sub-06/ses-03
#   9: sub-07/ses-02  10: sub-07/ses-03
#
# Usage:
#   sbatch --array=1 fit_prf.sbatch        # pilot: sub-03 ses-02 (gate passed here)
#   sbatch --array=2-10 fit_prf.sbatch     # the rest, after seff review
#==============================================================================

SUBJECTS=(03 03 04 04 05 05 06 06 07 07)
SESSIONS=(02 03 02 03 02 03 02 03 02 03)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
SESSION=${SESSIONS[$IDX]}

echo "=========================================="
echo "pRF fit (native volume) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: sub-${SUBJECT}   Session: ses-${SESSION}"
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

python ${REPO_ROOT}/scripts/fit_prf.py \
    --subject ${SUBJECT} \
    --session ${SESSION} \
    --jobs ${SLURM_CPUS_PER_TASK}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
echo "=========================================="

exit ${EXIT_STATUS}

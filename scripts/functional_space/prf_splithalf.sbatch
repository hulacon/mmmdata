#!/bin/bash
#SBATCH --job-name=prf_splithalf
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=112G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/prf_splithalf_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/prf_splithalf_%A_%a.err

#==============================================================================
# Session split-half of the pooled pRF fit: one array task per (subject, session).
# Positive polarity only.
#
#   1. functional_space/prf_splithalf_export.py  one session's runs, pooled fit mask
#   2. prf_analyzeprf_fit.m                      the pooled product's fit, unchanged
#   3. prf_analyzeprf_assemble.py                the pooled product's assembler, unchanged
#                                                (into a staging root in the work dir)
#   4. functional_space/prf_splithalf_project.py place under derivatives/, fsnative, fsaverage6
#
# Sizing: the pooled product's array (fit_prf_analyzeprf.sbatch, 28 CPUs /
# 112 G) ran 57-81 min per fit with MaxRSS ~50-54 G. A half has the same mask
# and the same pseudo-run count, so the same cost; 4 h is ~3x headroom.
#
# Usage:
#   sbatch --array=0-5 --export=ALL,UNITS="##:## ##:## ..." prf_splithalf.sbatch   # subject:session pairs
#==============================================================================

set -u
: "${UNITS:?set UNITS (space-separated subject:session pairs, bare labels) via --export}"
# shellcheck disable=SC2206
UNIT_ARR=(${UNITS})
UNIT="${UNIT_ARR[${SLURM_ARRAY_TASK_ID}]}"
SUBJECT="${UNIT%%:*}"
SESSION="${UNIT##*:}"

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}: sub-${SUBJECT} half ses-${SESSION} — $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
# Modules first, then name the venv interpreter absolutely: a module load after
# venv activation would replace `python`.
module load matlab/R2026a
PY="${REPO_ROOT}/.venv/bin/python"

: "${ANALYZEPRF_DIR:?config/base.toml [paths] analyzeprf_dir is not set}"
: "${KNKUTILS_DIR:?config/base.toml [paths] knkutils_dir is not set}"
export ANALYZEPRF_DIR KNKUTILS_DIR
APRF_COMMIT=$(git -C "${ANALYZEPRF_DIR}" rev-parse --short HEAD 2>/dev/null || echo unrecorded)
KNK_COMMIT=$(git -C "${KNKUTILS_DIR}" rev-parse --short HEAD 2>/dev/null || echo unrecorded)
echo "analyzePRF ${APRF_COMMIT}; knkutils ${KNK_COMMIT}"

WORK="${WORK_DIR}/prf_splithalf/sub-${SUBJECT}/ses-${SESSION}"
STAGING="${WORK}/assembled"
mkdir -p "${WORK}"

echo "--- step 1: export"
"${PY}" "${REPO_ROOT}/scripts/functional_space/prf_splithalf_export.py" \
    --subject "${SUBJECT}" --session "${SESSION}" --out-dir "${WORK}" || exit 1

echo "--- step 2: analyzePRF"
MAT="${WORK}/sub-${SUBJECT}_task-prf_space-T1w_desc-pooled_analyzeprfinput.mat"
matlab -batch "addpath('${REPO_ROOT}/scripts'); prf_analyzeprf_fit('${MAT}', '${WORK}/prf', 0, ${SLURM_CPUS_PER_TASK}, 50000)" || exit 1

echo "--- step 3: assemble"
rm -rf "${STAGING}"
"${PY}" "${REPO_ROOT}/scripts/prf_analyzeprf_assemble.py" \
    --export-dir "${WORK}" --chunk-dir "${WORK}/prf" --subject "${SUBJECT}" --space T1w \
    --polarity prf --out-root "${STAGING}" \
    --analyzeprf-commit "${APRF_COMMIT}" --knkutils-commit "${KNK_COMMIT}" || exit 1

echo "--- step 4: place, fsnative, fsaverage6"
"${PY}" "${REPO_ROOT}/scripts/functional_space/prf_splithalf_project.py" \
    --subject "${SUBJECT}" --session "${SESSION}" --staging "${STAGING}"
EXIT_STATUS=$?
echo "End: $(date)   exit ${EXIT_STATUS}"
exit ${EXIT_STATUS}

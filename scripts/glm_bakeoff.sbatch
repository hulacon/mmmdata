#!/bin/bash
#SBATCH --job-name=glm_bakeoff
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glm_bakeoff_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glm_bakeoff_%A_%a.err

#==============================================================================
# GLM estimator bake-off — one cell per array task
#==============================================================================
# Design record: mmmdata-agents/docs/workbench/glm-strategy/. Runner:
# scripts/glm_bakeoff.py; the units file is written by `glm_bakeoff.py plan`
# into <derivatives>/glm_bakeoff/units.txt (one "sub-XX cell-id" per line).
#
# Sizing is a guess until the pilot: the production contrast-map runner
# measured 3.7-5.3 G / 1-2 min for fLoc and motor with nilearn; TBencoding is
# 42 runs of 210 volumes, and the remlfit engine (ARMA(1,1) per voxel, 8 OpenMP
# threads) is the slow one — a voxelwise-HRF x remlfit x TBencoding cell runs
# 42 x 20 3dREMLfit calls. 8 h is the envelope for that corner; revise from
# seff after the pilot units.
#
# Usage:
#   python scripts/glm_bakeoff.py plan            # freeze harness, write units.txt
#   N=$(wc -l < /gpfs/projects/hulacon/shared/mmmdata/derivatives/glm_bakeoff/units.txt)
#   sbatch --array=1-${N}%24 scripts/glm_bakeoff.sbatch
#   sbatch --array=5 --export=ALL,EXTRA="--force" scripts/glm_bakeoff.sbatch
#   GLOBAL holds arguments that go BEFORE the verb (--out-base, --space ...):
#   sbatch --array=7,61 --export=ALL,GLOBAL="--out-base ${OUT}/glm_bakeoff_pass2",EXTRA="--keep-per-run" \
#       scripts/glm_bakeoff.sbatch
#   A track that planned its own units file passes it in GLOBAL too:
#   sbatch --array=1-18 --export=ALL,GLOBAL="--out-base ${OUT}/glm_bakeoff_pass2 --units trackA.txt" \
#       scripts/glm_bakeoff.sbatch
#==============================================================================

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array; each task fits one unit of units.txt}"
EXTRA="${EXTRA:-}"
GLOBAL="${GLOBAL:-}"

echo "=========================================="
echo "GLM bake-off — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"

# EVERY `module load` goes BEFORE the venv, and the interpreter is named by
# absolute path below. Both matter: fsl/6.0.7.9 ships its own python 3.11 in
# $FSLDIR/bin, so loading it after `activate` puts FSL's python ahead of the
# venv's on PATH and the job dies on `No module named nilearn` (array
# 47264306, 2026-09-12 — 30/30 fits lost to exactly this).
module load afni/24.1.22 2>/dev/null || true
command -v 3dREMLfit >/dev/null || export PATH="/packages/afni/24.1.22:${PATH}"
# FSL for the FILM engine levels (Track A). film_gls is on PATH in the login
# environment but not necessarily inside a batch job.
module load fsl/6.0.7.9 2>/dev/null || true
command -v film_gls >/dev/null || export PATH="/packages/fsl/6.0.7.9/fsl/bin:${PATH}"
source "${REPO_ROOT}/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TMPDIR="${TMPDIR:-/tmp}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
echo "python:   $("${PYTHON}" -c 'import sys, nilearn; print(sys.version.split()[0], "nilearn", nilearn.__version__)')"
echo "film_gls: $(command -v film_gls || echo MISSING)"
echo "3dREMLfit: $(command -v 3dREMLfit || echo MISSING)"

# shellcheck disable=SC2086
"${PYTHON}" "${REPO_ROOT}/scripts/glm_bakeoff.py" ${GLOBAL} fit --unit "${SLURM_ARRAY_TASK_ID}" ${EXTRA}

echo ""
echo "End Time: $(date)"

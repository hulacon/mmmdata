#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=space_geometry
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/space_geometry_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/space_geometry_%j.err

#==============================================================================
# Feature-space geometry: the space x space matrix for shared1000
#   (workbench feature-space-geometry, charter Settles-when)
#==============================================================================
# Four measures over every pair of the 35 image-level spaces at n = 1000:
#   ridge A->B (both directions), linear + RBF CKA, second-order RSA, and
#   top-20 neighbour overlap against a 1000-draw permutation null.
#
# RUN THIS ON A COMPUTE NODE, NEVER ON A LOGIN NODE. Measured 2026-08-24: the
# same RidgeCV fit is 0.17 s with BLAS threads bounded and 26 s unbounded on
# login1, because OpenBLAS sizes its pool from the 48 visible cores while ~12
# are actually free. That artefact is what the charter recorded as a ~50
# core-hour blocker. Inside this allocation the cgroup bounds the pool, so
# the whole matrix is ~75 min.
#
# Usage:  sbatch space_geometry.sbatch [outdir]
#==============================================================================

set -euo pipefail

PY=/gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
STORE=/gpfs/projects/hulacon/shared/mmmdata/derivatives/stimuli_features/psytwill
OUTDIR="${1:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata-agents/docs/workbench/feature-space-geometry/out}"

echo "host      : $(hostname)"
echo "cpus      : ${SLURM_CPUS_PER_TASK:-?}"
echo "outdir    : ${OUTDIR}"
echo "psytwill  : $($PY -c 'import psytwill; print(psytwill.__version__, psytwill.__file__)')"
echo "started   : $(date -Is)"
echo

# Fold structure: shared1000 rows are exchangeable images, so folds are random
# and the null is a full row shuffle (no --group-by, no --block-size). Both
# become mandatory on the movie grid — see charter design constraint 2.
srun "$PY" -m psytwill.cli compare \
    "${STORE}/shared1000_image_features.parquet:image" \
    "${STORE}/shared1000_ebind_features.parquet:ebindgrp" \
    "${STORE}/shared1000_caption_chunks_features.parquet:cap" \
    "${STORE}/shared1000_humancap_chunks_features.parquet:humancap" \
    --key stimulus_id \
    --pool mean \
    --k 20 \
    --n-splits 5 \
    --permutations 1000 \
    --seed 0 \
    --name shared1000_geometry \
    -o "${OUTDIR}"

echo
echo "finished  : $(date -Is)"

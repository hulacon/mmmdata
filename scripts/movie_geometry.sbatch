#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=movie_geometry
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/movie_geometry_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/movie_geometry_%j.err

#==============================================================================
# Feature-space geometry: the space x space matrix on the movie 0.5 s grid
#   (workbench feature-space-geometry, movie leg — build order step 2)
#==============================================================================
# The four natively-gridded movie groups (frames, ebind, audio_frames,
# caption_chunks) compared at one window width per submission. The ebind
# group is included although frames already carries ebind: the dedupe must
# fire and be recorded in the load report, which makes every run a live
# check that the two copies are still identical.
#
# Fold structure is NOT optional here (charter design constraint 2): folds
# group by clip and the neighbour-overlap null permutes temporal blocks.
# --window is also not optional, even at the native 0.5 s grain — the visual
# groups stamp bin starts and the audio groups bin centers, and binning is
# what puts them on one row index.
#
# RUN ON A COMPUTE NODE ONLY — login-node BLAS timings are ~150x pessimistic.
#
# Usage:
#   sbatch [--time=... --partition=...] movie_geometry.sbatch \
#       WIDTH BLOCK PERMS NAME [MEASURES] [STRIDE]
#
#   WIDTH    window width in seconds (0.5 | 4 | 32)
#   BLOCK    overlap-null block size in rows (~30-60 s at the width)
#   PERMS    permutation draws (0 skips the null)
#   NAME     output file stem, e.g. movies_geometry_w4
#   MEASURES optional comma list (default: all five)
#   STRIDE   optional; keep every Nth row per clip (default 1). Used to run
#            the n^2 measures (rsa, cka_rbf) at the native grain, where
#            full-n costs 65-154 h (probe 46627576).
#==============================================================================

set -euo pipefail

WIDTH="${1:?width in seconds}"
BLOCK="${2:?block size in rows}"
PERMS="${3:?permutation draws}"
NAME="${4:?output name}"
MEASURES="${5:-}"
STRIDE="${6:-1}"

PY=/gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
STORE=/gpfs/projects/hulacon/shared/mmmdata/derivatives/stimuli_features/psytwill
OUTDIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata-agents/docs/workbench/feature-space-geometry/out

echo "host      : $(hostname)"
echo "cpus      : ${SLURM_CPUS_PER_TASK:-?}"
echo "width     : ${WIDTH}s  block=${BLOCK}  perms=${PERMS}  stride=${STRIDE}"
echo "measures  : ${MEASURES:-all}"
echo "psytwill  : $($PY -c 'import psytwill; print(psytwill.__version__, psytwill.__file__)')"
echo "started   : $(date -Is)"
echo

srun "$PY" -m psytwill.cli compare \
    "${STORE}/movies_frames_features.parquet:frames" \
    "${STORE}/movies_ebind_features.parquet:ebindgrp" \
    "${STORE}/movies_audio_frames_features.parquet:audio" \
    "${STORE}/movies_caption_chunks_features.parquet:cap" \
    --key stimulus_id,time \
    --window "${WIDTH}" \
    --pool mean \
    --group-by \
    --block-size "${BLOCK}" \
    --permutations "${PERMS}" \
    --stride "${STRIDE}" \
    ${MEASURES:+--measures "${MEASURES}"} \
    --k 20 \
    --n-splits 5 \
    --seed 0 \
    --name "${NAME}" \
    -o "${OUTDIR}"

echo
echo "finished  : $(date -Is)"

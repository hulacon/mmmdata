#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=space_decompose
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/space_decompose_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/space_decompose_%j.err

#==============================================================================
# Cross-space decomposition: CV-CCA shared-dimension counts vs one source
#   (workbench feature-space-geometry, design constraint 4)
#==============================================================================
# `psytwill decompose` runs cross-validated CCA from the source space against
# every other space and counts the components whose HELD-OUT canonical
# correlation clears a permutation threshold (longest prefix). No fitted
# transform is exported: this is a measurement on these stimuli, not a
# reusable basis — a reusable basis is fit-corpus territory (psytwill
# fitcorpus), deliberately out of scope here.
#
# Fold and null structure track the geometry runs exactly: shared1000 uses
# random folds and a full row-shuffle null; movies use clip-grouped folds and
# temporal-block permutation, and --window is mandatory even at the native
# grain (bin-start vs bin-center stamps).
#
# RUN ON A COMPUTE NODE ONLY — login-node BLAS timings are ~150x pessimistic.
#
# Usage:
#   sbatch space_decompose.sbatch shared1000 SOURCE NAME [PERMS] [RANK_CAP]
#   sbatch space_decompose.sbatch movies WIDTH BLOCK SOURCE NAME [PERMS] [RANK_CAP]
#
#   SOURCE    the space decomposed against, e.g. image:ebind / frames:ebind
#   WIDTH     movie window width in seconds (0.5 | 4 | 32)
#   BLOCK     null block size in rows (~30-60 s at the width)
#   PERMS     permutation draws (default 250)
#   RANK_CAP  max PCA rank per space before whitening (default 128)
#==============================================================================

set -euo pipefail

MODE="${1:?mode: shared1000 | movies}"

PY=/gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
STORE=/gpfs/projects/hulacon/shared/mmmdata/derivatives/stimuli_features/psytwill
OUTDIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata-agents/docs/workbench/feature-space-geometry/out

echo "host      : $(hostname)"
echo "cpus      : ${SLURM_CPUS_PER_TASK:-?}"
echo "mode      : ${MODE}"
echo "psytwill  : $($PY -c 'import psytwill; print(psytwill.__version__, psytwill.__file__)')"
echo "started   : $(date -Is)"
echo

if [[ "${MODE}" == "shared1000" ]]; then
    SOURCE="${2:?source space, e.g. image:ebind}"
    NAME="${3:?output name}"
    PERMS="${4:-250}"
    RANK_CAP="${5:-128}"
    srun "$PY" -m psytwill.cli decompose \
        "${STORE}/shared1000_image_features.parquet:image" \
        "${STORE}/shared1000_ebind_features.parquet:ebindgrp" \
        "${STORE}/shared1000_caption_chunks_features.parquet:cap" \
        "${STORE}/shared1000_humancap_chunks_features.parquet:humancap" \
        --key stimulus_id \
        --pool mean \
        --source "${SOURCE}" \
        --rank-cap "${RANK_CAP}" \
        --n-splits 5 \
        --permutations "${PERMS}" \
        --seed 0 \
        --name "${NAME}" \
        -o "${OUTDIR}"
elif [[ "${MODE}" == "movies" ]]; then
    WIDTH="${2:?width in seconds}"
    BLOCK="${3:?block size in rows}"
    SOURCE="${4:?source space, e.g. frames:ebind}"
    NAME="${5:?output name}"
    PERMS="${6:-250}"
    RANK_CAP="${7:-128}"
    srun "$PY" -m psytwill.cli decompose \
        "${STORE}/movies_frames_features.parquet:frames" \
        "${STORE}/movies_ebind_features.parquet:ebindgrp" \
        "${STORE}/movies_audio_frames_features.parquet:audio" \
        "${STORE}/movies_caption_chunks_features.parquet:cap" \
        --key stimulus_id,time \
        --window "${WIDTH}" \
        --pool mean \
        --group-by \
        --block-size "${BLOCK}" \
        --source "${SOURCE}" \
        --rank-cap "${RANK_CAP}" \
        --n-splits 5 \
        --permutations "${PERMS}" \
        --seed 0 \
        --name "${NAME}" \
        -o "${OUTDIR}"
else
    echo "unknown mode '${MODE}' (shared1000 | movies)" >&2
    exit 2
fi

echo
echo "finished  : $(date -Is)"

#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=segment_geometry
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/segment_geometry_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/segment_geometry_%j.err

#==============================================================================
# Feature-space geometry at segment grain: the irregular tail
#   (workbench feature-space-geometry, time-aware cross mode's second half)
#==============================================================================
# `psytwill project` pools the four gridded movie groups onto an irregular
# interval index (human scene segments or transcript chunks), then
# `psytwill compare` runs the usual measures at that grain — rows are real
# annotation units, folds group by clip, the null permutes contiguous
# segments within clip. The ebind group is projected although frames already
# carries ebind: the dedupe must fire at load, which keeps every run a live
# check that the two copies are still identical.
#
# Odd/even-bin projections are also written per group: they are the two
# halves of the split-half stability ceiling for pooled machine targets
# (pre-registered co-primary reading, workbench log 2026-08-26).
#
# RUN ON A COMPUTE NODE ONLY — the frames group read needs job memory
# (login-node attempt was OOM-killed), and login-node BLAS timings are
# ~150x pessimistic besides.
#
# Usage:  sbatch segment_geometry.sbatch GRAIN [PERMS]
#   GRAIN  segc | segb | transcript
#   PERMS  permutation draws (default 1000)
#==============================================================================

set -euo pipefail

GRAIN="${1:?grain: segc | segb | transcript}"
PERMS="${2:-1000}"

PY=/gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
STORE=/gpfs/projects/hulacon/shared/mmmdata/derivatives/stimuli_features/psytwill
OUTDIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata-agents/docs/workbench/feature-space-geometry/out
PROJDIR="${OUTDIR}/projected"
WINDOW=0.5

# Block sizes are pre-registered (workbench log 2026-08-26): ~30-60 s of
# contiguous segments at each grain's median duration.
case "${GRAIN}" in
  segc)       INTERVALS="${STORE}/movies_annot_chunks_features.parquet";      TEXTPFX=annot; BLOCK=6 ;;
  segb)       INTERVALS="${STORE}/movies_annot_segb_chunks_features.parquet"; TEXTPFX=annot; BLOCK=2 ;;
  transcript) INTERVALS="${STORE}/movies_transcript_chunks_features.parquet"; TEXTPFX=trans; BLOCK=6 ;;
  *) echo "unknown grain '${GRAIN}' (segc | segb | transcript)" >&2; exit 2 ;;
esac

mkdir -p "${PROJDIR}"

echo "host      : $(hostname)"
echo "cpus      : ${SLURM_CPUS_PER_TASK:-?}"
echo "grain     : ${GRAIN}  block=${BLOCK}  perms=${PERMS}"
echo "psytwill  : $($PY -c 'import psytwill; print(psytwill.__version__, psytwill.__file__)')"
echo "started   : $(date -Is)"
echo

for GROUP in frames ebind audio_frames caption_chunks; do
  case "${GROUP}" in
    frames)         SRC="${STORE}/movies_frames_features.parquet" ;;
    ebind)          SRC="${STORE}/movies_ebind_features.parquet" ;;
    audio_frames)   SRC="${STORE}/movies_audio_frames_features.parquet" ;;
    caption_chunks) SRC="${STORE}/movies_caption_chunks_features.parquet" ;;
  esac
  for BINS in all odd even; do
    OUT="${PROJDIR}/${GRAIN}_${GROUP}_${BINS}.parquet"
    echo "--- project ${GROUP} (${BINS}) -> ${OUT}"
    srun "$PY" -m psytwill.cli project "${SRC}" \
        --intervals "${INTERVALS}" \
        --window "${WINDOW}" \
        --bins "${BINS}" \
        -o "${OUT}"
  done
done

echo
echo "--- compare at ${GRAIN} grain"
srun "$PY" -m psytwill.cli compare \
    "${PROJDIR}/${GRAIN}_frames_all.parquet:frames" \
    "${PROJDIR}/${GRAIN}_ebind_all.parquet:ebindgrp" \
    "${PROJDIR}/${GRAIN}_audio_frames_all.parquet:audio" \
    "${PROJDIR}/${GRAIN}_caption_chunks_all.parquet:cap" \
    "${INTERVALS}:${TEXTPFX}" \
    --key stimulus_id,chunk_idx \
    --pool mean \
    --group-by \
    --block-size "${BLOCK}" \
    --permutations "${PERMS}" \
    --k 20 \
    --n-splits 5 \
    --seed 0 \
    --name "segments_geometry_${GRAIN}" \
    -o "${OUTDIR}"

echo
echo "finished  : $(date -Is)"

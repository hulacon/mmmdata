#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=fitcorpus_stage
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fitcorpus_stage_%x_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fitcorpus_stage_%x_%j.err

#==============================================================================
# psytwill-space fit-corpus staging — Talapas wrapper (one corpus per job)
#==============================================================================
# Site wrapper around psytwill's `scripts/fitcorpus/stage.py`: packs a
# short-file corpus (speech utterances, music clips, spoken words) into
# grid-aligned units — one wav per unit on scratch, segment tables and the
# unit manifest in the durable tree — which `fitcorpus_extract.sbatch` then
# consumes. Arguments pass straight through:
#
#   sbatch -J stage_librispeech fitcorpus_stage.sbatch librispeech
#   sbatch -J stage_gigaspeech  --mem=48G fitcorpus_stage.sbatch gigaspeech --budget-hours 40
#   sbatch -J stage_fma         fitcorpus_stage.sbatch fma --per-genre 250
#   sbatch -J stage_musopen     fitcorpus_stage.sbatch musopen
#   sbatch -J stage_narratives  fitcorpus_stage.sbatch narratives
#   sbatch -J stage_twp         fitcorpus_stage.sbatch twp-unpresented
#
# GigaSpeech holds the selected segments' audio in memory while packing
# (~4.6 GB for 40 h) — give it the larger --mem. Idempotent per unit.
# No email flags by standing preference.
#==============================================================================

set -uo pipefail
echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"
echo "Args: $*"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
export PATH="${STIMFEAT}/bin:${PATH}"   # ffmpeg/ffprobe for decoding
export LD_LIBRARY_PATH="${STIMFEAT}/lib:${LD_LIBRARY_PATH:-}"
PY="${STIMFEAT}/bin/python"

export PSYTWILL_FITCORPUS_DURABLE="${PSYTWILL_FITCORPUS_DURABLE:-/gpfs/projects/hulacon/shared/psytwill/fit-corpora}"
export PSYTWILL_FITCORPUS_SCRATCH="${PSYTWILL_FITCORPUS_SCRATCH:-/gpfs/scratch/hulacon/bhutch/psytwill-fitcorpus}"
TWP_ROOT="${BIDS_DIR:-/gpfs/projects/hulacon/shared/mmmdata}/stimuli"

STAGE=/gpfs/projects/hulacon/shared/mmmdata/code/psytwill/scripts/fitcorpus/stage.py

EXTRA=()
if [[ "${1:-}" == "twp-unpresented" ]]; then
    EXTRA=(--twp-root "${TWP_ROOT}")
fi

rc=0
"${PY}" "${STAGE}" "$@" "${EXTRA[@]}" || rc=$?
echo "Finished $(date -Is) rc=${rc}"
exit ${rc}

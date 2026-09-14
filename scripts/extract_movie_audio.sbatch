#!/bin/bash
#SBATCH --job-name=movie_audio
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/movie_audio_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/movie_audio_%j.err

#==============================================================================
# Per-film audio tracks for stimulus-feature viewers — one job, all films
#==============================================================================
# Backfills derivatives/stimuli_features/movies/<slug>/audio.m4a via
# scripts/extract_movie_audio.py (companion to the frames/ screengrabs).
# Encoding is seconds per film, so a single serial job covers all of them.
# Idempotent: existing audio files are skipped, so a resubmit only redoes
# missing films.
#
# Usage:
#   sbatch extract_movie_audio.sbatch                  # all films
#   sbatch --export=ALL,SLUG=<slug> extract_movie_audio.sbatch   # one film
#==============================================================================

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

module load ffmpeg

# Any modern python works here; the script only needs the stdlib.
STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
PY="${STIMFEAT}/bin/python"

FEATURES_ROOT="${OUTPUT_DIR}/stimuli_features"

if [[ -n "${SLUG:-}" ]]; then
    SELECT=(--slug "${SLUG}")
else
    SELECT=(--all)
fi

cd "${REPO_ROOT}/scripts"
"${PY}" extract_movie_audio.py \
    --features-root "${FEATURES_ROOT}" \
    "${SELECT[@]}"

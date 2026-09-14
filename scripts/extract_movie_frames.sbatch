#!/bin/bash
#SBATCH --job-name=movie_frames
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/movie_frames_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/movie_frames_%A_%a.err

#==============================================================================
# Movie screengrabs for stimulus-feature dashboards — one array task per film
#==============================================================================
# Backfills derivatives/stimuli_features/movies/<slug>/frames/ (one JPEG per
# feature time point) via scripts/extract_movie_frames.py, which reads each
# film directory's video sidecar for the source .mov and frame interval.
# Idempotent: complete frames/ dirs are skipped, so a resubmit only redoes
# missing films.
#
# Usage:
#   sbatch --array=1 extract_movie_frames.sbatch        # pilot: first film
#   sbatch --array=1-N extract_movie_frames.sbatch      # N = film-dir count:
#   python scripts/extract_movie_frames.py --features-root <root> --list
#==============================================================================

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

# stimfeat env without conda: the interpreter directly, plus the activate.d
# hooks (shared caches, PYTHONNOUSERSITE) which key off CONDA_PREFIX.
STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
export LD_LIBRARY_PATH="${STIMFEAT}/lib:${LD_LIBRARY_PATH:-}"
PY="${STIMFEAT}/bin/python"

FEATURES_ROOT="${OUTPUT_DIR}/stimuli_features"

"${PY}" "${REPO_ROOT}/scripts/extract_movie_frames.py" \
    --features-root "${FEATURES_ROOT}" \
    --index "${SLURM_ARRAY_TASK_ID}"

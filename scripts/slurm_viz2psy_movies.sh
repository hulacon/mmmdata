#!/bin/bash
#SBATCH --job-name=viz2psy_movie
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-59%10
#SBATCH --output=logs/viz2psy_movie_%A_%a.out
#SBATCH --error=logs/viz2psy_movie_%A_%a.err

# SLURM array job: score each of the 60 movies with viz2psy.
# Each array task processes one movie (~492 frames at 0.5s interval).
# Estimated runtime per movie: ~40 minutes on a single A100.
# %10 throttle: at most 10 concurrent tasks.
# Requires 3g.40gb MIG slice (or larger) — 10gb slices OOM on saliency model.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=/home/bhutch/.local/envs/viz2psy
STIMULI=/gpfs/projects/hulacon/shared/mmmdata/stimuli
MOVIES_DIR="${STIMULI}/movies/movie_files"
SCORES_DIR="${STIMULI}/movies/viz2psy_scores"
SUFFIX="_trimmed_normalized_filtered"
FRAME_INTERVAL=0.5

mkdir -p "${SCORES_DIR}"

# Build sorted movie list and select this task's movie
mapfile -t MOVIES < <(ls "${MOVIES_DIR}"/*.mov | sort)
MOVIE="${MOVIES[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${MOVIE}" ]]; then
    echo "ERROR: No movie found for array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

# Derive clean movie name (strip suffix)
BASENAME=$(basename "${MOVIE}" .mov)
STEM="${BASENAME%${SUFFIX}}"
OUTPUT="${SCORES_DIR}/${STEM}_scores.csv"

echo "=== viz2psy movie extraction ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array:  ${SLURM_ARRAY_TASK_ID} of ${#MOVIES[@]}"
echo "Movie:  ${STEM}"
echo "Input:  ${MOVIE}"
echo "Output: ${OUTPUT}"
echo ""

# Run viz2psy with all models
${VENV}/bin/viz2psy --all "${MOVIE}" \
    -o "${OUTPUT}" \
    --device cuda \
    --frame-interval "${FRAME_INTERVAL}"

echo ""
echo "=== Generating HTML dashboard ==="
${VENV}/bin/viz2psy-viz dashboard "${OUTPUT}" \
    --video-path "${MOVIE}" \
    -o "${SCORES_DIR}/${STEM}_scores_dashboard.html"

echo ""
echo "=== Done ==="
echo "Output: ${OUTPUT}"
echo "Date:   $(date)"

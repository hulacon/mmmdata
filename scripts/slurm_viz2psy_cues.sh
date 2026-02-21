#!/bin/bash
#SBATCH --job-name=viz2psy_cues
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz2psy_cues_%j.out
#SBATCH --error=logs/viz2psy_cues_%j.err

# Run viz2psy on 60 movie cue images with GPU acceleration.

set -euo pipefail

VENV=/home/bhutch/.local/envs/viz2psy
STIMULI=/gpfs/projects/hulacon/shared/mmmdata/stimuli
INPUT="${STIMULI}/movies/movie_cues/*.jpg"
OUTPUT="${STIMULI}/movies/viz2psy_cue_scores.csv"

echo "=== viz2psy cue image extraction ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Run viz2psy with all models
${VENV}/bin/viz2psy --all ${INPUT} -o "${OUTPUT}" --device cuda

echo ""
echo "=== Generating HTML dashboard ==="
${VENV}/bin/viz2psy-viz dashboard "${OUTPUT}" \
    --image-root "${STIMULI}/movies/movie_cues" \
    -o "${STIMULI}/movies/viz2psy_cue_scores_dashboard.html"

echo ""
echo "=== Done ==="
echo "Output: ${OUTPUT}"
echo "Dashboard: ${STIMULI}/movies/viz2psy_cue_scores_dashboard.html"
echo "Date: $(date)"

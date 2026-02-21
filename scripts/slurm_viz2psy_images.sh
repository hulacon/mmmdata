#!/bin/bash
#SBATCH --job-name=viz2psy_images
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/viz2psy_images_%j.out
#SBATCH --error=logs/viz2psy_images_%j.err

# Run viz2psy on 1,000 shared1000 NSD images with GPU acceleration.
# Estimated runtime: ~1.5-2 hours on a single A100.

set -euo pipefail

VENV=/home/bhutch/.local/envs/viz2psy
STIMULI=/gpfs/projects/hulacon/shared/mmmdata/stimuli
INPUT="${STIMULI}/shared1000/images/*.png"
OUTPUT="${STIMULI}/shared1000/viz2psy_scores.csv"

echo "=== viz2psy shared1000 image extraction ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Run viz2psy with all models
${VENV}/bin/viz2psy --all ${INPUT} -o "${OUTPUT}" --device cuda

echo ""
echo "=== Generating HTML dashboard ==="
${VENV}/bin/viz2psy-viz dashboard "${OUTPUT}" \
    --image-root "${STIMULI}/shared1000/images" \
    -o "${STIMULI}/shared1000/viz2psy_scores_dashboard.html"

echo ""
echo "=== Done ==="
echo "Output: ${OUTPUT}"
echo "Dashboard: ${STIMULI}/shared1000/viz2psy_scores_dashboard.html"
echo "Date: $(date)"

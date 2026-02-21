#!/bin/bash
# Submit all viz2psy SLURM jobs for MMMData stimuli.
#
# Usage:
#   bash scripts/submit_viz2psy.sh              # submit all (images + movies)
#   bash scripts/submit_viz2psy.sh images       # images only
#   bash scripts/submit_viz2psy.sh movies       # movies array only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

TARGET="${1:-all}"

submit_images() {
    local job_id
    job_id=$(sbatch --parsable "${SCRIPT_DIR}/slurm_viz2psy_images.sh")
    echo "Submitted images job: ${job_id}"
    echo "  1,000 shared1000 PNGs -> stimuli/shared1000/viz2psy_scores.csv"
}

submit_movies() {
    local array_id
    array_id=$(sbatch --parsable "${SCRIPT_DIR}/slurm_viz2psy_movies.sh")
    echo "Submitted movies array job: ${array_id}"
    echo "  60 movies (array 0-59, max 10 concurrent)"
    echo "  -> stimuli/movies/viz2psy_scores/{movie}_scores.csv"
}

case "${TARGET}" in
    images)
        submit_images
        ;;
    movies)
        submit_movies
        ;;
    all)
        submit_images
        submit_movies
        ;;
    *)
        echo "Usage: $0 [images|movies|all]" >&2
        exit 1
        ;;
esac

echo ""
echo "Monitor with: squeue -u \$USER"

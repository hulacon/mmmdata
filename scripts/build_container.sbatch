#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=build_container
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/build_container_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/build_container_%j.err

#==============================================================================
# Build an Apptainer image into the shared containers directory
#==============================================================================
# Usage:
#   sbatch build_container.sbatch <docker-uri> <image-filename>
#   sbatch build_container.sbatch docker://nipreps/fmriprep:25.2.5 fmriprep-25.2.5.sif
#
# The image lands in $SINGULARITY_DIR (from config/base.toml + local.toml).
# Build cache/tmp live on GPFS beside the target, never in $HOME (quota-bound)
# or /tmp (node-local, small), and are removed on exit.
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

DOCKER_URI="${1:?usage: sbatch build_container.sbatch <docker-uri> <image-filename>}"
IMAGE_NAME="${2:?usage: sbatch build_container.sbatch <docker-uri> <image-filename>}"
TARGET="${SINGULARITY_DIR}/${IMAGE_NAME}"

if [ -e "${TARGET}" ]; then
    echo "ERROR: ${TARGET} already exists; remove it first if a rebuild is intended." >&2
    exit 1
fi

BUILD_TMP="${SINGULARITY_DIR}/.build-${SLURM_JOB_ID}"
mkdir -p "${BUILD_TMP}/cache" "${BUILD_TMP}/tmp"
trap 'rm -rf "${BUILD_TMP}"' EXIT
export APPTAINER_CACHEDIR="${BUILD_TMP}/cache"
export APPTAINER_TMPDIR="${BUILD_TMP}/tmp"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"

# Build to a partial name, then rename: a killed job never leaves a
# plausible-looking but truncated image at the target path.
apptainer build "${TARGET}.partial" "${DOCKER_URI}"
mv "${TARGET}.partial" "${TARGET}"
chmod g+rw "${TARGET}"

echo "Built ${TARGET} ($(du -h "${TARGET}" | cut -f1))"
apptainer inspect "${TARGET}"

echo "Finished $(date -Is)"

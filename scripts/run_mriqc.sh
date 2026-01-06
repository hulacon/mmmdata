#!/bin/bash
#
# run_mriqc.sh - Run MRIQC on all subjects and sessions
#
# This script runs MRIQC quality control on the MMMData BIDS dataset
# using Singularity/Apptainer containerization.
#
# Usage:
#   ./run_mriqc.sh [participant|group]
#
# Arguments:
#   participant - Run participant-level analysis (default)
#   group       - Run group-level analysis (requires participant-level to be complete)
#

# Configuration
BIDS_DIR="/projects/hulacon/shared/mmmdata"
OUTPUT_DIR="${BIDS_DIR}/derivatives/mriqc"
WORK_DIR="${OUTPUT_DIR}/work"
SINGULARITY_DIR="${BIDS_DIR}/singularity_images"
MRIQC_VERSION="24.0.0"
MRIQC_IMAGE="${SINGULARITY_DIR}/mriqc-${MRIQC_VERSION}.simg"

# Analysis level (participant or group)
ANALYSIS_LEVEL="${1:-participant}"

# Number of parallel processes (adjust based on your system resources)
N_PROCS=4

# Memory limit (in GB)
MEM_GB=16

# FreeSurfer license (if you have one, otherwise MRIQC will skip FS-dependent metrics)
# FS_LICENSE_FILE="/path/to/license.txt"

# ===== SETUP =====

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORK_DIR}"
mkdir -p "${SINGULARITY_DIR}"

# Download MRIQC Singularity image if it doesn't exist
if [ ! -f "${MRIQC_IMAGE}" ]; then
    echo "MRIQC Singularity image not found. Downloading..."
    echo "This may take several minutes..."
    singularity pull "${MRIQC_IMAGE}" \
        docker://nipreps/mriqc:${MRIQC_VERSION}

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download MRIQC image"
        exit 1
    fi
    echo "Download complete!"
fi

# ===== RUN MRIQC =====

echo "=========================================="
echo "Running MRIQC ${MRIQC_VERSION}"
echo "=========================================="
echo "BIDS Directory: ${BIDS_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Analysis Level: ${ANALYSIS_LEVEL}"
echo "=========================================="

# Build the singularity command
CMD="singularity run --cleanenv \
    -B ${BIDS_DIR}:${BIDS_DIR}:ro \
    -B ${OUTPUT_DIR}:${OUTPUT_DIR} \
    -B ${WORK_DIR}:${WORK_DIR} \
    ${MRIQC_IMAGE} \
    ${BIDS_DIR} \
    ${OUTPUT_DIR} \
    ${ANALYSIS_LEVEL}"

# Add common options
CMD="${CMD} \
    --work-dir ${WORK_DIR} \
    --nprocs ${N_PROCS} \
    --mem ${MEM_GB} \
    --verbose-reports \
    --no-sub"

# Add FreeSurfer license if available
if [ ! -z "${FS_LICENSE_FILE}" ] && [ -f "${FS_LICENSE_FILE}" ]; then
    CMD="${CMD} --fs-license-file ${FS_LICENSE_FILE}"
fi

# For participant-level: process all participants
# MRIQC will automatically detect all subjects and sessions in the BIDS dataset
if [ "${ANALYSIS_LEVEL}" == "participant" ]; then
    echo "Processing all participants and sessions..."
    echo ""
    echo "Running command:"
    echo "${CMD}"
    echo ""

    # Run MRIQC (direct execution without eval for security)
    ${CMD}

elif [ "${ANALYSIS_LEVEL}" == "group" ]; then
    echo "Generating group-level reports..."
    echo ""
    echo "Running command:"
    echo "${CMD}"
    echo ""

    # Run group-level analysis (direct execution without eval for security)
    ${CMD}

else
    echo "ERROR: Invalid analysis level '${ANALYSIS_LEVEL}'"
    echo "Must be 'participant' or 'group'"
    exit 1
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "MRIQC completed successfully!"
    echo "=========================================="
    echo "Results are in: ${OUTPUT_DIR}"
    echo ""
    echo "Next steps:"
    if [ "${ANALYSIS_LEVEL}" == "participant" ]; then
        echo "  - Review individual reports in: ${OUTPUT_DIR}"
        echo "  - Run group analysis: ./run_mriqc.sh group"
    else
        echo "  - Review group report: ${OUTPUT_DIR}/group_*.html"
    fi
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: MRIQC failed"
    echo "=========================================="
    echo "Check the logs above for error messages"
    exit 1
fi

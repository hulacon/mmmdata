#!/bin/bash
#==============================================================================
# Build BIDS input tree for one subject/session with NORDIC-denoised BOLDs
#==============================================================================
# Creates the BIDS-compliant input directory that fMRIPrep will consume:
#   derivatives/nordic/bids_input/{sub}/{ses}/
#
# - NORDIC-denoised BOLDs: hardlinked (no disk duplication)
# - Sidecars, events, physio, SBRef: copied from raw BIDS
# - Fieldmaps: copied from raw BIDS
#
# Usage (standalone):
#   NORDIC_SUBJECT=sub-03 NORDIC_SESSION=ses-04 bash nordic_build_bids_input.sh
#
# Usage (SLURM — typically as dependency target):
#   sbatch --export=ALL,NORDIC_SUBJECT=sub-03,NORDIC_SESSION=ses-04 \
#          --job-name=bids_tree --mem=4G --time=00:30:00 --cpus-per-task=1 \
#          --partition=compute --account=hulacon \
#          --output=logs/bids_tree_%j.out --error=logs/bids_tree_%j.err \
#          nordic_build_bids_input.sh
#==============================================================================

set -uo pipefail

# Validate required environment variables
if [ -z "${NORDIC_SUBJECT:-}" ] || [ -z "${NORDIC_SESSION:-}" ]; then
    echo "ERROR: NORDIC_SUBJECT and NORDIC_SESSION must be set."
    exit 1
fi

SUB="${NORDIC_SUBJECT}"
SES="${NORDIC_SESSION}"

# Load configuration
export REPO_ROOT="${REPO_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"
source "${REPO_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/load_config.sh"

# Paths
RAW_FUNC="${BIDS_DIR}/${SUB}/${SES}/func"
RAW_FMAP="${BIDS_DIR}/${SUB}/${SES}/fmap"
NORDIC_FUNC="${BIDS_DIR}/derivatives/nordic/${SUB}/${SES}/func"
BIDS_INPUT="${BIDS_DIR}/derivatives/nordic/bids_input"
OUT_FUNC="${BIDS_INPUT}/${SUB}/${SES}/func"
OUT_FMAP="${BIDS_INPUT}/${SUB}/${SES}/fmap"

echo "=========================================="
echo "Building BIDS input tree"
echo "=========================================="
echo "Subject: ${SUB}"
echo "Session: ${SES}"
echo "Raw BIDS: ${RAW_FUNC}"
echo "NORDIC:   ${NORDIC_FUNC}"
echo "Output:   ${BIDS_INPUT}/${SUB}/${SES}/"
echo "=========================================="
echo ""

# --- Validate inputs ---
if [ ! -d "${RAW_FUNC}" ]; then
    echo "ERROR: Raw func directory not found: ${RAW_FUNC}"
    exit 1
fi

if [ ! -d "${NORDIC_FUNC}" ]; then
    echo "ERROR: NORDIC output directory not found: ${NORDIC_FUNC}"
    echo "Run nordic_denoise.sbatch first."
    exit 1
fi

# Count expected vs actual NORDIC outputs
N_RAW=$(ls "${RAW_FUNC}"/*_bold.nii.gz 2>/dev/null | wc -l)
N_NORDIC=$(ls "${NORDIC_FUNC}"/*_bold.nii.gz 2>/dev/null | wc -l)

if [ "${N_RAW}" -ne "${N_NORDIC}" ]; then
    echo "WARNING: Run count mismatch — raw has ${N_RAW} BOLDs, NORDIC has ${N_NORDIC}"
    echo "Proceeding with available NORDIC outputs."
fi

# --- func/ directory ---
mkdir -p "${OUT_FUNC}"

# Hardlink NORDIC-denoised BOLDs
echo "Hardlinking ${N_NORDIC} NORDIC BOLDs..."
for nordic_bold in "${NORDIC_FUNC}"/*_bold.nii.gz; do
    fname=$(basename "${nordic_bold}")
    target="${OUT_FUNC}/${fname}"
    if [ ! -f "${target}" ]; then
        ln "${nordic_bold}" "${target}"
    fi
done

# Copy sidecars and associated files from raw BIDS
# (everything except _bold.nii.gz which comes from NORDIC)
echo "Copying sidecars, events, physio, SBRef from raw BIDS..."
for raw_file in "${RAW_FUNC}"/*; do
    fname=$(basename "${raw_file}")
    target="${OUT_FUNC}/${fname}"

    # Skip raw BOLD NIfTIs (we use NORDIC versions)
    if [[ "${fname}" == *_bold.nii.gz ]]; then
        continue
    fi

    # Skip if already exists
    if [ -f "${target}" ]; then
        continue
    fi

    cp "${raw_file}" "${target}"
done

echo "  func/: $(ls "${OUT_FUNC}" | wc -l) files"

# --- fmap/ directory ---
if [ -d "${RAW_FMAP}" ]; then
    mkdir -p "${OUT_FMAP}"
    echo "Copying fieldmaps..."
    for fmap_file in "${RAW_FMAP}"/*; do
        fname=$(basename "${fmap_file}")
        target="${OUT_FMAP}/${fname}"
        if [ ! -f "${target}" ]; then
            cp "${fmap_file}" "${target}"
        fi
    done
    echo "  fmap/: $(ls "${OUT_FMAP}" | wc -l) files"
else
    echo "  No fieldmaps for ${SES} (skipping fmap/)"
fi

# --- scans.tsv (if present at session level) ---
SCANS_TSV="${BIDS_DIR}/${SUB}/${SES}/${SUB}_${SES}_scans.tsv"
SCANS_TARGET="${BIDS_INPUT}/${SUB}/${SES}/$(basename "${SCANS_TSV}")"
if [ -f "${SCANS_TSV}" ] && [ ! -f "${SCANS_TARGET}" ]; then
    cp "${SCANS_TSV}" "${SCANS_TARGET}"
    echo "  Copied scans.tsv"
fi

echo ""
echo "Done. BIDS input tree ready for fMRIPrep: ${BIDS_INPUT}/${SUB}/${SES}/"

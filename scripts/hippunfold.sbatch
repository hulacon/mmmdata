#!/bin/bash
#SBATCH --job-name=hippunfold
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hippunfold_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hippunfold_%A_%a.err

#==============================================================================
# HippUnfold: Hippocampal Subfield Segmentation & Unfolding (SLURM Job)
#==============================================================================
# Runs HippUnfold on BIDS data using T1w + T2w inputs, with the combined
# T1T2w nnU-net model for best segmentation quality.
#
# Usage:
#   Single subject:   sbatch --export=ALL,SUBJECT=sub-## hippunfold.sbatch
#   Array (all):      sbatch --array=1-N hippunfold.sbatch
#   T2w acquisition:  --export=ALL,SUBJECT=sub-##,T2W_ACQ=oblcor   (default SPC)
#   Version:          --export=ALL,SUBJECT=sub-##,HIPPUNFOLD_VERSION=2.2.0
#                     (default 1.5.2; v2 needs more memory, see below)
#   Another dataset:  --export=ALL,SUBJECT=sub-##,INPUT_BIDS=<bids>,DERIV_ROOT=<dir>,
#                     T1W_FILTER="acquisition=<acq>"
#                     (defaults: this dataset's BIDS root, its derivatives/, and
#                     "acquisition=MPR run=01"; DERIV_ROOT must be outside any
#                     other dataset's raw tree)
#   Resume a killed run: --export=ALL,SUBJECT=sub-##,EXTRA_ARGS="--rerun-incomplete"
#                     (after clearing the tree's stale .snakemake/locks; extra
#                     args are passed through to Snakemake)
#
# Output goes to ${DERIV_ROOT}/hippunfold[_v<major>][_<acq>]/:
#   1.5.2 SPC    -> derivatives/hippunfold/
#   1.5.2 oblcor -> derivatives/hippunfold_oblcor/
#   2.x   SPC    -> derivatives/hippunfold_v2/
#   2.x   oblcor -> derivatives/hippunfold_v2_oblcor/
# Major versions get separate trees because their surfaces and file layout
# differ; downstream code (functional_space/grayordinates.py) reads the 1.5.2
# tree. Snakemake locks the whole output dir, so subjects sharing a tree must
# run one after another (--dependency=afterok:<jobid>).
#
# Memory: 1.5.2 peaks ~4.3 GB (16G is plenty). 2.x's docs put nnU-net
# inference at ~32 GB, so submit v2 with --mem=48G until a pilot says less.
#==============================================================================

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Activate Python virtual environment (needed for config loader)
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

# Load configuration from config files
source "${REPO_ROOT}/scripts/load_config.sh"

# Input dataset + derivative root: this dataset unless overridden
INPUT_BIDS="${INPUT_BIDS:-${BIDS_DIR}}"
DERIV_ROOT="${DERIV_ROOT:-${BIDS_DIR}/derivatives}"
T1W_FILTER="${T1W_FILTER:-acquisition=MPR run=01}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- Determine subject ---
if [ -n "${SUBJECT}" ]; then
    # Explicit subject passed via --export=SUBJECT=sub-XX
    SUB="${SUBJECT}"
elif [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    # Array mode: map task ID to subject
    SUBJECTS=($(ls -d ${INPUT_BIDS}/sub-* | xargs -n 1 basename | sort))
    if [ ${SLURM_ARRAY_TASK_ID} -gt ${#SUBJECTS[@]} ]; then
        echo "ERROR: Array task ID ${SLURM_ARRAY_TASK_ID} exceeds number of subjects (${#SUBJECTS[@]})"
        exit 1
    fi
    SUB=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
else
    echo "ERROR: Specify subject via --export=SUBJECT=sub-XX or use --array"
    exit 1
fi

# Extract participant label (number only, no "sub-" prefix)
PARTICIPANT_LABEL="${SUB#sub-}"

echo "Processing: ${SUB} (participant label: ${PARTICIPANT_LABEL})"
echo ""

# --- Container setup ---
HIPPUNFOLD_VERSION="${HIPPUNFOLD_VERSION:-1.5.2}"
HIPPUNFOLD_MAJOR="${HIPPUNFOLD_VERSION%%.*}"
CONTAINER="${SINGULARITY_DIR}/hippunfold-v${HIPPUNFOLD_VERSION}.sif"
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: HippUnfold container not found at ${CONTAINER}"
    exit 1
fi

# Set up temporary directory for Singularity
export SINGULARITY_TMPDIR="${CODE_ROOT}/tmp/hippunfold_${SLURM_JOB_ID}"
mkdir -p "${SINGULARITY_TMPDIR}"

# Bind paths for container access
export SINGULARITY_BINDPATH="/gpfs:/gpfs"

# HippUnfold model cache (downloaded on first run); one per major version so
# v2's models and atlases never mix with the cache 1.5.2 ran from
if [ "${HIPPUNFOLD_MAJOR}" = "1" ]; then
    export HIPPUNFOLD_CACHE_DIR="${CODE_ROOT}/.cache/hippunfold"
    TREE="hippunfold"
else
    export HIPPUNFOLD_CACHE_DIR="${CODE_ROOT}/.cache/hippunfold_v${HIPPUNFOLD_MAJOR}"
    TREE="hippunfold_v${HIPPUNFOLD_MAJOR}"
fi

# 2.x containers run /app/entrypoint.sh, which execs its first argument, so the
# command name must come first; 1.5.2's runscript IS hippunfold. 2.2.0 also
# reports `--version` as 2.0.0 (upstream hard-codes it in snakebids.yml at the
# v2.2.0 tag); the container's org.opencontainers.image.revision label is the
# reliable provenance.
if [ "${HIPPUNFOLD_MAJOR}" = "1" ]; then
    ENTRY_CMD=()
else
    ENTRY_CMD=(hippunfold)
fi
mkdir -p "${HIPPUNFOLD_CACHE_DIR}"

# T2w acquisition: default SPC, override via --export=T2W_ACQ=oblcor
T2W_ACQ="${T2W_ACQ:-SPC}"

# Output directory: separate dirs per T2w acquisition for comparison
if [ "${T2W_ACQ}" = "SPC" ]; then
    OUTPUT_DIR="${DERIV_ROOT}/${TREE}"
else
    OUTPUT_DIR="${DERIV_ROOT}/${TREE}_${T2W_ACQ}"
fi
mkdir -p "${OUTPUT_DIR}"

# --- Run HippUnfold ---
echo "Running HippUnfold ${HIPPUNFOLD_VERSION} with T1T2w model..."
echo "  Input:  ${INPUT_BIDS}"
echo "  T1w filter: ${T1W_FILTER}"
echo "  Output: ${OUTPUT_DIR}"
echo "  T2w acquisition: ${T2W_ACQ}"
echo "  Container: ${CONTAINER}"
echo ""

singularity run -e \
    --env HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR}" \
    "${CONTAINER}" \
    "${ENTRY_CMD[@]}" \
    "${INPUT_BIDS}" \
    "${OUTPUT_DIR}" \
    participant \
    --participant-label "${PARTICIPANT_LABEL}" \
    --modality T2w \
    --force-nnunet-model T1T2w \
    --t1-reg-template \
    --output-spaces T1w \
    --filter_T1w ${T1W_FILTER} \
    --filter_T2w acquisition="${T2W_ACQ}" \
    --cores ${SLURM_CPUS_PER_TASK} \
    -p ${EXTRA_ARGS}

# Capture exit status
EXIT_STATUS=$?

# --- Completion ---
echo ""
echo "=========================================="
echo "Job Completion Information"
echo "=========================================="
echo "Subject: ${SUB}"
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

# Clean up temporary files
if [ -d "${SINGULARITY_TMPDIR}" ]; then
    echo "Cleaning up temporary files..."
    rm -rf "${SINGULARITY_TMPDIR}"
fi

exit ${EXIT_STATUS}

#!/bin/bash
#SBATCH --job-name=hsf
#SBATCH --account=hulacon
#SBATCH --partition=compute
# 8 h, not 3: HSF spends ~2 h on model fetch + roiloc before the first
# segmentation starts, then ~50 min PER HEMISPHERE. The original 3 h fitted one
# side only, which is why every acq-SPC run and sub-04's oblcor run died mid
# `side 2/2` with CANCELLED ... DUE TO TIME LIMIT and left no left-hemisphere
# segmentation (jobs 42754212-42754214, 42754252, 2026-03-10).
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hsf_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hsf_%A_%a.err

#==============================================================================
# HSF: Hippocampal Segmentation Factory (SLURM Job)
#==============================================================================
# Runs HSF v1.2.3 on T2w images for hippocampal subfield segmentation.
# Provides independent cross-validation against HippUnfold.
#
# Usage:
#   SPC (default):    sbatch --export=SUBJECT=sub-03 hsf.sbatch
#   Oblique-coronal:  sbatch --export=SUBJECT=sub-03,T2W_ACQ=oblcor hsf.sbatch
#   All 3 subjects:   sbatch --array=1-3 hsf.sbatch
#
# Output goes to: derivatives/hsf/ (SPC) or derivatives/hsf_oblcor/ (oblcor)
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

# --- Determine subject ---
if [ -n "${SUBJECT}" ]; then
    SUB="${SUBJECT}"
elif [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    SUBJECTS=($(ls -d ${BIDS_DIR}/sub-* | xargs -n 1 basename | sort))
    if [ ${SLURM_ARRAY_TASK_ID} -gt ${#SUBJECTS[@]} ]; then
        echo "ERROR: Array task ID ${SLURM_ARRAY_TASK_ID} exceeds number of subjects (${#SUBJECTS[@]})"
        exit 1
    fi
    SUB=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
else
    echo "ERROR: Specify subject via --export=SUBJECT=sub-XX or use --array"
    exit 1
fi

echo "Processing: ${SUB}"
echo ""

# --- HSF setup ---
HSF_VENV="/home/bhutch/.local/envs/hsf"
HSF_BIN="${HSF_VENV}/bin/hsf"

if [ ! -f "${HSF_BIN}" ]; then
    echo "ERROR: HSF not found at ${HSF_BIN}"
    exit 1
fi

# T2w acquisition: default SPC, override via --export=T2W_ACQ=oblcor
T2W_ACQ="${T2W_ACQ:-SPC}"

# Input and roiloc config depend on acquisition
INPUT_DIR="${BIDS_DIR}/${SUB}/ses-01/anat"
if [ "${T2W_ACQ}" = "oblcor" ]; then
    INPUT_PATTERN="*acq-oblcor_T2w.nii.gz"
    ROILOC_CFG="default_corot2"
else
    INPUT_PATTERN="*acq-SPC_T2w.nii.gz"
    ROILOC_CFG="default_t2iso"
fi

if ! ls ${INPUT_DIR}/${INPUT_PATTERN} 1>/dev/null 2>&1; then
    echo "ERROR: No T2w file found matching ${INPUT_DIR}/${INPUT_PATTERN}"
    exit 1
fi

# Output directory: separate dirs per T2w acquisition
if [ "${T2W_ACQ}" = "SPC" ]; then
    OUTPUT_DIR="${BIDS_DIR}/derivatives/hsf/${SUB}"
else
    OUTPUT_DIR="${BIDS_DIR}/derivatives/hsf_${T2W_ACQ}/${SUB}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "Input directory: ${INPUT_DIR}"
echo "Input pattern: ${INPUT_PATTERN}"
echo "T2w acquisition: ${T2W_ACQ}"
echo "ROILoc config: ${ROILOC_CFG}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# --- Run HSF ---
echo "Running HSF (bagging_accurate, T2w ${T2W_ACQ})..."
echo ""

# Limit onnxruntime thread count to allocated CPUs (prevents 100+ thread spawning)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${OUTPUT_DIR}"

# NOTE on resume: HSF's own `overwrite: false` is unsafe here. filter_mris ->
# _get_segmentations skips an MRI when it finds *any* *_hippocampus_seg.nii.gz
# beside it (`len(...) == 0` at factory.py:177), so ONE hemisphere is enough to
# mark a subject done. Every partially-timed-out run in this dataset would have
# been silently skipped by a re-run. Now that segmentations are relocated to
# derivatives/ that trap is dead — the raw dir is always empty, nothing is ever
# skipped, and a re-run genuinely redoes both hemispheres. Do not "optimise"
# this by re-adding a skip-if-output-exists check without handling the
# one-of-two case.
#
# files.output_dir MUST be absolute. HSF resolves it as
# `original_mri_path.parent / dir_name` (roiloc_wrapper.save_hippocampi), so the
# default relative "hsf_outputs" lands inside the raw BIDS anat/ tree. An
# absolute path wins the pathlib join and keeps the crops out of the raw tree.
"${HSF_BIN}" \
    files.path="${INPUT_DIR}/" \
    files.pattern="${INPUT_PATTERN}" \
    files.output_dir="${OUTPUT_DIR}/hsf_outputs" \
    roiloc="${ROILOC_CFG}" \
    segmentation=bagging_accurate \
    segmentation.ca_mode="1/2/3" \
    hardware=onnxruntime \
    'hardware.engine_settings.execution_providers=["CPUExecutionProvider"]' \
    2>&1

EXIT_STATUS=$?

# --- Relocate the native-space segmentation out of the raw BIDS tree ---
# This one is NOT configurable. hsf/factory.py:137 does `output_path =
# mri.parent / fname` against the ORIGINAL T2w, so the native-space
# *_hippocampus_seg.nii.gz always lands in the raw anat/ dir no matter what
# files.output_dir says. Moving it is the only fix. Scoped to this run's acq so
# a concurrent run of the other acquisition is never touched.
echo ""
echo "Relocating native-space segmentations out of ${INPUT_DIR}..."
shopt -s nullglob
STRAYS=("${INPUT_DIR}"/${SUB}_ses-01_acq-${T2W_ACQ}_T2w_*_hippocampus_seg.nii.gz)
shopt -u nullglob
if [ ${#STRAYS[@]} -eq 0 ]; then
    echo "  none found (expected 2: left + right)"
else
    # Overwrite deliberately. These files were produced by THIS job, for this
    # subject and this acquisition (the glob guarantees both), so anything
    # already at the destination is the same product from an older run. HSF uses
    # 20x test-time augmentation, so a re-run's output differs slightly — and a
    # left hemisphere from today paired with a right hemisphere from March is
    # worse than replacing both. Refusing here would be the unsafe option: it
    # strands this run's output in the raw BIDS tree, which is the exact leak
    # this block exists to stop.
    for f in "${STRAYS[@]}"; do
        dest="${OUTPUT_DIR}/$(basename "${f}")"
        [ -e "${dest}" ] && echo "  replacing older ${dest}"
        mv -f -v "${f}" "${dest}"
    done
fi

# --- List outputs ---
echo ""
echo "=========================================="
echo "Output files:"
echo "=========================================="
find "${OUTPUT_DIR}" -name "*.nii*" -type f 2>/dev/null

echo ""
echo "Raw BIDS anat/ must be clean of HSF products — anything listed below is a leak:"
find "${INPUT_DIR}" -maxdepth 1 \( -name "*hippocampus*" -o -name "*_seg*" -o -name "hsf_outputs" \) 2>/dev/null

echo ""
echo "=========================================="
echo "Job Completion Information"
echo "=========================================="
echo "Subject: ${SUB}"
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}

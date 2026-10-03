#!/bin/bash
#SBATCH --job-name=after_conversion
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --dependency=singleton
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/after_conversion_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/after_conversion_%j.err

#==============================================================================
# After conversion: the dataset-side steps for sessions duckbrain converted
#==============================================================================
# duckbrain converts what comes off the scanner (MR data). Everything else a
# session needs stays here, by the ownership line in the constellation
# contracts. Run this after duckbrain's `converted` stage, and before MRIQC
# on anatomy (MRIQC must see defaced images).
#
#   1. Deface: any raw anat in the subject's tree (idempotent; --raw-only
#      when the subject has no fMRIPrep yet), then audit the whole tree.
#   2. Physio triage: re-parse every PhysioLog in mmmsourcedata.
#   3. Inventory: regenerate file_inventory.csv. Physio rows pair with a
#      converted BOLD, which is why this follows conversion, not precedes it.
#   4. Convert events / beh / physio for the named sessions only (4b: the
#      early-session localizers, which the inventory does not route). Every
#      converter overwrites, so --sessions keeps other sessions untouched.
#   5. scans.tsv for the named sessions.
#   6. Report each BOLD run's events and physio, so a gap is printed.
#
# Steps 2-3 rewrite tables shared by every job; --dependency=singleton keeps
# two of these jobs from running at once.
#
# Usage (from the mmmdata repo root):
#   sbatch scripts/after_conversion.sbatch sub-## ses-## [ses-## ...]
#
# Sessions are arguments, not an --export variable: --export splits its value
# on commas, so SESSIONS=ses-##,ses-## arrives as the first session only.
# REPO_ROOT (optional, via --export) overrides the submit directory.
#==============================================================================

set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "ERROR: usage: sbatch scripts/after_conversion.sbatch sub-## ses-## [ses-## ...]" >&2
    exit 1
fi
SUBJECT="$1"; shift
SES_LIST="$*"
SESSIONS="${SES_LIST// /,}"
REPO_ROOT="$(cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}" && pwd)"
if [ ! -f "${REPO_ROOT}/scripts/load_config.sh" ]; then
    echo "ERROR: ${REPO_ROOT} is not the mmmdata repo; pass REPO_ROOT or submit from it" >&2
    exit 1
fi

source "${REPO_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/load_config.sh" || exit 1

VENV_PY="${REPO_ROOT}/.venv/bin/python"
# pydicom (triage, physio) lives in the shared stimfeat env, not the venv;
# behavioral_convert.sbatch and physio_triage.sbatch run the converters there.
STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
STIM_PY="${STIMFEAT}/bin/python3"
CONVERTER_DIR="${REPO_ROOT}/src/python/raw2bids_converters"
SUB_LABEL="${SUBJECT#sub-}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Subject: ${SUBJECT}  Sessions: ${SES_LIST}"
echo "BIDS: ${BIDS_DIR}"

step() { echo; echo "=== $* ==="; }
fail() { echo "FAILED: $*" >&2; exit 1; }

for SES in ${SES_LIST}; do
    [ -d "${BIDS_DIR}/${SUBJECT}/${SES}" ] || fail "${SUBJECT}/${SES} is not in the BIDS tree; convert it first"
done

# --- 1. Deface ---------------------------------------------------------------
step "1. Deface"
ANAT_COUNT=0
for SES in ${SES_LIST}; do
    N=$(find "${BIDS_DIR}/${SUBJECT}/${SES}/anat" \( -name "*_T1w.nii.gz" -o -name "*_T2w.nii.gz" \) 2>/dev/null | wc -l)
    ANAT_COUNT=$((ANAT_COUNT + N))
done
if [ "${ANAT_COUNT}" -gt 0 ]; then
    DEFACE_FLAGS=""
    [ -d "${OUTPUT_DIR}/fmriprep/${SUBJECT}" ] || DEFACE_FLAGS="--raw-only"
    echo "${ANAT_COUNT} anat NIfTIs in these sessions; deface_anat.py run ${DEFACE_FLAGS}"
    module load fsl/6.0.7.9 2>/dev/null || true
    export FSLOUTPUTTYPE=NIFTI_GZ
    # `module load fsl` puts FSL's own python ahead of the venv on PATH, so
    # the interpreter is named absolutely.
    "${VENV_PY}" "${REPO_ROOT}/scripts/deface_anat.py" run --subject "${SUB_LABEL}" ${DEFACE_FLAGS} \
        || fail "deface_anat.py run"
else
    echo "no anat in these sessions"
fi
"${VENV_PY}" "${REPO_ROOT}/scripts/deface_anat.py" audit || fail "deface_anat.py audit"

# --- 2-4. Converters ---------------------------------------------------------
source "${STIMFEAT}/etc/conda/activate.d/stimfeat-env.sh"  # PYTHONNOUSERSITE=1
cd "${CONVERTER_DIR}" || fail "cd ${CONVERTER_DIR}"

step "2. Physio triage"
"${STIM_PY}" generate_physio_triage.py || fail "generate_physio_triage.py"

step "3. Inventory"
"${STIM_PY}" generate_inventory.py || fail "generate_inventory.py"

step "4. Convert events / beh / physio"
"${STIM_PY}" run_all.py --subjects "${SUBJECT}" --sessions "${SESSIONS}" --verbose \
    || fail "run_all.py"

# The inventory routes only the final session's localizers (ses-30's
# behavioral/{motor,auditory}_localizer/). Early-session localizers sit flat
# in behavioral/ and go straight to their converters, as sub-06/07's did.
step "4b. Localizer events (fLoc .mat; motor / tone / auditory timing CSVs)"
for SES in ${SES_LIST}; do
    BEH="${SOURCE_DIR}/${SUBJECT}/${SES}/behavioral"
    MATS=$(ls "${BEH}"/*_fLoc_oddball_run*.mat 2>/dev/null)
    if [ -n "${MATS}" ]; then
        "${STIM_PY}" floc_events.py ${MATS} || fail "floc_events.py ${SUBJECT} ${SES}"
    fi
    for CSV in "${BEH}"/localizer_{motor,tone,auditory}_*_timing.csv; do
        [ -e "${CSV}" ] || continue
        "${STIM_PY}" localizer_events.py "${CSV}" || fail "localizer_events.py ${CSV}"
    done
done

# --- 5. scans.tsv -----------------------------------------------------------
step "5. scans.tsv"
"${VENV_PY}" "${REPO_ROOT}/scripts/build_scans_tsv.py" --subjects "${SUBJECT}" --sessions ${SES_LIST} \
    || fail "build_scans_tsv.py"

# --- 6. Report --------------------------------------------------------------
step "6. Per-run report (events / physio beside each BOLD)"
for SES in ${SES_LIST}; do
    for BOLD in "${BIDS_DIR}/${SUBJECT}/${SES}"/func/*_bold.nii.gz; do
        [ -e "${BOLD}" ] || { echo "${SUBJECT} ${SES}: no BOLD"; continue; }
        STEM="${BOLD%_bold.nii.gz}"
        EV=$([ -f "${STEM}_events.tsv" ] && echo events || echo NO-events)
        PH=$(ls "${STEM}"_recording-*_physio.tsv.gz 2>/dev/null | sed 's/.*_recording-\([^_]*\)_physio.*/\1/' | paste -sd+ -)
        echo "$(basename "${STEM}")  ${EV}  physio=${PH:-NONE}"
    done
done

echo
echo "Done at $(date)"

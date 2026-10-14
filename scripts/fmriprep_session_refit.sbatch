#!/bin/bash
#SBATCH --job-name=fmriprep_refit
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_refit_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_refit_%j.err

#==============================================================================
# fMRIPrep functional REFIT of one session — anatomy reused, func fit redone
#==============================================================================
# For a session whose functional fit must change (e.g. a run's B0FieldSource
# was corrected in the raw sidecars) while the subject's anatomy stays exactly
# as the campaign built it.
#
# Why not --derivatives fmriprep=<tree> (as fmriprep_t1w_backfill.sbatch does):
# fMRIPrep would then reuse the session's own precomputed func outputs too
# (coreg boldref, boldref->anat xfm), i.e. keep the very fit being replaced.
# Here --derivatives points at an ANAT-ONLY view of the tree (symlinks to
# dataset_description.json + sub-##/anat), so the anatomical preproc and its
# transforms and (via --fs-subjects-dir) the FreeSurfer recon are reused, and
# the fieldmap estimation, HMC, SDC, coregistration and all resampling rerun.
#
# Flags mirror the campaign's per-session invocation (duckbrain advance_one,
# 25.2.5) plus the 2026-09-08 backfill, so the output carries all five spaces.
# Output goes to a STAGING tree by default; swapping it into the production
# tree is a separate, checked step.
#
# Usage:
#   sbatch --export=ALL,SUBJECT=##,SESSION=## fmriprep_session_refit.sbatch
# Env: SUBJECT, SESSION (bare labels, required); OUT_ROOT (default: a staging
# dir on hulacon scratch); SPACES (default: the five production spaces).
#==============================================================================

set -u

echo "=== fMRIPrep session refit: sub-${SUBJECT:?} ses-${SESSION:?} ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/load_config.sh"

FMRIPREP_TREE="${OUTPUT_DIR}/fmriprep"                 # the campaign tree (read)
FS_SUBJECTS_DIR="${FMRIPREP_TREE}/sourcedata/freesurfer"
CONTAINER="${SINGULARITY_DIR}/fmriprep-25.2.5.sif"
FS_LICENSE="${CODE_ROOT}/../fs_license.txt"
SPACES="${SPACES:-MNI152NLin2009cAsym:res-2 fsaverage:den-41k func T1w fsnative}"
OUT_ROOT="${OUT_ROOT:-/gpfs/scratch/hulacon/${USER}/refit/fmriprep}"

# Short on purpose (path-buffer trap, see the campaign record); per-unit.
WORK_DIR="/gpfs/scratch/hulacon/dbw/refit/sub-${SUBJECT}_ses-${SESSION}"
export SINGULARITYENV_TEMPLATEFLOW_HOME="$WORK_DIR/templateflow"
export SINGULARITY_TMPDIR="$WORK_DIR/tmp"
mkdir -p "$WORK_DIR" "$SINGULARITYENV_TEMPLATEFLOW_HOME" "$SINGULARITY_TMPDIR" "$OUT_ROOT"

for p in "$CONTAINER" "$FS_LICENSE" "$FMRIPREP_TREE/dataset_description.json" \
         "$FMRIPREP_TREE/sub-${SUBJECT}/anat" \
         "$FS_SUBJECTS_DIR/sub-${SUBJECT}/scripts/recon-all.done"; do
    [ -e "$p" ] || { echo "ERROR: missing $p"; exit 2; }
done

# Anat-only derivatives view (see header). Symlinks resolve inside the
# container because the BIDS root, which holds the tree, is bound at its path.
ANAT_VIEW="$WORK_DIR/anat_view"
rm -rf "$ANAT_VIEW"; mkdir -p "$ANAT_VIEW/sub-${SUBJECT}"
ln -s "$FMRIPREP_TREE/dataset_description.json" "$ANAT_VIEW/dataset_description.json"
ln -s "$FMRIPREP_TREE/sub-${SUBJECT}/anat" "$ANAT_VIEW/sub-${SUBJECT}/anat"

# Same per-session filter duckbrain writes: func-side suffixes only, anat unfiltered.
FILTER="$WORK_DIR/bids_filter.json"
cat > "$FILTER" <<JSON
{
  "bold": {"session": "${SESSION}"},
  "sbref": {"session": "${SESSION}"},
  "fmap": {"session": "${SESSION}"}
}
JSON

ATTEMPT_STAMP="$WORK_DIR/.attempt"; touch "$ATTEMPT_STAMP"

# Binds: BIDS root read-only (covers the campaign tree, so nothing here can
# write into production); output + work read-write.
singularity run --cleanenv --env PYTHONNOUSERSITE=1 \
  -B "${BIDS_DIR}:${BIDS_DIR}:ro" \
  -B "${OUT_ROOT}:${OUT_ROOT}" \
  -B "$WORK_DIR:$WORK_DIR" \
  -B "${CODE_ROOT}/..:${CODE_ROOT}/..:ro" \
  "$CONTAINER" \
  "${BIDS_DIR}" "${OUT_ROOT}" participant \
  --participant-label "${SUBJECT}" \
  --bids-filter-file "$FILTER" \
  --output-spaces ${SPACES} \
  --fs-license-file "$FS_LICENSE" \
  --fs-subjects-dir "$FS_SUBJECTS_DIR" \
  --nprocs "${SLURM_CPUS_PER_TASK}" \
  --mem-mb $(( SLURM_MEM_PER_NODE - 8192 )) \
  -w "$WORK_DIR" \
  --derivatives "anat=${ANAT_VIEW}" \
  --fs-no-resume \
  --no-track-sessions --subject-anatomical-reference unbiased \
  --skip-bids-validation --notrack
EXIT_CODE=$?

echo "Finished: $(date)  Exit code: $EXIT_CODE"
echo "--- outputs under $OUT_ROOT/sub-${SUBJECT}/ses-${SESSION}:"
ls "$OUT_ROOT/sub-${SUBJECT}/ses-${SESSION}/func" 2>/dev/null | wc -l

# Keep the nipype cache on any failure or crash file; it is the evidence.
if [ "$EXIT_CODE" -eq 0 ] &&
   [ -z "$(find "$OUT_ROOT" -maxdepth 2 -name 'crash-*' -newer "$ATTEMPT_STAMP" 2>/dev/null)" ]; then
  rm -rf "$WORK_DIR"
else
  echo "work dir preserved: $WORK_DIR"
fi
exit $EXIT_CODE

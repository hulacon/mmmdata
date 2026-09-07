#!/bin/bash
#SBATCH --job-name=fmriprep_t1w
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_t1w_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_t1w_%j.err

#==============================================================================
# fMRIPrep T1w-space backfill — one session, fit reused, resampling only
#==============================================================================
# Adds the `T1w` output space to an fMRIPrep tree that was run without it,
# WITHOUT redoing the fit. fMRIPrep >= 23.2 reuses its own precomputed
# derivatives when --derivatives points at the prior tree: the hmc/coreg
# boldrefs, the head-motion, boldref->anat and boldref->fieldmap transforms,
# the anatomical preproc + transforms, and (via --fs-subjects-dir) the
# FreeSurfer recon. Only the single-shot resample to the new space runs, so
# the product is a genuine fMRIPrep T1w-space BOLD, not a second interpolation.
#
# The fMRIPrep flags mirror the re-preprocessing campaign's per-session
# invocation (duckbrain advance_one, 25.2.5) exactly, except:
#   * --output-spaces T1w            (only the space being added)
#   * --derivatives fmriprep=<tree>  (named; the tree is read-only here)
#   * --fs-subjects-dir <tree>/sourcedata/freesurfer
#                                    (needed whenever OUT_ROOT != the tree)
#   * -w on a short scratch path     (FS 7.3.2 path-buffer trap: keep short)
#
# Usage (pilot into a staging tree, the default):
#   sbatch --export=ALL,SUBJECT=03,SESSION=30 fmriprep_t1w_backfill.sbatch
# Production into the real tree, once the pilot is vetted:
#   sbatch --export=ALL,SUBJECT=03,SESSION=30,OUT_ROOT=<derivatives>/fmriprep \
#          fmriprep_t1w_backfill.sbatch
# Env: SUBJECT, SESSION (bare labels, required); OUT_ROOT (fMRIPrep output
# dir, default: a staging dir on hulacon scratch); SPACES (default "T1w").
#==============================================================================

set -u

# Array mode: UNIT_LIST is a file of "<sub> <ses>" lines (bare labels); task N
# takes line N. Submit with e.g.
#   sbatch --array=1-100%6 --output=.../fmriprep_t1w_%A_%a.out \
#          --export=ALL,UNIT_LIST=<file>,OUT_ROOT=<tree> fmriprep_t1w_backfill.sbatch
if [ -n "${UNIT_LIST:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r SUBJECT SESSION < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$UNIT_LIST")
fi

echo "=== fMRIPrep T1w backfill: sub-${SUBJECT:?} ses-${SESSION:?} ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/load_config.sh"

FMRIPREP_TREE="${OUTPUT_DIR}/fmriprep"                 # the campaign tree (read)
FS_SUBJECTS_DIR="${FMRIPREP_TREE}/sourcedata/freesurfer"
CONTAINER="${SINGULARITY_DIR}/fmriprep-25.2.5.sif"
FS_LICENSE="${CODE_ROOT}/../fs_license.txt"
SPACES="${SPACES:-T1w}"
OUT_ROOT="${OUT_ROOT:-/gpfs/scratch/hulacon/${USER}/t1w_pilot/fmriprep}"

# Short on purpose (path-buffer trap, see the campaign record); per-unit so
# concurrent sessions never share a nipype cache.
WORK_DIR="/gpfs/scratch/hulacon/dbw/t1w/sub-${SUBJECT}_ses-${SESSION}"
export SINGULARITYENV_TEMPLATEFLOW_HOME="$WORK_DIR/templateflow"
export SINGULARITY_TMPDIR="$WORK_DIR/tmp"
mkdir -p "$WORK_DIR" "$SINGULARITYENV_TEMPLATEFLOW_HOME" "$SINGULARITY_TMPDIR" "$OUT_ROOT"

# Same per-session filter duckbrain writes: func-side suffixes only, anat unfiltered.
FILTER="$WORK_DIR/bids_filter.json"
cat > "$FILTER" <<JSON
{
  "bold": {"session": "${SESSION}"},
  "sbref": {"session": "${SESSION}"},
  "fmap": {"session": "${SESSION}"}
}
JSON

for p in "$CONTAINER" "$FS_LICENSE" "$FMRIPREP_TREE/dataset_description.json" \
         "$FS_SUBJECTS_DIR/sub-${SUBJECT}/scripts/recon-all.done"; do
    [ -e "$p" ] || { echo "ERROR: missing $p"; exit 2; }
done

ATTEMPT_STAMP="$WORK_DIR/.attempt"; touch "$ATTEMPT_STAMP"

# Binds: BIDS root read-only (this also covers the campaign tree, so fMRIPrep
# cannot write into production from here); output + work read-write. When
# OUT_ROOT is the tree itself, the rw bind of OUT_ROOT overlaps the ro bind of
# the BIDS root — Singularity resolves the more specific (rw) one, which is the
# campaign's own self-reuse layout.
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
  --derivatives "fmriprep=${FMRIPREP_TREE}" \
  --fs-no-resume \
  --no-track-sessions --subject-anatomical-reference unbiased \
  --skip-bids-validation --notrack
EXIT_CODE=$?

echo "Finished: $(date)  Exit code: $EXIT_CODE"
echo "--- T1w-space outputs under $OUT_ROOT/sub-${SUBJECT}/ses-${SESSION}/func:"
ls "$OUT_ROOT/sub-${SUBJECT}/ses-${SESSION}/func" 2>/dev/null | grep "space-T1w" || echo "(none)"

# Keep the nipype cache on any failure or crash file; it is the evidence.
if [ "$EXIT_CODE" -eq 0 ] &&
   [ -z "$(find "$OUT_ROOT" -maxdepth 2 -name 'crash-*' -newer "$ATTEMPT_STAMP" 2>/dev/null)" ]; then
  rm -rf "$WORK_DIR"
else
  echo "work dir preserved: $WORK_DIR"
fi
exit $EXIT_CODE

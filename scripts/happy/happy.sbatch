#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=happy
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/happy_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/happy_%A_%a.err

#==============================================================================
# rapidtide `happy`: cardiac waveform from BOLD, one run per array task
#==============================================================================
# Input is the RAW BOLD (no slice-timing correction, which happy needs), with
# slice times from its BIDS sidecar and fMRIPrep's confounds TSV as the motion
# regressors (happy reads its trans_*/rot_* columns).
#
# Usage:
#   sbatch --array=1-N happy.sbatch <runs.txt> <out_dir> [pure|ref] [happy options...]
#
#   runs.txt  one BOLD path per line, relative to BIDS_DIR
#             (sub-##/ses-##/func/sub-##_ses-##_task-X[_run-##]_bold.nii.gz)
#   out_dir   outputs land in <out_dir>/<mode>/sub-##/ses-##/
#   mode      pure (default): cardiac estimated from BOLD alone.
#             ref: also pass the run's recording-pulse physio as --cardiacfile,
#             so happy reports its own BOLD-vs-pulse correlation and lag
#             (corrcoeff_raw2pleth / delay_raw2pleth in _desc-runinfo.json).
#             In ref mode later passes use the pulse, so only `pure` outputs
#             stand for what a run without physio would get.
#   options   anything after the mode is passed to happy unchanged
#             (e.g. --maxhr 100). Give each setting its own out_dir.
#
# Env: shared/envs/rapidtide, built by setup_env.sh beside this file.
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

RUNS=${1:?usage: happy.sbatch <runs.txt> <out_dir> [pure|ref]}
OUT=${2:?usage: happy.sbatch <runs.txt> <out_dir> [pure|ref]}
MODE=${3:-pure}
EXTRA=("${@:4}")
[[ "$MODE" == pure || "$MODE" == ref ]] || { echo "mode must be pure or ref, got: $MODE" >&2; exit 2; }

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "$REPO_ROOT/scripts/load_config.sh"
set -e

ENV=/gpfs/projects/hulacon/shared/envs/rapidtide
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"

TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
REL=$(sed -n "${TASK}p" "$RUNS")
[[ -n "$REL" ]] || { echo "no line $TASK in $RUNS" >&2; exit 2; }

BOLD="$BIDS_DIR/$REL"
SIDECAR="${BOLD%.nii.gz}.json"
BASE=$(basename "${BOLD%_bold.nii.gz}")
SUBSES=$(dirname "$(dirname "$REL")")              # sub-##/ses-##
CONFOUNDS="$OUTPUT_DIR/fmriprep/$SUBSES/func/${BASE}_desc-confounds_timeseries.tsv"
PULSE="$(dirname "$BOLD")/${BASE}_recording-pulse_physio.json"

for f in "$BOLD" "$SIDECAR" "$CONFOUNDS"; do
  [[ -f "$f" ]] || { echo "missing input: $f" >&2; exit 1; }
done

DEST="$OUT/$MODE/$SUBSES"
mkdir -p "$DEST"

# --discardmotionfiltered: the motion-filtered BOLD copy is ~75 % of the output
# (~750 MB of ~1 GB per run) and nothing downstream reads it.
ARGS=(--nprocs "$SLURM_CPUS_PER_TASK" --motionfile "$CONFOUNDS" --noprogressbar --discardmotionfiltered)
if [[ "$MODE" == ref ]]; then
  [[ -f "$PULSE" ]] || { echo "ref mode needs the pulse physio: $PULSE" >&2; exit 1; }
  # The sidecar supplies SamplingFrequency and StartTime (BIDS: seconds of the
  # first sample relative to volume 0); happy reads both from it.
  ARGS+=(--cardiacfile "$PULSE:cardiac")
fi
ARGS+=("${EXTRA[@]}")

echo "Job ${SLURM_JOB_ID}_${TASK} on $(hostname), started $(date -Is)"
echo "mode   : $MODE"
echo "bold   : $BOLD"
echo "motion : $CONFOUNDS"
echo "extra  : ${EXTRA[*]:-none}"
"$ENV/bin/python" -c 'import rapidtide, torch; print("rapidtide", rapidtide.__version__, "| torch", torch.__version__)'

"$ENV/bin/happy" "$BOLD" "$SIDECAR" "$DEST/$BASE" "${ARGS[@]}"

echo "finished $(date -Is)"

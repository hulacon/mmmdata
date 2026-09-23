#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=sidecar_refresh
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/sidecar_refresh_%x_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/sidecar_refresh_%x_%j.err

#==============================================================================
# Contract B sidecar refresh — Talapas wrapper
#==============================================================================
# Runs `<extractor> sidecar refresh` (aud2psy / viz2psy / word2psy), which
# rewrites .meta.json sidecars to the current Contract B schema (1.1: per-model
# `nulls`) without touching a CSV, and refuses any sidecar whose tables hold
# NaN in an undeclared column. Single-threaded and I/O-bound: it reads every
# family's tables back from GPFS.
#
#   sbatch -J viz_dry stimfeat_sidecar_refresh.sbatch viz2psy --dry-run
#   sbatch -J aud_apply stimfeat_sidecar_refresh.sbatch aud2psy
#   sbatch stimfeat_sidecar_refresh.sbatch word2psy --dry-run /some/tree
#
# With no path arguments it covers both trees: the psytwill fit corpora and
# the dataset's stimulus-feature store. Refused sidecars are listed on stderr;
# exit status is 1 if any were refused. No email flags by standing preference.
#==============================================================================

set -uo pipefail
echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"
echo "Args: $*"

USAGE="usage: stimfeat_sidecar_refresh.sbatch aud2psy|viz2psy|word2psy [--dry-run] [PATH...]"
EXTRACTOR="${1:?${USAGE}}"
shift
case "${EXTRACTOR}" in
    aud2psy|viz2psy|word2psy) ;;
    *) echo "unknown extractor '${EXTRACTOR}': expected aud2psy, viz2psy or word2psy" >&2; exit 2 ;;
esac

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done

FIT_CORPORA="${PSYTWILL_FITCORPUS_DURABLE:-/gpfs/projects/hulacon/shared/psytwill/fit-corpora}"
STORE="${BIDS_DIR}/derivatives/stimuli_features"

ARGS=("$@")
has_path=0
for a in "${ARGS[@]}"; do [[ "${a}" != -* ]] && has_path=1; done
if [[ ${has_path} -eq 0 ]]; then
    ARGS+=("${FIT_CORPORA}" "${STORE}")
fi

echo "=== versions ==="
"${STIMFEAT}/bin/python" -c "import ${EXTRACTOR}; print('${EXTRACTOR}', ${EXTRACTOR}.__version__)"

echo "=== refresh ==="
rc=0
"${STIMFEAT}/bin/${EXTRACTOR}" sidecar refresh "${ARGS[@]}" || rc=$?

echo "Finished $(date -Is) rc=${rc}"
exit ${rc}

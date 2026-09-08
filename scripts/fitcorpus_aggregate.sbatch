#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=fc_aggregate
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fitcorpus_%x_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fitcorpus_%x_%j.err

#==============================================================================
# psytwill-space fit-corpus aggregation — Talapas wrapper
#==============================================================================
# Runs psytwill's portable driver `scripts/fitcorpus/extract.py aggregate`,
# which builds one Contract B `features` parquet per (corpus, source, table)
# group under <durable>/<corpus>/groups/. CPU-only; arguments pass through:
#
#   sbatch fitcorpus_aggregate.sbatch --corpus nsd
#   sbatch fitcorpus_aggregate.sbatch --corpus movie10 --source audio
#
# Refuses partial groups unless --allow-partial. No email flags by standing
# preference.
#==============================================================================

set -uo pipefail
echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"
echo "Args: $*"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
PY="${STIMFEAT}/bin/python"

export PSYTWILL_FITCORPUS_DURABLE="${PSYTWILL_FITCORPUS_DURABLE:-/gpfs/projects/hulacon/shared/psytwill/fit-corpora}"
export PSYTWILL_FITCORPUS_SCRATCH="${PSYTWILL_FITCORPUS_SCRATCH:-/gpfs/scratch/hulacon/bhutch/psytwill-fitcorpus}"
export PSYTWILL_NSD_HDF5="${PSYTWILL_NSD_HDF5:-/gpfs/projects/hulacon/shared/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5}"

DRIVER=/gpfs/projects/hulacon/shared/mmmdata/code/psytwill/scripts/fitcorpus/extract.py

echo "=== versions ==="
"${PY}" -c 'import psytwill; print("psytwill", psytwill.__version__)'

echo "=== aggregate ==="
rc=0
"${PY}" "${DRIVER}" aggregate "$@" || rc=$?

echo "Finished $(date -Is) rc=${rc}"
exit ${rc}

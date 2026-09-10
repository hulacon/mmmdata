#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=space_fit
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/space_fit_%x_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/space_fit_%x_%j.err

#==============================================================================
# psytwill-space private-block fit — Talapas wrapper around `psytwill space`
#==============================================================================
# Every argument passes through to `psytwill space` (fit | project | check):
#
#   sbatch space_fit.sbatch fit --block V \
#       --features <durable>/nsd/groups/nsd_image_features.parquet \
#       --exclude-ids <durable>/nsd/inputs/shared1000_ext_ids.txt \
#       -o <durable>/nsd/space
#
# Threads are pinned to the allocation so BLAS does not oversubscribe.
# No email flags by standing preference.
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== versions ==="
"${STIMFEAT}/bin/python" -c 'import psytwill; print("psytwill", psytwill.__version__)'

echo "=== psytwill space $1 ==="
rc=0
"${STIMFEAT}/bin/psytwill" space "$@" || rc=$?

echo "Finished $(date -Is) rc=${rc}"
exit ${rc}

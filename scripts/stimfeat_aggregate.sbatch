#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=stimfeat_agg
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_%x_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_%x_%j.err

#==============================================================================
# Contract B §4.2 — psytwill aggregation (settles-when #5)
#==============================================================================
# Thin wrapper around `stimfeat_campaign.py aggregate`; every argument is
# passed straight through, so a job is described by its filters.
#
#   sbatch stimfeat_aggregate.sbatch                     # every group
#   sbatch stimfeat_aggregate.sbatch --set movies        # one set
#   sbatch stimfeat_aggregate.sbatch --redo              # rebuild existing
#
# CPU only, and deliberately not on the gpu partition: aggregation is a melt
# and a parquet write, no model ever loads.
#
# 32G is generous. psytwill 0.5.0 streams one input at a time, so peak memory
# tracks the largest single input (~7 GB here), not the 302 M-row total that
# the pre-0.5.0 concat would have needed (~135 GB). If a future input dwarfs
# the rest, raise --mem rather than sharding the group: sharding would split
# a group's key space and defeat the duplicate-key check.
#
# Idempotent: a group whose parquet exists is skipped unless --redo.
#==============================================================================

set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"
echo "Filters: $*"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
PY="${STIMFEAT}/bin/python"

DRIVER="${REPO_ROOT}/scripts/stimfeat_campaign.py"

echo "=== groups ==="
"${PY}" "${DRIVER}" aggregate --dry-run "$@"

echo "=== build ==="
rc=0
"${PY}" "${DRIVER}" aggregate "$@" || rc=$?

echo "Finished $(date -Is) rc=${rc}"
exit ${rc}

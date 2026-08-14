#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=catalog_sweep
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/catalog_sweep_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/catalog_sweep_%j.err

#==============================================================================
# Contract A catalog sweep — bids2table over the full mmmdata tree
#==============================================================================
# Indexes the raw dataset and every derivative dataset, writes
# $INVENTORY_DIR/catalog.duckdb (+ catalog_parquet/, catalog_sweep_report.json),
# and measures per-dataset row counts and wall time.
#
# Usage:  sbatch catalog_sweep.sbatch
#
# derivatives/bids_validation is excluded: it is a byte-level duplicate of the
# raw tree (see mmmdata-agents docs/CLUSTER-TODO.md §5); indexing it would
# double the file table for no query anyone wants.
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

CATALOG_PY=/gpfs/projects/hulacon/shared/envs/catalog/bin/python

"${CATALOG_PY}" "${REPO_ROOT}/scripts/catalog_sweep.py" \
    --root "${BIDS_DIR}" \
    --out-dir "${INVENTORY_DIR}" \
    --exclude derivatives/bids_validation

echo "Finished $(date -Is)"

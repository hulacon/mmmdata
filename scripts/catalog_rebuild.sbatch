#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=catalog_rebuild
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/catalog_rebuild_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/catalog_rebuild_%j.err

#==============================================================================
# Contract A catalog rebuild — thin wrapper over the duckbrain engine
#==============================================================================
# Rebuilds $BIDS_DIR/inventory/catalog.duckdb: bids2table sweep (files +
# datasets tables), expectations tier (resolution view), QC-decision ingest,
# and the supplemental tier — all four, idempotently.
#
# The engine lives in duckbrain (duckbrain.catalog, migrated 2026-08-17 per
# constellation-contracts §3.2): this dataset keeps only the declaration
# ($BIDS_DIR/expectations/dataset.toml), whose [catalog] section carries the
# indexing scope (bids_validation exclude, raw-tree walk globs). Nothing
# dataset-specific belongs in this file beyond paths from load_config.sh.
#
# Usage:  sbatch catalog_rebuild.sbatch
# (a full rebuild measures ~40 s on the login node; the job form exists for
# unattended/scheduled runs)
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

# The duckbrain shared env carries the engine + the [catalog] extra
# (bids2table/duckdb/pyarrow); the former standalone catalog env is retired.
DUCKBRAIN_PY=/gpfs/projects/hulacon/shared/envs/duckbrain/bin/python

"${DUCKBRAIN_PY}" -m duckbrain.catalog rebuild --root "${BIDS_DIR}"

echo "Finished $(date -Is)"

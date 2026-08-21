#!/bin/bash
#==============================================================================
# rebuild_catalog.sh - Nightly rebuild of the Contract A data catalog
#==============================================================================
# Runs `duckbrain.catalog rebuild`, which sweeps every discovered dataset under
# the BIDS root and replaces the catalog at <root>/inventory/catalog.duckdb.
# Each tier is idempotent and replaces only its own tables, so re-running is
# always safe; a failed run leaves the previous catalog in place.
#
# Why nightly: the catalog is the designated answer to "what data exists"
# (docs/INDEX.md), and a stale catalog answers confidently and wrongly. Before
# this was scheduled it had gone four days without a rebuild. Cost is ~35 s of
# CPU and ~290 MB, so the cadence is limited by nothing.
#
# Known gap as of 2026-08-21: derivatives keyed by stimulus_id rather than BIDS
# entities (stimuli_features, srm_stimulus_space) index as 0 rows. That is a
# Contract A tier question, not a fault in this script -- see
# mmmdata-agents/docs/workbench/stimfeat-campaign/log.md. A green run here does
# NOT mean the catalog knows about everything on disk.
#
# Concurrency: a rebuild replaces tables, so a reader holding the DB open
# across the swap can error. At 03:00 that is near-zero risk. If agents start
# querying routinely, build to a temp DB and swap instead.
#
# Scheduled nightly via scrontab (see work_reaper.scrontab, which carries both
# entries; install with `scrontab work_reaper.scrontab`). Manual invocation:
#   rebuild_catalog.sh [bids-root]
#==============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "${SCRIPT_DIR}")}"

# The catalog engine lives in duckbrain's env, whose python3 is also new enough
# for load_config.sh's tomllib (the system python3 is 3.6). Put it on PATH
# rather than activating the conda env: activation needs `module load
# miniconda3/20260319`, which a cron shell does not have.
DUCKBRAIN_ENV="${DUCKBRAIN_ENV:-/gpfs/projects/hulacon/shared/envs/duckbrain}"
if [ ! -x "${DUCKBRAIN_ENV}/bin/python3" ]; then
    echo "ERROR: no python3 at ${DUCKBRAIN_ENV}/bin. Set DUCKBRAIN_ENV to the" >&2
    echo "       duckbrain conda prefix (envs live in shared/envs/, plural)." >&2
    exit 1
fi
export PATH="${DUCKBRAIN_ENV}/bin:${PATH}"

source "${REPO_ROOT}/scripts/load_config.sh"

ROOT="${1:-${BIDS_DIR}}"

case "${ROOT}" in
    /gpfs/projects/hulacon/*) ;;
    *)
        echo "ERROR: refusing to rebuild outside /gpfs/projects/hulacon: ${ROOT}" >&2
        exit 1
        ;;
esac
if [ ! -d "${ROOT}" ]; then
    echo "ERROR: BIDS root ${ROOT} does not exist." >&2
    exit 1
fi

# The log is opened in append mode, so each run delimits itself.
echo "=============================================================="
echo "catalog rebuild  start=$(date -Is)  root=${ROOT}"
echo "  python3: $(command -v python3)"
echo "=============================================================="

START=${SECONDS}
python3 -m duckbrain.catalog rebuild --root "${ROOT}"
STATUS=$?
ELAPSED=$((SECONDS - START))

if [ ${STATUS} -eq 0 ]; then
    echo "catalog rebuild  OK  ${ELAPSED}s  end=$(date -Is)"
else
    # Exit non-zero so Slurm marks the job FAILED rather than burying it here.
    echo "catalog rebuild  FAILED (exit ${STATUS})  ${ELAPSED}s  end=$(date -Is)" >&2
fi
exit ${STATUS}

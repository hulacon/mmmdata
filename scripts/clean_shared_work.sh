#!/bin/bash
#==============================================================================
# clean_shared_work.sh - Nightly reaper for the shared pipeline work root
#==============================================================================
# Deletes per-run work directories under [paths] shared_work_root in which
# nothing has been modified for AGE_DAYS days. Run dirs live at depth 4:
#   <root>/<user>/<tree>/<tool>/<run-dir>
#   e.g.  work/bhutch/mmmdata_work/fmriprep/sub-03_ses-18
# A running pipeline touches its work tree constantly, so an idle tree is a
# leftover from a failed or abandoned run (successful runs already clean up
# after themselves in the sbatch wrappers). Anything a user parks under the
# work root is subject to the same rule — it is scratch space by contract.
#
# Scheduled nightly via scrontab (see work_reaper.scrontab; install with
# `scrontab work_reaper.scrontab`). Manual invocation:
#   clean_shared_work.sh [root] [age-days]
#==============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "${SCRIPT_DIR}")}"

# venv first: load_config.sh needs a python3 with tomllib (system is 3.6)
source "${REPO_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/load_config.sh"

ROOT="${1:-${SHARED_WORK_ROOT}}"
AGE_DAYS="${2:-7}"

case "${ROOT}" in
    /gpfs/projects/hulacon/*) ;;
    *)
        echo "ERROR: refusing to reap outside /gpfs/projects/hulacon: ${ROOT}" >&2
        exit 1
        ;;
esac
if [ ! -d "${ROOT}" ]; then
    echo "Work root ${ROOT} does not exist; nothing to do."
    exit 0
fi

echo "=== $(date '+%F %T') reaping ${ROOT} (idle > ${AGE_DAYS} days) ==="

# Reap run dirs (depth 4) whose entire tree is idle. The inner find exits at
# the first entry (file or dir) modified within the window, so live runs are
# cheap to skip and never touched.
find "${ROOT}" -mindepth 4 -maxdepth 4 -type d -print0 |
while IFS= read -r -d '' dir; do
    if [ -z "$(find "${dir}" -newermt "-${AGE_DAYS} days" -print -quit)" ]; then
        size=$(du -sh "${dir}" 2>/dev/null | cut -f1)
        echo "reap: ${dir} (${size:-?})"
        rm -rf "${dir}" || echo "  (partial: some entries not removable)"
    fi
done

# Stray idle files above the run-dir level, then prune idle empty dirs
# (never the per-user level itself)
find "${ROOT}" -mindepth 2 -maxdepth 3 -type f ! -newermt "-${AGE_DAYS} days" -print -delete
find "${ROOT}" -mindepth 2 -maxdepth 3 -depth -type d -empty ! -newermt "-${AGE_DAYS} days" -print -delete

echo "=== $(date '+%F %T') done ==="

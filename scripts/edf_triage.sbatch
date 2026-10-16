#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=edf_triage
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/edf_triage_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/edf_triage_%j.err

# EDF triage: scan-window quality audit for every EDF under the source tree
# (flat or in per-recording directories). Writes
# src/python/raw2bids_converters/edf_triage.csv (generated data, untracked),
# which generate_inventory.py requires: an EDF missing from it is refused.
# ~10 min for ~100 files, <200 MB RAM. Extra arguments pass through
# (e.g. --output <path> to write somewhere else for a comparison).
#
# The EDF reader (eyelinkio) is in the shared stimfeat env, not mmmdata's own
# .venv, which is why this driver names that interpreter.

set -uo pipefail

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
ENV=/gpfs/projects/hulacon/shared/envs/stimfeat
source ${ENV}/etc/conda/activate.d/stimfeat-env.sh  # PYTHONNOUSERSITE=1 + shared model caches
PYTHON=${ENV}/bin/python3

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo ""

"$PYTHON" "${REPO_ROOT}/scripts/edf_triage.py" "$@"
RC=$?

echo ""
echo "edf_triage.py exit code: $RC"
echo "Job finished: $(date)"
exit $RC

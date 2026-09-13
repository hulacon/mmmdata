#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=stimfeat_py
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_py_%x_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_py_%x_%j.err

#==============================================================================
# Run an arbitrary Python script under the shared `stimfeat` env on a compute
# node. For the one-off scans that read the Contract B / fit-corpus parquet
# tables: too heavy for a login node, too specific to earn a named driver.
#==============================================================================
# First argument is the script; everything after it passes through.
#
#   sbatch --job-name=a_coverage stimfeat_python.sbatch \
#       <repo>/docs/workbench/<slug>/out/scan.py --corpora nsd -o <dir>
#
# Override the defaults per submission, e.g. --mem=160G --time=8:00:00.
# The interpreter is named absolutely: a `module load` anywhere upstream
# replaces a bare `python` with the module's own.
# No email flags by standing preference.
#==============================================================================

set -uo pipefail
echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

if [ $# -lt 1 ]; then
    echo "error: no script given. Usage: sbatch stimfeat_python.sbatch <script.py> [args...]" >&2
    exit 2
fi
SCRIPT="$1"; shift
if [ ! -f "${SCRIPT}" ]; then
    echo "error: script ${SCRIPT} does not exist (paths are resolved on the compute node, not the submit host)" >&2
    exit 2
fi
echo "Script: ${SCRIPT}"
echo "Args: $*"

STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "=== versions ==="
"${STIMFEAT}/bin/python" -c 'import psytwill, pyarrow; print("psytwill", psytwill.__version__, "| pyarrow", pyarrow.__version__)'

echo "=== run ==="
rc=0
"${STIMFEAT}/bin/python" "${SCRIPT}" "$@" || rc=$?

echo "Finished $(date -Is) rc=${rc}"
exit ${rc}

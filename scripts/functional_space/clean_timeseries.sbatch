#!/bin/bash
#SBATCH --job-name=fs_clean
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_clean_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_clean_%A_%a.err
# Sizing (pilot 47483379, /usr/bin/time -v): a 628-volume NATencoding run
# 67 s wall / 5.7 G peak (the MNI 4D read dominates). The longest run in scope
# is 841 volumes; 16 G / 30 min is headroom. ~225 MB on disk per 628 volumes.

#==============================================================================
# functional-space cleaned grayordinate time series — one run per array task
#==============================================================================
# Driver: scripts/functional_space/clean_timeseries.py.
#
# 1. Once per subject / tree (seconds; login node is fine):
#      python scripts/functional_space/clean_timeseries.py geometry
# 2. Dry-run manifest + units file:
#      python scripts/functional_space/clean_timeseries.py plan \
#          --units units.txt --manifest dryrun.tsv
# 3. Submit one array task per line:
#      sbatch --array=1-$(wc -l < units.txt)%30 --export=ALL,UNITS=units.txt clean_timeseries.sbatch
# 4. After the array: python scripts/functional_space/clean_timeseries.py collect
#
# wb_command comes from the shared functional-space env by absolute path (see
# grayordinates.wb_command_path); nothing is activated but the .venv. Do not
# `module load` anything after the venv is activated: it replaces `python`.
#==============================================================================

set -euo pipefail

: "${UNITS:?set UNITS (path to the units file from \`clean_timeseries.py plan --units\`) via --export}"
EXTRA="${EXTRA:-}"

echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-<none>} on ${SLURMD_NODENAME}; units ${UNITS}; start $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

# shellcheck disable=SC2086
python "${REPO_ROOT}/scripts/functional_space/clean_timeseries.py" run \
    --units "${UNITS}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA}

echo "End $(date)"

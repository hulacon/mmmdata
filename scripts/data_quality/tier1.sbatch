#!/bin/bash
#SBATCH --job-name=dq_tier1
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/dq_tier1_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/dq_tier1_%A_%a.err
# Sizing (pilot 47405632, /usr/bin/time -v, 3 regimes per run): a 628-volume
# NATencoding run 38 s / 5.4 G peak; the 1,858-volume NATretrieval run (the
# longest in the dataset) 94 s / 15.8 G. The peak is one full float32 read of
# the 4D plus the masked copy; the solver itself is chunked. 24 G / 20 min is
# headroom for every run. Revise from seff after the first array.

#==============================================================================
# data-quality tier 1 — one run per array task, every requested regime
#==============================================================================
# Design record: mmmdata-agents/docs/workbench/data-quality/. Driver:
# scripts/data_quality/tier1.py.
#
# 1. Write the units file (one run per line) from the plan verb:
#      python scripts/data_quality/tier1.py plan --units units.txt
# 2. Submit one array task per line:
#      sbatch --array=1-$(wc -l < units.txt)%40 --export=ALL,UNITS=units.txt tier1.sbatch
# 3. Optional: EXTRA="--include-provisional" or EXTRA="--regimes none,drift"
#
# Do not `module load` anything after the venv is activated: it replaces
# `python` with the module's interpreter.
#==============================================================================

set -euo pipefail

: "${UNITS:?set UNITS (path to the units file from `tier1.py plan --units`) via --export}"
EXTRA="${EXTRA:-}"

echo "=========================================="
echo "data-quality tier 1 — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}  Array task: ${SLURM_ARRAY_TASK_ID:-<none>}"
echo "Node: ${SLURMD_NODENAME}"
echo "Units: ${UNITS}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

# shellcheck disable=SC2086
python "${REPO_ROOT}/scripts/data_quality/tier1.py" run \
    --units "${UNITS}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA}

echo ""
echo "End Time: $(date)"

#!/bin/bash
#SBATCH --job-name=dq_glmsingle
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/dq_glmsingle_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/dq_glmsingle_%A_%a.err
# Sizing: each beta type is a pickled dict loaded whole (enc TYPEB ~15 GB, ret
# arms 5-8 GB), one at a time, plus the masked betas and their z-scores
# (~2.4 GB each for the 2,562-trial enc arm), then 19 shuffle nulls per set.
# Measured (47532037, 9 fits): enc 6-7 min / 20.7 G MaxRSS, ret arms 1-5 min /
# 11.0 G. 64 G / 2 h is ~3x headroom.

#==============================================================================
# data-quality tier 1, GLMsingle-native cells (T2.11/T2.12): one fit per task
#==============================================================================
# Design record: mmmdata-agents/docs/workbench/data-quality/. Driver:
# scripts/data_quality/tier1.py (verbs glmsingle-plan, glmsingle).
#
# 1. Units (one `sub<TAB>arm` line per missing/stale cell):
#      python scripts/data_quality/tier1.py glmsingle-plan --units gs_units.txt
# 2. Submit:
#      sbatch --array=1-$(wc -l < gs_units.txt) --export=ALL,UNITS=gs_units.txt tier1_glmsingle.sbatch
# 3. Then `tier1.py collect` and `tier2.py build --parts glmsingle`.
#
# Do not `module load` anything after the venv is activated: it replaces
# `python` with the module's interpreter.
#==============================================================================

set -euo pipefail

: "${UNITS:?set UNITS (path to the units file from `tier1.py glmsingle-plan --units`) via --export}"
EXTRA="${EXTRA:-}"

echo "Job ID: ${SLURM_JOB_ID}  Array task: ${SLURM_ARRAY_TASK_ID:-<none>}  Node: ${SLURMD_NODENAME}"
echo "Units: ${UNITS}  Start: $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

# shellcheck disable=SC2086
/usr/bin/time -v python "${REPO_ROOT}/scripts/data_quality/tier1.py" glmsingle \
    --units "${UNITS}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA}

echo "End: $(date)"

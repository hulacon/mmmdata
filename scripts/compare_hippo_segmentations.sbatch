#!/bin/bash
#SBATCH --job-name=hippcmp
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hippcmp_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hippcmp_%A_%a.err

#==============================================================================
# Hippocampal segmentation comparison (scripts/compare_hippo_segmentations.py).
#
#   One subject:   sbatch --export=ALL,SUB=## compare_hippo_segmentations.sbatch
#   Array:         sbatch --array=0-5 --export=ALL,SUBS="03 04 05 06 07 09" compare_hippo_segmentations.sbatch
#   Group tables:  sbatch --export=ALL,GROUP=1 compare_hippo_segmentations.sbatch
# Optional OUT=<dir> overrides the default output tree.
#
# Modules load BEFORE the venv: a later `module load` replaces `python`.
#==============================================================================
set -euo pipefail

if [ -n "${SUBS:-}" ]; then
    read -r -a _subs <<< "${SUBS}"
    SUB="${_subs[${SLURM_ARRAY_TASK_ID:?SUBS needs --array}]}"
fi
OUT_ARGS=(); [ -n "${OUT:-}" ] && OUT_ARGS=(--out "${OUT}")
if [ -n "${GROUP:-}" ]; then WHO=(--group); else WHO=(--sub "${SUB:?set SUB, SUBS+--array, or GROUP=1}"); fi

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: ${WHO[*]}  ($(date))"

module load ants/2.5.2
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"
export TMPDIR="${TMPDIR:-/tmp}"
git -C "${REPO_ROOT}" log -1 --format='mmmdata %h'
git -C "${REPO_ROOT}" status --short src/python/neuroimaging/hippo_compare.py scripts/compare_hippo_segmentations.py \
    | sed 's/^/  dirty: /'

python "${REPO_ROOT}/scripts/compare_hippo_segmentations.py" "${WHO[@]}" "${OUT_ARGS[@]}" --jobs "${SLURM_CPUS_PER_TASK}"
echo "Done $(date)"

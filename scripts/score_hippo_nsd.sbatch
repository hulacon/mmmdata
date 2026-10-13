#!/bin/bash
#SBATCH --job-name=hippnsd
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hippnsd_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hippnsd_%A_%a.err

#==============================================================================
# Hippocampal segmentations vs NSD manual labels (scripts/score_hippo_nsd.py).
#
#   One subject:   sbatch --export=ALL,SUB=##,PPDATA=<nsddata/ppdata>,EVAL_ROOT=<dir> score_hippo_nsd.sbatch
#   Array:         sbatch --array=0-7 --export=ALL,SUBS="01 02 ... 08",PPDATA=...,EVAL_ROOT=... score_hippo_nsd.sbatch
#   Group tables:  sbatch --export=ALL,GROUP=1,EVAL_ROOT=<dir> score_hippo_nsd.sbatch
#
# Modules load BEFORE the venv: a later `module load` replaces `python`.
#==============================================================================
set -euo pipefail

if [ -n "${SUBS:-}" ]; then
    read -r -a _subs <<< "${SUBS}"
    SUB="${_subs[${SLURM_ARRAY_TASK_ID:?SUBS needs --array}]}"
fi
if [ -n "${GROUP:-}" ]; then
    WHO=(--group)
else
    WHO=(--sub "${SUB:?set SUB, SUBS+--array, or GROUP=1}" --ppdata "${PPDATA:?set PPDATA}")
fi

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: ${WHO[*]}  ($(date))"

module load ants/2.5.2
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"
export TMPDIR="${TMPDIR:-/tmp}"
git -C "${REPO_ROOT}" log -1 --format='mmmdata %h'
git -C "${REPO_ROOT}" status --short src/python/neuroimaging/hippo_nsd.py src/python/neuroimaging/hippo_compare.py \
    scripts/score_hippo_nsd.py | sed 's/^/  dirty: /'

python "${REPO_ROOT}/scripts/score_hippo_nsd.py" "${WHO[@]}" --eval-root "${EVAL_ROOT:?set EVAL_ROOT}" \
    --jobs "${SLURM_CPUS_PER_TASK}"
echo "Done $(date)"

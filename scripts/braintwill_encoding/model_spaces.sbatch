#!/bin/bash
#SBATCH --job-name=enc_models
#SBATCH --account=hulacon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/enc_models_%A_%a.out
# Time and memory: set from one task per register (the widest model) before the array;
# override with sbatch --time/--mem on the command line.

#==============================================================================
# encoding harness: every model space × subject on one register, as an array
#==============================================================================
# Task i scores model space MODEL_SPACES[i // n_subjects] for SUBJECTS[i % n_subjects]
# (both read from harness.py, sorted), so the array size is n_models × n_subjects:
#
#   sbatch --array=0-80%10 --export=ALL,REGISTER=films model_spaces.sbatch
#   sbatch --array=0-80%10 --export=ALL,REGISTER=tb-enc-ts model_spaces.sbatch
#
# Cached scores are read back, never refit, so a resubmitted array only fits what is missing.
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
DIR="${REPO_ROOT}/scripts/braintwill_encoding"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
: "${REGISTER:?set REGISTER (films|tb-enc-ts)}" "${SLURM_ARRAY_TASK_ID:?submit as an array}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

read -r SPACE SUB < <(cd "${DIR}" && "${PY}" - "${SLURM_ARRAY_TASK_ID}" <<'EOF'
import sys
import harness as h
spaces, subs = sorted(h.MODEL_SPACES), list(h.fm.SUBJECTS)
i = int(sys.argv[1])
if i >= len(spaces) * len(subs):
    sys.exit(f"task {i} is past the {len(spaces)} x {len(subs)} grid")
print(spaces[i // len(subs)], subs[i % len(subs)])
EOF
)

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${REGISTER} ${SPACE} sub-${SUB} on ${SLURMD_NODENAME}; start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v "${PY}" "${DIR}/harness.py" score --register "${REGISTER}" --space "${SPACE}" --sub "${SUB}" \
    --backend torch_cuda
echo "End $(date)"

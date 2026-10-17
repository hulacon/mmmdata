#!/bin/bash
#SBATCH --job-name=enc_rescue
#SBATCH --account=hulacon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/enc_rescue_%A_%a.out
# Time and memory: set from one task (the widest member) before the array; override with
# sbatch --time/--mem. One-band film model fits took 1.5-5 min at <= 36 GB (read (b)).

#==============================================================================
# encoding harness: rescue spaces x subjects on one register, as an array
#==============================================================================
# Task i scores SPACES[i // n_subjects] for SUBJECTS[i % n_subjects] (subjects from
# harness.py), so the array size is n_spaces x n_subjects:
#
#   sbatch --array=0-14 --export=ALL,REGISTER=films,SPACES=rescue-V:rescue-V+m1:... rescue.sbatch
#
# Cached scores are read back, never refit, so a resubmitted array only fits what is missing.
# Then: rescue.py read (rescue_cpu.sbatch).
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
DIR="${REPO_ROOT}/scripts/braintwill_encoding"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
: "${REGISTER:?set REGISTER (films|tb-enc)}" "${SPACES:?set SPACES (colon-separated: sbatch --export splits on commas)}" \
  "${SLURM_ARRAY_TASK_ID:?submit as an array}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

read -r SPACE SUB < <(cd "${DIR}" && "${PY}" - "${SLURM_ARRAY_TASK_ID}" "${SPACES}" <<'PYEOF'
import sys
import harness as h
spaces, subs = sys.argv[2].split(":"), list(h.fm.SUBJECTS)
for s in spaces:
    h.parse_rescue(s)
i = int(sys.argv[1])
if i >= len(spaces) * len(subs):
    sys.exit(f"task {i} is past the {len(spaces)} x {len(subs)} grid")
print(spaces[i // len(subs)], subs[i % len(subs)])
PYEOF
)

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${REGISTER} ${SPACE} sub-${SUB} on ${SLURMD_NODENAME}; start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v "${PY}" "${DIR}/harness.py" score --register "${REGISTER}" --space "${SPACE}" --sub "${SUB}" \
    --backend torch_cuda
echo "End $(date)"

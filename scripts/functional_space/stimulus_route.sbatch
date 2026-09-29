#!/bin/bash
#SBATCH --job-name=fs_stim
#SBATCH --account=hulacon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_stim_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_stim_%j.err
# The full A100 is requested by model, so the job never lands on a 10 GB MIG slice.
# Sizing: unmeasured until the first dry-run fit; 64 G / 1 h is a first guess
# (3 encoders on the GPU; host holds 3 subjects' film series + one probe chunk
# of predictions + the per-piece Grams).

#==============================================================================
# functional-space stimulus route — one partition job
#==============================================================================
# Driver: scripts/functional_space/stimulus_route.py (fits encoders and
# transforms on alignment data; computes no score).
#
#   python scripts/functional_space/stimulus_route.py plan --space ebind --pct 0 --draw 0 --target 03
#   sbatch --export=ALL,SPACE=ebind,PCT=0,DRAW=0,TARGET=03 stimulus_route.sbatch
#   (optional: SCENARIO=secondary, EXTRA="--save-grams")
#
# Outputs: <derivatives>/functional_space/routes/stimulus-<space>/<scenario>/
# pct-<pct>/draw-<draw>/target-<sub>/ (cross_sub-*.npz + stimulus.json).
#==============================================================================

set -euo pipefail

: "${SPACE:?set SPACE (ebind|vgg19)}" "${PCT:?set PCT}" "${DRAW:?set DRAW}" "${TARGET:?set TARGET}"
SCENARIO="${SCENARIO:-primary}"
REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID} ${SPACE} ${SCENARIO} ${PCT}% draw ${DRAW} target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/stimulus_route.py" fit \
    --space "${SPACE}" --scenario "${SCENARIO}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}" \
    --backend torch_cuda ${EXTRA:-}
echo "End $(date)"

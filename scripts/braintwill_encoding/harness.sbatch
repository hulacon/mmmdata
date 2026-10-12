#!/bin/bash
#SBATCH --job-name=enc_harness
#SBATCH --account=hulacon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/enc_harness_%j.out
# The full A100 is requested by model, so the job never lands on a 10 GB MIG slice.
# Sizing: unmeasured. functional-space's 15-film encoders took 3 s (one band) to ~80 s
# (three bands) on this GPU; a nested score on the 46-film pool is 6 fits on ~3x the rows
# (kernel solves scale ~n^3), and the host holds ~7k x 89,820 float32 responses (~2.6 GB)
# plus the design. 96 G / 2 h is a first guess; read /usr/bin/time and resize.

#==============================================================================
# encoding harness (braintwill.encoding) — one verb, one subject or one job
#==============================================================================
# Driver: scripts/braintwill_encoding/harness.py.
#
#   sbatch --export=ALL,VERB=score,ARGS="--space ebind --sub ##" harness.sbatch
#   sbatch --export=ALL,VERB=equivalence,ARGS="--space ebind --pct 0 --draw 0 --target ##" harness.sbatch
#
# Outputs: <derivatives>/encoding_harness/ (scores under register-*/, equivalence/*.json).
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
: "${VERB:?set VERB (plan|score|equivalence)}" "${ARGS:?set ARGS}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

echo "Job ${SLURM_JOB_ID} ${VERB} ${ARGS} on ${SLURMD_NODENAME}; start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# shellcheck disable=SC2086  # ARGS is a list of flags
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/braintwill_encoding/harness.py" "${VERB}" ${ARGS} \
    --backend torch_cuda
echo "End $(date)"

#!/bin/bash
#SBATCH --job-name=fs_lag
#SBATCH --account=hulacon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_lag_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_lag_%j.err
# A 40 GB MIG slice suffices: no probe predictions are held, only the
# ~2.3k x 83k training target. Sizing: 8 refits x 3 subjects; VGG19 ~35 s and EBind ~3 s per fit on an A100
# (encoder sizing fits), so 1 h covers either space with room to spare.

#==============================================================================
# functional-space encoder timing check (lag sweep) — one feature space
#==============================================================================
# Driver: scripts/functional_space/encoding.py lagcheck (alignment films only;
# computes no score). One record per subject:
#   <derivatives>/functional_space/dryrun/lagcheck_<space>_sub-<sub>.json
#
#   sbatch --export=ALL,SPACE=ebind lagcheck.sbatch
#   (optional: PCT=0 DRAW=0 TARGET=03 SUBS="03 04 05")
#==============================================================================

set -euo pipefail

: "${SPACE:?set SPACE (ebind|vgg19|psytwill)}"
PCT="${PCT:-0}" DRAW="${DRAW:-0}" TARGET="${TARGET:-03}" SUBS="${SUBS:-03 04 05}"
REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID} ${SPACE} ${PCT}% draw ${DRAW} target sub-${TARGET} subs ${SUBS} on ${SLURMD_NODENAME}; start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for SUB in ${SUBS}; do
    echo "== sub-${SUB}"
    "${PY}" "${REPO_ROOT}/scripts/functional_space/encoding.py" lagcheck \
        --space "${SPACE}" --sub "${SUB}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}" --backend torch_cuda
done
echo "End $(date)"

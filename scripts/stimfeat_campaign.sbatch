#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=stimfeat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_%x_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_%x_%j.err

#==============================================================================
# Contract B §4.2 extraction campaign — one slice of the matrix per job
#==============================================================================
# Thin wrapper around scripts/stimfeat_campaign.py: every argument is passed
# straight through to its `run` verb, so a job is described by its filters.
#
#   sbatch stimfeat_campaign.sbatch --set movies --source frames
#   sbatch stimfeat_campaign.sbatch --set shared1000 --source image
#   sbatch -J stimfeat_annot stimfeat_campaign.sbatch --set movies --source annot
#
# The driver is idempotent per cell (the §4.1 sidecar is the done-marker), so
# a job that hits the wall clock is resumed by resubmitting it unchanged. It
# also refuses to start unless the prefix pre-flight is CLEAN, and §4.1-checks
# every family it writes before counting the cell as done.
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"
echo "Filters: $*"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

# stimfeat env without conda: the interpreter directly, plus the activate.d
# hooks (shared HF/torch caches, PYTHONNOUSERSITE) which key off CONDA_PREFIX.
STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
# HF credentials: the activate.d hooks point HF_HOME at the PIRG-shared cache,
# whose default token file would be group-readable. Resolve the token from the
# user's private home file instead; the value never enters the environment.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
# torchcodec loads its FFmpeg backend from the env's own conda-bundled libs
export LD_LIBRARY_PATH="${STIMFEAT}/lib:${LD_LIBRARY_PATH:-}"
PY="${STIMFEAT}/bin/python"

DRIVER="${REPO_ROOT}/scripts/stimfeat_campaign.py"

echo "=== derived inputs ==="
"${PY}" "${DRIVER}" inputs

# `plan` takes only the filter flags, so a run-only flag (--fail-fast,
# --no-verify) must not abort the job before anything is extracted.
echo "=== manifest for this slice ==="
"${PY}" "${DRIVER}" plan "$@" || echo "(manifest skipped: run-only flags in: $*)"

# `run` exits 1 on any failed cell, which is information, not a reason to
# lose the summary line -- so capture it rather than letting set -e take it.
echo "=== run ==="
rc=0
"${PY}" "${DRIVER}" run "$@" || rc=$?

echo "Finished $(date -Is) rc=${rc}"
exit ${rc}

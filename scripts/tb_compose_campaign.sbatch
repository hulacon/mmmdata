#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=tb_compose
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_compose_%x_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_compose_%x_%j.err

#==============================================================================
# TB compose campaign -- one job per filter (contracts §4.3 item 5)
#==============================================================================
# Thin wrapper around scripts/tb_compose_campaign.py `run`: every argument is
# passed through, so a job is described by its filters.
#
#   sbatch -J tb_compose_03 tb_compose_campaign.sbatch --subject 03
#   sbatch tb_compose_campaign.sbatch                      # all subjects
#   sbatch tb_compose_campaign.sbatch --no-media --subject 06
#
# ~20 s per run on a login node (tables 11 s + half-scale media); a full
# subject (98 runs) fits well inside the header. The driver is idempotent
# (the verb's input signature + a populated frames/ dir), so a job that hits
# the wall clock is resumed by resubmitting it unchanged; any run with
# unmapped media or a grid end != scan end fails the job.
#==============================================================================

set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"
echo "Filters: $*"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

# ffmpeg for the media leg. Modules load BEFORE the env is set up and the
# interpreter is named absolutely below, so the module's own python cannot
# shadow it.
module load ffmpeg/6.0.0
command -v ffmpeg

STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
PY="${STIMFEAT}/bin/python"

"${PY}" "${REPO_ROOT}/scripts/tb_compose_campaign.py" run "$@"
echo "finished $(date -Is)"

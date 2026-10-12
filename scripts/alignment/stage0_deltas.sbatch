#!/bin/bash
#SBATCH --job-name=align_stage0
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/align_stage0_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/align_stage0_%j.err

#==============================================================================
# One subject: candidate BOLD->anatomy alignments (arms B and C) and how far each
# would move every run, transforms only (scripts/alignment/stage0_deltas.py).
#
#   sbatch --export=ALL,SUB=##,REF=##,OUT=<dir>[,ARMS=B,C][,CHECK=<entity_prefix>] stage0_deltas.sbatch
# (ARMS takes a comma, so pass it as ARMS=B or ARMS=C; the default is both.)
#
# Modules load BEFORE the venv: a later `module load` replaces `python`.
#==============================================================================
set -euo pipefail

: "${SUB:?set SUB (bare subject label) via --export}"
: "${REF:?set REF (arm C reference session, bare label) via --export}"
: "${OUT:?set OUT (output root) via --export}"
CHECK_ARGS=(); [ -n "${CHECK:-}" ] && CHECK_ARGS=(--check-run "${CHECK}")

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: sub-${SUB} ref ses-${REF} -> ${OUT}  ($(date))"

module load ants/2.5.2 freesurfer/8.2.0
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"
# freesurfer/8.2.0 is an apptainer wrapper that binds no /gpfs path by default: inputs, outputs and the
# licence would all be invisible to bbregister.
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}/gpfs"
# Unconditional: the module sets FS_LICENSE to a licence file that does not exist.
export FS_LICENSE="${REPO_ROOT}/../fs_license.txt"
export TMPDIR="${TMPDIR:-/tmp}"
git -C "${REPO_ROOT}" log -1 --format='mmmdata %h'
git -C "${REPO_ROOT}" status --short src/python/neuroimaging/alignment_candidates.py scripts/alignment | sed 's/^/  dirty: /'

python "${REPO_ROOT}/scripts/alignment/stage0_deltas.py" --sub "${SUB}" --reference-session "${REF}" \
    --out "${OUT}" --arms "${ARMS:-B,C}" --jobs "${SLURM_CPUS_PER_TASK}" "${CHECK_ARGS[@]}"
echo "Done $(date)"

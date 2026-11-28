#!/bin/bash
#SBATCH --job-name=gnc_twins
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/gnc_twins_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/gnc_twins_%A_%a.err

#==============================================================================
# One session: GNC field from its ND/DIS3D anatomical twins
# (scripts/gnc/twin_field.py): convert + pair, register every pair, then the
# within-session repeatability floor from its two T1w pairs.
#
#   sbatch --export=ALL,SUB=##,SES=##,OUT=<scratch dir>[,REPO=<mmmdata checkout>] twin_field.sbatch
#
# Array form: one task per placement, PLACEMENTS a space-separated list of
# sub:ses pairs, each written to OUT_ROOT/sub-##_ses-##:
#   sbatch --array=0-<n-1> --export=ALL,PLACEMENTS="## : ## ...",OUT_ROOT=<dir> twin_field.sbatch
#
# REPO should be a frozen copy of the code when other work edits the checkout
# while the job runs (a later `python` call would read the edited script).
#
# Modules load BEFORE the venv: a later `module load` replaces `python`.
#==============================================================================
set -euo pipefail

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${PLACEMENTS:-}" ]; then
    read -r -a _P <<< "${PLACEMENTS}"
    _ONE="${_P[${SLURM_ARRAY_TASK_ID}]:?no placement for task ${SLURM_ARRAY_TASK_ID}}"
    SUB="${_ONE%%:*}"; SES="${_ONE##*:}"
    OUT="${OUT_ROOT:?set OUT_ROOT for the array form}/sub-${SUB}_ses-${SES}"
fi
: "${SUB:?set SUB (bare subject label) via --export}"
: "${SES:?set SES (bare session label) via --export}"
: "${OUT:?set OUT (scratch output dir) via --export}"
REPO="${REPO:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: sub-${SUB} ses-${SES} -> ${OUT}  ($(date))"

module load ants/2.5.2 freesurfer/8.2.0
export REPO_ROOT="${REPO}"
source "/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts/load_config.sh"
source "/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/.venv/bin/activate"
# freesurfer/8.2.0 is an apptainer wrapper that binds no /gpfs path by default.
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}/gpfs"
# The module wraps only some FreeSurfer commands; run SynthStrip from the image its wrappers exec.
FS_SIF=$(grep -o '[^ ]*freesurfer-[^ ]*\.sif' "$(command -v bbregister)" | head -1)
[ -f "${FS_SIF}" ] || { echo "cannot find the FreeSurfer image behind bbregister" >&2; exit 1; }
export SYNTHSTRIP_CMD="apptainer exec ${FS_SIF} /usr/local/freesurfer/8.2.0-1/bin/mri_synthstrip"
export FS_LICENSE="/gpfs/projects/hulacon/shared/mmmdata/code/fs_license.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if git -C "${REPO}" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "${REPO}" log -1 --format='mmmdata %h %d'
    git -C "${REPO}" status --short scripts/gnc | sed 's/^/  dirty: /'
else
    echo "mmmdata frozen copy at ${REPO} (CODE_VERSION=${CODE_VERSION:-unset})"
fi

mkdir -p "${OUT}"
python "${REPO}/scripts/gnc/twin_field.py" pairs --sub "${SUB}" --ses "${SES}" --out "${OUT}"
python "${REPO}/scripts/gnc/twin_field.py" register --out "${OUT}" --jobs 2 --threads "${SLURM_CPUS_PER_TASK}"
python "${REPO}/scripts/gnc/twin_field.py" floor --out "${OUT}" || echo "floor: skipped (needs two T1w pairs at one placement)"
echo "Done $(date)"

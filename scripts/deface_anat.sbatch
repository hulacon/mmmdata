#!/bin/bash
#SBATCH --job-name=deface_anat
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/deface_anat_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/deface_anat_%A_%a.err

#==============================================================================
# Deface every whole-head anatomical volume of one subject, in place
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/anat-defacing/ (DECIDED 2026-09-11:
# pydeface; derivatives in place; faced originals mirrored byte-identical to
# the sourcedata sibling; reports moved out, viewer bundles rebuilt).
#
# One array task = one subject. The script computes ONE pydeface mask on the
# fMRIPrep T1w and carries it to every other grid (ANTs from the fMRIPrep
# container, FSL flirt for same-session T2w), checks face ∩ brain = ∅ in every
# grid and the direction of every transform leg BEFORE writing, mirrors each
# original to mmmsourcedata, then zeroes the face voxels in the on-disk bytes.
# Nothing outside the face changes; provenance + masks + montages land under
# <source_dir>/anat_defacing/sub-XX/.
#
# Array task mapping:  1: sub-03  2: sub-04  3: sub-05  4: sub-06  5: sub-07
#
# Usage:
#   sbatch --export=ALL,DEFACE_DRY_RUN=1 --array=1 deface_anat.sbatch   # pilot: masks, checks, montages; no writes in BIDS
#   sbatch --array=1 deface_anat.sbatch                                  # sub-03 for real, after montage review
#   sbatch --array=2-5 deface_anat.sbatch                                # the rest
#   sbatch --export=ALL,DEFACE_RAW_ONLY=1,DEFACE_SUBJECT=08 deface_anat.sbatch
#       # converter guard for a subject without fMRIPrep (raw anat only)
#==============================================================================

SUBJECTS=(03 04 05 06 07)
if [ -n "${DEFACE_SUBJECT}" ]; then
    SUBJECT="${DEFACE_SUBJECT}"
else
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    SUBJECT=${SUBJECTS[$IDX]}
fi

echo "=========================================="
echo "anat defacing — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: sub-${SUBJECT}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

# FSL (flirt, convert_xfm) — pydeface drives flirt through nipype
if ! command -v flirt >/dev/null 2>&1; then
    module load fsl/6.0.7.9 2>/dev/null || true
fi
if ! command -v flirt >/dev/null 2>&1; then
    echo "ERROR: flirt not on PATH and 'module load fsl/6.0.7.9' did not provide it" >&2
    exit 2
fi
export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}

FLAGS=""
if [ "${DEFACE_DRY_RUN:-0}" = "1" ]; then
    FLAGS="${FLAGS} --dry-run"
    echo "DRY RUN: masks, checks and montages only; nothing written in the BIDS tree"
fi
if [ "${DEFACE_RAW_ONLY:-0}" = "1" ]; then
    FLAGS="${FLAGS} --raw-only"
    echo "RAW ONLY: converter-guard mode (no fMRIPrep expected)"
fi

python ${REPO_ROOT}/scripts/deface_anat.py run --subject ${SUBJECT} ${FLAGS}
EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
echo "=========================================="

exit ${EXIT_STATUS}

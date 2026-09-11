#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=fitcorpus
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fitcorpus_%x_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fitcorpus_%x_%A_%a.err

#==============================================================================
# psytwill-space fit-corpus extraction — Talapas wrapper (one unit per task)
#==============================================================================
# Site wrapper around psytwill's portable driver `scripts/fitcorpus/extract.py`
# (the *2psy battery over external corpora, Contract B sidecars, ids in the
# ext-<corpus>-<native> namespace). Every argument passes straight through to
# the driver's `run` verb; as an array job each task takes one unit:
#
#   sbatch fitcorpus_extract.sbatch --corpus nsd --unit u000          # pilot
#   sbatch --array=0-72 fitcorpus_extract.sbatch --corpus nsd          # 73 units
#   sbatch --array=0-43 fitcorpus_extract.sbatch --corpus movie10
#   sbatch --array=0-71 fitcorpus_extract.sbatch --corpus friends
#
# Staged audio corpora (stage.py: librispeech, gigaspeech, fma, musopen,
# narratives, twp-unpresented) split each unit across TWO jobs, because the
# psychoacoustic model is CPU-only (MoSQITo) and ~70 % of the audio wall time
# (3.7x real time vs 1.8x for the other 19 models together, measured
# 2026-09-10 on a 12 min unit) — leaving it in the GPU job wastes the slice:
#
#   sbatch --array=0-N%24 --cpus-per-task=4 --mem=48G \
#          fitcorpus_extract.sbatch --corpus librispeech --skip-model psychoacoustic
#   sbatch --array=<indices> --partition=compute --gres=gpu:0 --cpus-per-task=4 --mem=<see below> \
#          --time=05:00:00 fitcorpus_extract.sbatch --corpus librispeech --model psychoacoustic
#
# (`--gres=gpu:0` on the command line overrides the header; CPU-only cells
# on the compute partition dodge the gpu QOS's 96-CPU per-user cap.)
#
# MEMORY for the psychoacoustic job scales with unit LENGTH, not corpus:
# MoSQITo's loudness_zwtv holds ~28 full-length float64 band signals plus
# temporaries, peaking at ~2.85 GB per minute of audio at 48 kHz (measured
# 2026-09-10: 64 G held every unit <= 21.8 min and lost every unit
# >= 22.8 min; sacct MaxRSS undersamples that peak at ~2.2 GB/min, so size
# from the kill threshold, not from MaxRSS). The peak lands ~40 min into a
# 24-min unit, so an undersized task wastes that much before dying. Bucket
# the array by `duration_sec` in `<corpus>/inputs/units.csv`:
#     <= 12 min  44G      <= 18 min  64G
#     <= 24 min  84G      <= 31 min  108G     (>= 20 % margin each)
# Units longer than ~31 min (the 56-min narratives story) go to the memory
# partition; 208G held it. The compute QOS caps one user at mem=4T, so the
# array's own throttle is unnecessary. Runtime is ~3-4.6x real time.
#
# Idempotent per cell (a §4.1 sidecar that passes `psytwill battery --check`
# is the done-marker); a task that hits the wall clock is resumed by
# resubmitting the same array index. No email flags by standing preference.
#==============================================================================

set -uo pipefail
echo "Job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID:-none}) on $(hostname), started $(date -Is)"
echo "Args: $*"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

# stimfeat env without conda: the interpreter directly, plus the activate.d
# hooks (shared HF/torch caches, PYTHONNOUSERSITE) which key off CONDA_PREFIX.
STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export LD_LIBRARY_PATH="${STIMFEAT}/lib:${LD_LIBRARY_PATH:-}"
PY="${STIMFEAT}/bin/python"

# Site roots for the portable driver (it refuses to guess them).
export PSYTWILL_FITCORPUS_DURABLE="${PSYTWILL_FITCORPUS_DURABLE:-/gpfs/projects/hulacon/shared/psytwill/fit-corpora}"
export PSYTWILL_FITCORPUS_SCRATCH="${PSYTWILL_FITCORPUS_SCRATCH:-/gpfs/scratch/hulacon/bhutch/psytwill-fitcorpus}"
export PSYTWILL_NSD_HDF5="${PSYTWILL_NSD_HDF5:-/gpfs/projects/hulacon/shared/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5}"

DRIVER=/gpfs/projects/hulacon/shared/mmmdata/code/psytwill/scripts/fitcorpus/extract.py

UNIT_ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    UNIT_ARGS=(--unit-index "${SLURM_ARRAY_TASK_ID}")
fi

echo "=== versions ==="
"${PY}" -c 'import viz2psy, aud2psy, word2psy, psytwill; print("viz2psy", viz2psy.__version__, "aud2psy", aud2psy.__version__, "word2psy", word2psy.__version__, "psytwill", psytwill.__version__)'

echo "=== plan for this slice ==="
"${PY}" "${DRIVER}" plan "$@" "${UNIT_ARGS[@]}" || echo "(plan skipped: run-only flags in: $*)"

echo "=== run ==="
rc=0
"${PY}" "${DRIVER}" run "$@" "${UNIT_ARGS[@]}" || rc=$?

echo "Finished $(date -Is) rc=${rc}"
exit ${rc}

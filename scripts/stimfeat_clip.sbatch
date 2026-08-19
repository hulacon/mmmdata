#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=stimfeat_clip
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_clip_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_clip_%j.err

#==============================================================================
# CLIP shared-space feature extraction — all three stimulus sets
#==============================================================================
# CLIP ViT-B-32 counterpart to stimfeat_ebind.sbatch, for comparing the two
# joint spaces (Contract B §4.1 sidecars, 4-digit indices). CLIP has no audio
# arm, so movie embeddings are vision-only (frames @0.5 s):
#   1. viz2psy  clip       shared1000 images   -> stimuli_features/shared1000/
#   2. word2psy clip_text  twp1000 words       -> stimuli_features/twp1000/
#   3. viz2psy  clip       movie frames @0.5s  -> stimuli_features/movies/<id>/
# Steps 1-2 skip if their outputs exist; step 3 skips per-clip, so the job is
# safe to re-run after a failure.
#
# Usage:  sbatch stimfeat_clip.sbatch
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
export LD_LIBRARY_PATH="${STIMFEAT}/lib:${LD_LIBRARY_PATH:-}"
PY="${STIMFEAT}/bin/python"

STIM_DIR="${BIDS_DIR}/stimuli"
REGISTRY="${STIM_DIR}/stimulus_registry"
OUT_ROOT="${BIDS_DIR}/derivatives/stimuli_features"

mkdir -p "${OUT_ROOT}/shared1000" "${OUT_ROOT}/twp1000"

if [[ -f "${OUT_ROOT}/shared1000/clip.csv" ]]; then
    echo "=== 1/3 shared1000 clip.csv exists, skipping"
else
    echo "=== 1/3 viz2psy clip: shared1000 images ==="
    "${PY}" -m viz2psy.cli clip "${STIM_DIR}"/shared1000/images/*.png \
        -o "${OUT_ROOT}/shared1000/clip.csv" --batch-size 64 --no-viz --quiet
fi

if [[ -f "${OUT_ROOT}/twp1000/clip_text_chunks.csv" ]]; then
    echo "=== 2/3 twp1000 clip_text exists, skipping"
else
    echo "=== 2/3 word2psy clip_text: twp1000 words ==="
    WORDS_CSV="${OUT_ROOT}/twp1000/.twp1000_words_input.csv"
    "${PY}" -c "
import csv
with open('${REGISTRY}/twp1000.tsv') as f, open('${WORDS_CSV}', 'w', newline='') as out:
    w = csv.writer(out); w.writerow(['stimulus_id', 'word'])
    for row in csv.DictReader(f, delimiter='\t'):
        w.writerow([row['stimulus_id'], row['stimulus_id']])
"
    "${PY}" -m word2psy.cli clip_text "${WORDS_CSV}" \
        --text-column word --id-column stimulus_id \
        -o "${OUT_ROOT}/twp1000/clip_text.csv"
fi

echo "=== 3/3 viz2psy clip: movie frames (0.5 s grid) ==="
MOVIE_LIST=$("${PY}" -c "
import csv
with open('${REGISTRY}/movies.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        print(f\"{row['stimulus_id']}\t{row['video_file']}\")
")

n_done=0 n_skip=0 n_fail=0
failed_ids=()
while IFS=$'\t' read -r MOVIE_ID MOVIE_FILE; do
    OUT_DIR="${OUT_ROOT}/movies/${MOVIE_ID}"
    if [[ -f "${OUT_DIR}/clip.csv" ]]; then
        echo "--- ${MOVIE_ID}: complete, skipping"
        n_skip=$((n_skip + 1))
        continue
    fi
    echo "--- ${MOVIE_ID} ($(date -Is))"
    mkdir -p "${OUT_DIR}"
    # A failed clip is reported at the end but does not kill the sweep.
    if "${PY}" -m viz2psy.cli clip "${STIM_DIR}/movies/${MOVIE_FILE}" \
            -o "${OUT_DIR}/clip.csv" \
            --stimulus-id "${MOVIE_ID}" --frame-interval 0.5 --no-save-frames \
            --batch-size 64 --no-viz --quiet; then
        n_done=$((n_done + 1))
    else
        echo "!!! ${MOVIE_ID} FAILED"
        n_fail=$((n_fail + 1))
        failed_ids+=("${MOVIE_ID}")
    fi
done <<< "${MOVIE_LIST}"

echo "Finished $(date -Is): ${n_done} extracted, ${n_skip} skipped, ${n_fail} failed"
if (( n_fail > 0 )); then
    echo "Failed clips: ${failed_ids[*]}"
    exit 1
fi

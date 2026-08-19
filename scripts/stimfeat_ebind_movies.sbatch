#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=stimfeat_ebind_movies
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_ebind_movies_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_ebind_movies_%j.err

#==============================================================================
# EBind movie extraction — all registry clips (workbench ebind-shared-space)
#==============================================================================
# Movie-only companion to stimfeat_ebind.sbatch: runs its steps 3-4 (frame
# and soundtrack EBind embeddings, 0.5 s grid, checkpoint
# encord-team/ebind-full, Contract B §4.1 sidecars) for EVERY movie in the
# stimulus registry, one at a time. Clips whose ebind.csv AND ebind_audio.csv
# already exist are skipped, so the job is safe to re-run after a failure.
# The shared1000/twp1000 steps are deliberately absent — their outputs are
# complete and a rerun would overwrite them.
#
# Usage:  sbatch stimfeat_ebind_movies.sbatch
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

# stimulus_id<TAB>video_file for every registry movie
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
    if [[ -f "${OUT_DIR}/ebind.csv" && -f "${OUT_DIR}/ebind_audio.csv" ]]; then
        echo "--- ${MOVIE_ID}: complete, skipping"
        n_skip=$((n_skip + 1))
        continue
    fi
    echo "=== ${MOVIE_ID} ($(date -Is)) ==="
    mkdir -p "${OUT_DIR}"
    # A failed clip is reported at the end but does not kill the sweep.
    if "${PY}" -m viz2psy.cli ebind "${STIM_DIR}/movies/${MOVIE_FILE}" \
            -o "${OUT_DIR}/ebind.csv" \
            --stimulus-id "${MOVIE_ID}" --frame-interval 0.5 --no-save-frames \
            --batch-size 64 --no-viz --quiet \
       && "${PY}" -m aud2psy.cli ebind_audio "${STIM_DIR}/movies/${MOVIE_FILE}" \
            -o "${OUT_DIR}/ebind_audio.csv" \
            --stimulus-id "${MOVIE_ID}" --hop 0.5; then
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

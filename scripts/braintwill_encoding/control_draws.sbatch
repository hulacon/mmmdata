#!/bin/bash
#SBATCH --job-name=ctrl_draws
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/ctrl_draws_%j.out
# Sizing: unmeasured. Loading the films frames table whole peaks ~24 GB; each set's
# matched_subspaces is ~2.8k QR + projections of a ~22k x ~600 pool. Read /usr/bin/time and resize.

#==============================================================================
# matched-subspace control draws for the encoding harness (stimfeat env: psytwill)
#==============================================================================
# Driver: scripts/braintwill_encoding/control_draws.py.
#
#   sbatch --export=ALL,POOL=<pool> control_draws.sbatch              (build: pool, bases, draws)
#   sbatch --export=ALL,POOL=<pool>,VERB=draws control_draws.sbatch   (draw features only)
#
# Outputs: <derivatives>/encoding_harness/controls/<pool>/ (pool cache, one .npz per set,
# draws/<set>/<film>.npy, pool.json). The harness reads them as ctrl:<pool>.<set>[.dNNN] arm sources.
#==============================================================================

set -euo pipefail
: "${POOL:?set POOL}"
VERB="${VERB:-build}"
case "$VERB" in build|draws) ;; *) echo "VERB must be build or draws, got $VERB" >&2; exit 2;; esac
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts/braintwill_encoding
PY=/gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
/usr/bin/time -v "$PY" -u control_draws.py "$VERB" --pool "$POOL"
"$PY" control_draws.py show --pool "$POOL"

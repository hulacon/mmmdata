#!/usr/bin/env bash
# Build the shared "rapidtide" conda environment on Talapas.
#
# One env for rapidtide's `happy` (cardiac waveform from BOLD). rapidtide 3.x
# runs its deep-learning cardiac filter on torch (TensorFlow is an optional
# extra, not installed here). The filter is a small 1-D network, so this env
# takes the CPU torch build: no GPU needed, and the wheel is ~10x smaller.
# rapidtide 3.2.0 caps torch below 2.13.
#
# Follows scripts/functional_space/setup_env.sh (itself after psytwill's
# stimfeat builder):
#   - `conda create --override-channels -c conda-forge`: the FSL installer's
#     ~/.condarc `channels: #!final` defeats every other way of choosing a
#     channel.
#   - PYTHONNOUSERSITE=1: ~/.local site-packages sit AHEAD of a conda env's
#     own site-packages.
#   - shared prefix, shared pip cache, group rwX.
#
#   ./scripts/happy/setup_env.sh                # shared prefix
#   ./scripts/happy/setup_env.sh --prefix PATH  # elsewhere
#
# Run on a compute node (srun/sbatch); compute nodes have network access.
# Rebuild policy: single writer. Delete the prefix and rebuild fresh before
# committing a new lock (an incremental solve is not a fresh one).
set -euo pipefail

umask 0002  # group-writable files from birth; parent setgid dirs handle group

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS=/gpfs/projects/hulacon/shared/envs
PREFIX=$ENVS/rapidtide
CACHE=$ENVS/cache
CONDA_MODULE=miniconda3/20260319
LOCK_DIR="$SCRIPT_DIR/env"
TORCH_INDEX=https://download.pytorch.org/whl/cpu
# Pin the local version: pip otherwise lets a different installed build
# satisfy a bare pin.
TORCH_PIN='torch==2.12.1+cpu'
RAPIDTIDE_PIN='rapidtide==3.2.0'
# rapidtide requires PyQt6 (for its GUI tools; happy does not use it). PyQt6
# 6.10+ ships no wheel for glibc 2.28 (RHEL 8), so pip falls back to an sdist
# build that fails; 6.9.1 is the newest with a manylinux_2_28 wheel.
PYQT_PIN='pyqt6==6.9.1'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)  PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# conda: a bare `conda` on $PATH here is FSL's — load the module explicitly.
source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$CONDA_MODULE" 2>/dev/null || {
  echo "error: could not 'module load $CONDA_MODULE'" >&2; exit 1
}

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$CACHE/pip"   # shared cache; also spares /home quota
mkdir -p "$CACHE/pip"

echo "prefix : $PREFIX"
echo "conda  : python=3.12 pip (conda-forge, --override-channels)"
echo "pip    : $TORCH_PIN + $RAPIDTIDE_PIN + $PYQT_PIN"
echo

if [[ -d "$PREFIX" && -n "$(ls -A "$PREFIX" 2>/dev/null)" ]]; then
  echo "==> prefix exists; conda step skipped (delete it for a fresh solve)"
else
  echo "==> conda create"
  conda create --prefix "$PREFIX" --override-channels -c conda-forge --yes \
    python=3.12 pip
fi

PY="$PREFIX/bin/python"

# torch first, from the PyTorch CPU index only, so the resolve below cannot
# pick a CUDA build from PyPI.
echo
echo "==> pip install torch (CPU)"
"$PY" -m pip install --index-url "$TORCH_INDEX" "$TORCH_PIN"

echo
echo "==> pip install rapidtide"
"$PY" -m pip install "$RAPIDTIDE_PIN" "$PYQT_PIN"

# Import check from INSIDE the prefix, then exercise the CLI entry point.
echo
"$PY" - "$PREFIX" <<'PYCHECK'
import sys
prefix = sys.argv[1]
assert sys.executable.startswith(prefix), sys.executable
import numpy, scipy, nibabel, torch, rapidtide
import rapidtide.happy_supportfuncs  # noqa: F401
import rapidtide.dlfiltertorch  # noqa: F401
bad = []
for m in (numpy, scipy, nibabel, torch, rapidtide):
    where = getattr(m, '__file__', '') or ''
    print(f'{m.__name__:<10}', getattr(m, '__version__', 'unknown'))
    if not where.startswith(prefix):
        bad.append(f'{m.__name__} imported from {where}')
if bad:
    print('\nFAIL: modules resolved from outside the environment:', file=sys.stderr)
    print('\n'.join(bad), file=sys.stderr)
    sys.exit(1)
if torch.version.cuda is not None:
    sys.exit(f'FAIL: torch has a CUDA build ({torch.version.cuda}); expected CPU')
PYCHECK
"$PREFIX/bin/happy" --help > /dev/null
echo "happy --help runs"

# Lock manifests: conda side (explicit URLs) + pip side.
mkdir -p "$LOCK_DIR"
CONDA_LOCK="$LOCK_DIR/lock-linux-64.txt"
PIP_LOCK="$LOCK_DIR/pip-lock-linux-64.txt"
echo
echo "==> writing $CONDA_LOCK"
{
  echo "# rapidtide conda layer, resolved by setup_env.sh against a fresh"
  echo "# prefix. Recreate: conda create --prefix <p> --file env/lock-linux-64.txt"
  echo "# then the pip layer per env/pip-lock-linux-64.txt."
  conda list --prefix "$PREFIX" --explicit
} > "$CONDA_LOCK"
echo "==> writing $PIP_LOCK"
{
  echo "# rapidtide pip layer, resolved by setup_env.sh."
  echo "# torch comes from $TORCH_INDEX ($TORCH_PIN)."
  "$PY" -m pip list --format=freeze
} > "$PIP_LOCK"
echo "    $(grep -cv '^#' "$PIP_LOCK") pip packages pinned"

# activate.d hook: user-site guard for every `conda activate` user.
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
cat > "$PREFIX/etc/conda/activate.d/rapidtide-env.sh" <<'HOOK'
# Written by mmmdata/scripts/happy/setup_env.sh.
# ~/.local site-packages shadow a conda env's own — disable user site.
export _RAPIDTIDE_SAVED_PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-}"
export PYTHONNOUSERSITE=1
HOOK
cat > "$PREFIX/etc/conda/deactivate.d/rapidtide-env.sh" <<'HOOK'
if [ -n "${_RAPIDTIDE_SAVED_PYTHONNOUSERSITE:-}" ]; then
  export PYTHONNOUSERSITE="$_RAPIDTIDE_SAVED_PYTHONNOUSERSITE"
else
  unset PYTHONNOUSERSITE
fi
unset _RAPIDTIDE_SAVED_PYTHONNOUSERSITE
HOOK
echo
echo "done: $PREFIX"

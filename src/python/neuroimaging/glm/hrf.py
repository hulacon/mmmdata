"""HRF kernels beyond nilearn's named models: GLMsingle's canonical library.

GLMsingle ships 20 canonical HRFs (``getcanonicalhrflibrary.tsv``, 501 samples
at 0.1 s, each the response to a 0.1 s stimulus) and, per voxel, the index of
the one that fit best (``HRFindex`` in every TYPEB and later output). The
bake-off's per-voxel HRF arm (glm-strategy log, DECIDED 2026-09-08) reuses
that choice: kernel ``k`` becomes a nilearn-compatible callable, so a design
built with ``hrf_model="glmsingle:k"`` differs from the ``"spm"`` design only
in the kernel. Which voxels get which design is :mod:`.voxelwise_hrf`'s job.

The library is read from the installed ``glmsingle`` package rather than
vendored, so the kernels are exactly the ones the encoding fits used.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

LIBRARY_DT = 0.1  # s; sampling of the shipped library
LIBRARY_SIZE = 20
GLMSINGLE_PREFIX = "glmsingle:"
_NILEARN_NAMED = {
    "spm", "spm + derivative", "spm + derivative + dispersion",
    "glover", "glover + derivative", "glover + derivative + dispersion",
}


@lru_cache(maxsize=1)
def glmsingle_library() -> np.ndarray:
    """The 20 library HRFs, shape (20, 501), at 0.1 s, each peak-normalised.

    Raises ImportError naming the package when ``glmsingle`` is not installed:
    the per-voxel arm cannot run without the very library the ``HRFindex``
    maps index into.
    """
    try:
        import glmsingle.hrf as hrf_pkg
    except ImportError as exc:
        raise ImportError(
            "glmsingle is not importable, so its HRF library cannot be read. The "
            "per-voxel HRF arm needs the same package that produced the HRFindex maps."
        ) from exc
    path = Path(hrf_pkg.__file__).resolve().parent / "getcanonicalhrflibrary.tsv"
    lib = np.genfromtxt(path).T
    if lib.shape[0] != LIBRARY_SIZE:
        raise ValueError(f"{path}: expected {LIBRARY_SIZE} HRFs, found shape {lib.shape}")
    peaks = np.abs(lib).max(axis=1, keepdims=True)
    return lib / peaks


def library_kernel(k: int) -> Callable[..., np.ndarray]:
    """Kernel ``k`` as the callable nilearn accepts for ``hrf_model``.

    nilearn calls ``hrf_model(t_r, oversampling)`` and expects the impulse
    response sampled at ``t_r / oversampling``. The library sample is
    interpolated onto that grid over its full 50 s support.
    """
    if not 0 <= k < LIBRARY_SIZE:
        raise ValueError(f"GLMsingle library index must be in 0..{LIBRARY_SIZE - 1}, got {k}")
    lib = glmsingle_library()
    t_lib = np.arange(lib.shape[1]) * LIBRARY_DT

    def kernel(t_r: float, oversampling: int = 50, time_length: float = 50.0, onset: float = 0.0):
        dt = t_r / oversampling
        t = np.arange(0, time_length, dt) - onset
        return np.interp(t, t_lib, lib[k], left=0.0, right=0.0)

    kernel.__name__ = f"glmsingle_hrf_{k:02d}"
    return kernel


def resolve_hrf_model(name: Union[str, Callable, None]) -> Any:
    """What to hand nilearn for a config's ``hrf_model``.

    Named nilearn models pass through; ``"glmsingle:<k>"`` becomes the library
    callable. Anything else is refused by name so a typo cannot fall back to
    nilearn's default silently.
    """
    if name is None or callable(name):
        return name
    if name in _NILEARN_NAMED or name == "fir":
        return name
    m = re.fullmatch(rf"{GLMSINGLE_PREFIX}(\d+)", name)
    if m:
        return library_kernel(int(m.group(1)))
    raise ValueError(
        f"Unknown hrf_model {name!r}; use a nilearn name ({sorted(_NILEARN_NAMED)}) or "
        f"'{GLMSINGLE_PREFIX}<k>' for GLMsingle library kernel k (0..{LIBRARY_SIZE - 1})"
    )


def hrfindex_to_image(hrfindex: np.ndarray, reference: Any) -> Any:
    """Wrap a GLMsingle ``HRFindex`` array as a NIfTI on ``reference``'s grid.

    Refuses a shape mismatch: an HRFindex from one grid stitched onto another
    would assign kernels to the wrong voxels without any later error.
    """
    import nibabel as nib

    idx = np.asarray(hrfindex)
    ref_shape = tuple(reference.shape[:3])
    if idx.shape != ref_shape:
        raise ValueError(
            f"HRFindex shape {idx.shape} does not match the reference grid {ref_shape}; "
            "the encoding fit and this fit must share a space and resolution"
        )
    if idx.min() < 0 or idx.max() >= LIBRARY_SIZE:
        raise ValueError(f"HRFindex values must be in 0..{LIBRARY_SIZE - 1}; found {idx.min()}..{idx.max()}")
    return nib.Nifti1Image(idx.astype(np.int16), reference.affine)


def extract_hrfindex(glmsingle_npy: Path) -> np.ndarray:
    """Read ``HRFindex`` out of a saved GLMsingle output dictionary.

    The TYPEB/C/D files are pickled dicts holding the full beta arrays, so
    this loads the whole file (tens of GB for a 42-run subject); run it once
    on a large-memory node and keep the NIfTI it feeds :func:`hrfindex_to_image`.
    """
    d = np.load(glmsingle_npy, allow_pickle=True).item()
    if "HRFindex" not in d:
        raise KeyError(f"{glmsingle_npy} has no HRFindex (keys: {sorted(d)[:10]}...)")
    return np.asarray(d["HRFindex"])


def load_hrfindex(path: Union[str, Path], reference: Optional[Any] = None) -> Any:
    """Load an HRFindex NIfTI, checking its grid against ``reference`` if given."""
    import nibabel as nib

    img = nib.load(str(path))
    if reference is not None:
        if tuple(img.shape[:3]) != tuple(reference.shape[:3]) or not np.allclose(img.affine, reference.affine, atol=1e-3):
            raise ValueError(
                f"HRFindex {path} is on a different grid (shape {img.shape[:3]}) than the data "
                f"(shape {reference.shape[:3]}); both must be the same space and resolution"
            )
    return img

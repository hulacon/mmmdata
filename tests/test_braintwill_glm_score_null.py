"""score_null.py: voxel-FPR accumulation and the tissue-class partition, on toy inputs."""

import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_glm"
sys.path.insert(0, str(SCRIPTS))
try:
    import score_null as sn  # noqa: E402
except ImportError as exc:  # braintwill not importable: adapter refuses loudly
    pytest.skip(f"score_null not importable: {exc}", allow_module_level=True)


def _write_hits(path: Path, n_vox: int, draws: dict) -> None:
    """``draws`` maps 'kind_contrast_dNN' -> (sig, ok) boolean arrays."""
    arrays = {}
    for key, (sig, ok) in draws.items():
        arrays[key] = np.packbits(sig)
        arrays[f"{key}_ok"] = np.packbits(ok)
    np.savez_compressed(path, **arrays)


def test_accumulate_counts_only_valid_fits(tmp_path):
    n = 11  # not a multiple of 8: unpackbits must honour count
    cell = ("block", "AvsB")
    sig = np.zeros(n, bool)
    sig[[0, 1, 2]] = True
    ok = np.ones(n, bool)
    ok[2] = False                # significant but masked: must count as neither hit nor fit
    ok[10] = False
    _write_hits(tmp_path / "r0.npz", n, {"block_AvsB_d00": (sig, ok), "block_AvsB_d01": (sig & False, ok)})
    _write_hits(tmp_path / "r1.npz", n, {"block_AvsB_d00": (sig, np.ones(n, bool))})
    hits, fits, draws = sn.accumulate([tmp_path / "r0.npz", tmp_path / "r1.npz"], n, [cell])
    assert draws == [2, 1]
    assert hits[cell].tolist() == [2, 2, 1] + [0] * 8
    assert fits[cell].tolist() == [3, 3, 1] + [3] * 7 + [1]


def test_summarise_excludes_voxels_without_a_valid_fit():
    s = sn.summarise(np.array([1, 0, 5, 0]), np.array([10, 10, 10, 0]))
    assert s["voxels"] == 3 and s["excluded_no_valid_fit"] == 1
    assert s["median_fpr"] == pytest.approx(0.1)
    assert s["pooled_mean_fpr"] == pytest.approx(6 / 30)
    assert s["fits_min"] == 10


def test_tissue_classes_partition_with_csf_adjacent_first(tmp_path):
    shape, aff = (7, 7, 7), np.eye(4)
    gm, wm, csf = (np.zeros(shape, np.float32) for _ in range(3))
    wm[:] = 0.6
    gm[:] = 0.4
    gm[:, :, 4:], wm[:, :, 4:] = 0.7, 0.3   # z >= 4 is GM
    csf[0, 0, 0] = 1.0                      # one CSF voxel, outside the mask below
    sub = tmp_path / "sub-01" / "anat"
    sub.mkdir(parents=True)
    for name, arr in (("GM", gm), ("WM", wm), ("CSF", csf)):
        nib.Nifti1Image(arr, aff).to_filename(sub / f"sub-01_label-{name}_probseg.nii.gz")
    mask = np.zeros(shape, np.uint8)
    mask[:2, :2, :] = 1                     # includes the CSF neighbourhood and both GM and WM
    mask[0, 0, 0] = 0                       # the CSF voxel itself is outside the mask
    mask_img = nib.Nifti1Image(mask, aff)
    label = sn.tissue_classes(tmp_path, "01", mask_img)
    vol = np.full(shape, -1)
    vol[mask.astype(bool)] = label
    adj = sn.CLASSES.index("CSF-adjacent")
    # adjacency to a CSF voxel outside the mask still counts; 26-neighbourhood
    assert vol[1, 1, 1] == adj and vol[1, 0, 0] == adj and vol[0, 1, 1] == adj
    assert vol[1, 1, 2] == sn.CLASSES.index("WM")
    assert vol[1, 1, 5] == sn.CLASSES.index("GM")
    assert set(np.unique(label)) <= {0, 1, 2}
    assert label.size == mask.sum()

"""scripts/braintwill_encoding/nsd.py — the NSD shared1000 register (synthetic responses)."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NSD_REGISTER = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_encoding" / "nsd.py"


@pytest.fixture(scope="module")
def reg():
    spec = importlib.util.spec_from_file_location("braintwill_encoding_nsd", NSD_REGISTER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (ImportError, SyntaxError) as exc:  # braintwill needs Python >= 3.12
        pytest.skip(f"braintwill.encoding not importable here: {exc}")
    return mod


@pytest.fixture
def root(tmp_path):
    """subj01/fsaverage6 with 40 images (one with 2 trials), 6 units, 2 ROIs; features in a TSV."""
    rng = np.random.default_rng(0)
    d = tmp_path / "out" / "subj01" / "fsaverage6"
    d.mkdir(parents=True)
    n = 40
    ids = [f"shared{i + 1:04d}_nsd{i:05d}" for i in range(n)]
    pd.DataFrame({"shared_idx": np.arange(1, n + 1), "stimulus_id": ids,
                  "n_reps": [2] + [3] * (n - 1)}).to_csv(d / "images.tsv", sep="\t", index=False)
    pd.DataFrame({"stimulus_id": ids}).to_csv(d / "trials.tsv", sep="\t", index=False)
    feats = pd.DataFrame({"stimulus_id": ids, "a_1": rng.normal(size=n), "a_2": rng.normal(size=n),
                          "b_1": rng.normal(size=n)})
    y = np.column_stack([feats.a_1 * 2 + rng.normal(scale=.3, size=n), rng.normal(size=(n, 5))])
    np.save(d / "image_mean_z.npy", (3 + y).astype(np.float32))
    np.savez(d / "rois.npz", first=np.array([1, 1, 0, 0, 0, 0], bool), last=np.array([0, 0, 0, 0, 0, 1], bool))
    feats.to_csv(tmp_path / "feats.tsv", sep="\t", index=False)
    return tmp_path


def _args(root, **kw):
    a = dict(subj=1, space="fsaverage6", root=root / "out", features=str(root / "feats.tsv"),
             bands="a=^a_,b=^b_", rois=None, min_reps=None, outer_folds=5, solver="random_search",
             n_iter=None, backend="numpy", cache=root / "cache")
    a.update(kw)
    return argparse.Namespace(**a)


def test_table_features_assign_bands_in_order(reg, root):
    f = reg.table_features(root / "feats.tsv", {"a": "^a_", "b": "^b_"})
    first = next(iter(f.values()))
    assert first.bands == {"a": slice(0, 2), "b": slice(2, 3)} and first.x.shape == (1, 3)
    with pytest.raises(KeyError, match="band 'c'"):
        reg.table_features(root / "feats.tsv", {"c": "^c_"})


def test_rows_columns_and_standardization(reg, root):
    req = reg.Request(_args(root, rois="first,last"))
    assert len(req.rows) == 39  # the image with 2 trials is left out by default
    assert req.units.tolist() == [0, 1, 5]
    x, y, groups, bands = req.design()
    assert x.shape == (39, 3) and y.shape == (39, 3) and list(groups) == req.rows.stimulus_id.tolist()
    assert np.allclose(y.mean(0), 0, atol=1e-5) and np.allclose(y.std(0), 1, atol=1e-4)
    assert len(reg.Request(_args(root, min_reps=2)).rows) == 40


def test_the_key_changes_with_rois_and_folds(reg, root):
    k = reg.Request(_args(root)).key
    assert reg.Request(_args(root, rois="first")).key != k
    assert reg.Request(_args(root, outer_folds=10)).key != k
    assert reg.Request(_args(root)).key == k


def test_unknown_roi_is_refused(reg, root):
    with pytest.raises(SystemExit, match="no ROI"):
        reg.Request(_args(root, rois="nowhere"))


def test_score_is_cached_and_finds_the_signal(reg, root, capsys):
    pytest.importorskip("himalaya")
    reg.cmd_score(_args(root, n_iter=3))
    first = capsys.readouterr().out
    assert first.startswith("wrote ")
    reg.cmd_score(_args(root, n_iter=3))
    assert capsys.readouterr().out.startswith("cached: ")
    req = reg.Request(_args(root, n_iter=3))
    hit = reg.ScoreCache(root / "cache").get(req.key)
    assert hit.r2[0] > 0.5 and np.nanmax(hit.r2[1:]) < hit.r2[0]

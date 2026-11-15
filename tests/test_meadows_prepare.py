"""meadows_prepare: the Meadows arrangement and rating files become bench inputs keyed by shared1000 name."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "benchmarks"


@pytest.fixture(scope="module")
def mp():
    spec = importlib.util.spec_from_file_location("meadows_prepare", HERE / "meadows_prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture(tmp_path, *, n=5, parts=("p-a", "p-b"), double_rate=False):
    sio = pytest.importorskip("scipy.io")
    names = [f"shared{k:04d}_nsd{k + 100:05d}" for k in range(1, n + 1)]
    s1000 = tmp_path / "shared1000"
    (s1000 / "images").mkdir(parents=True)
    pd.DataFrame({"fileNames": [f"{s}.png" for s in names]}).to_csv(s1000 / "shared1000.csv", index=False)
    for k, s in enumerate(names):
        (s1000 / "images" / f"{s}.png").write_bytes(b"x" * (10 + k))
    stim = [{"name": s, "size": 10 + k} for k, s in enumerate(names)]
    rng = np.random.default_rng(0)
    tree = {p: {"tasks": [{"task": {"task_type": "multiarrange"}, "stimuli": stim[::-1],
                           "rdm": list(rng.uniform(size=n * (n - 1) // 2))}]} for p in parts}
    root = tmp_path / "meadows"
    root.mkdir()
    (root / "Meadows_nsd-multiple-arrangements_v_v2_tree.json").write_text(json.dumps(tree))
    tasks = ["valence - block 1", "valence - block 2", "arousal - block 1", "arousal - block 2"]
    prop = np.full((4, len(parts), n), np.nan)
    half = n // 2
    prop[0, :, :half] = 0.1
    prop[1, :, half:] = 0.2
    prop[2, :, :] = 0.7
    if double_rate:
        prop[3, 0, 0] = 0.9
    sio.savemat(root / "Meadows_nsd-multiple-arrangements_v_v2_3D.mat",
                {"dragrate_tasks": np.array(tasks, dtype=object), "participation_ids": np.array(parts, dtype=object),
                 "stimuli": np.array(names, dtype=object), "property": prop})
    return root, s1000, tree, names


def test_prepare_writes_pairs_in_squareform_order_and_ratings(mp, tmp_path):
    from scipy.spatial.distance import squareform

    root, s1000, tree, names = _fixture(tmp_path)
    out = tmp_path / "out"
    assert mp.main(["--root", str(root), "-o", str(out), "--shared1000", str(s1000), "--check-bytes"]) == 0
    pairs = pd.read_csv(out / "rdm_pairs.csv")
    assert len(pairs) == 2 * 10 and (pairs["image_a"] < pairs["image_b"]).all()
    D = squareform(np.asarray(tree["p-a"]["tasks"][0]["rdm"]))
    order = [s["name"] for s in tree["p-a"]["tasks"][0]["stimuli"]]  # reversed relative to the ids
    row = pairs[(pairs["participant"] == "p-a") & (pairs["image_a"] == names[0]) & (pairs["image_b"] == names[1])]
    assert row["dissimilarity"].item() == pytest.approx(D[order.index(names[0]), order.index(names[1])])
    r = pd.read_csv(out / "ratings.csv")
    assert list(r.columns) == ["participant", "stimulus_id", "valence", "arousal"]
    assert set(r["valence"]) == {0.1, 0.2} and (r["arousal"] == 0.7).all()
    rec = json.loads((out / "prepare.json").read_text())
    assert rec["checks"]["byte_sizes_match"] == "5/5"


def test_prepare_refuses_wrong_bytes_unknown_names_and_double_ratings(mp, tmp_path):
    root, s1000, _, names = _fixture(tmp_path / "a")
    (s1000 / "images" / f"{names[2]}.png").write_bytes(b"y")
    with pytest.raises(SystemExit, match="byte size"):
        mp.main(["--root", str(root), "-o", str(tmp_path / "o1"), "--shared1000", str(s1000), "--check-bytes"])
    pd.DataFrame({"fileNames": [f"{s}.png" for s in names[1:]]}).to_csv(s1000 / "shared1000.csv", index=False)
    with pytest.raises(SystemExit, match="not shared1000"):
        mp.main(["--root", str(root), "-o", str(tmp_path / "o2"), "--shared1000", str(s1000)])
    root, s1000, _, _ = _fixture(tmp_path / "b", double_rate=True)
    with pytest.raises(SystemExit, match="exactly 1"):
        mp.main(["--root", str(root), "-o", str(tmp_path / "o3"), "--shared1000", str(s1000)])

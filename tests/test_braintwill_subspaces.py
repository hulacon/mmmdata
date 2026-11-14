"""The settle-3 subspace reader (``subspaces.py``): its plan, gate, contrasts and pooling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_encoding"


@pytest.fixture(scope="module")
def ss():
    spec = importlib.util.spec_from_file_location("braintwill_encoding_subspaces", HERE / "subspaces.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (ImportError, SyntaxError) as exc:  # braintwill needs Python >= 3.12
        pytest.skip(f"braintwill.encoding not importable here: {exc}")
    return mod


def _objects(n=3, members=("a", "b"), empty=("c",)):
    o = {"S": {"kind": "shared", "dim": 4}, "U": {"kind": "unique-union", "dim": 3}}
    for m in members:
        o[f"U:{m}"] = {"kind": "unique", "dim": 2}
    for m in empty:
        o[f"U:{m}"] = {"kind": "unique", "dim": 0}
    for r in range(n):
        for k in ("S", "U", "ref"):
            o[f"{k}@d{r:03d}"] = {"kind": "shared", "dim": 3}
        for m in members:
            o[f"S@{m}.d{r:03d}"] = {"kind": "shared", "dim": 2}
    return o


def test_plan_finds_draws_by_index_and_members_with_a_unique_subspace(ss):
    p = ss.plan(_objects())
    assert p["draw_idx"] == [0, 1, 2] and p["members"] == ["a", "b"]
    assert p["draws"]["ref"][2] == "ref@d002" and p["member_draws"]["b"][1] == "S@b.d001"
    bad = _objects()
    del bad["U@d001"]
    with pytest.raises(SystemExit, match="do not share their indices"):
        ss.plan(bad)
    bad = _objects()
    del bad["S@a.d000"]
    with pytest.raises(SystemExit, match="member a"):
        ss.plan(bad)


def test_combinations_follow_the_gate(ss):
    p = ss.plan(_objects())
    gate = {"S": True, "U": True, "U:a": True, "U:b": False}
    combos, skipped = ss.combinations(p, gate)
    assert combos["delta"] == {**{f"S@d{r:03d}": pytest.approx(1 / 3) for r in range(3)},
                               **{f"U@d{r:03d}": pytest.approx(-1 / 3) for r in range(3)}}
    assert combos["U:a_vs_S"]["U:a"] == 1.0 and combos["U:a_vs_S"]["S@a.d002"] == pytest.approx(-1 / 3)
    assert "U:b" not in combos and skipped == {"U:b": "gate: False"}
    combos, skipped = ss.combinations(p, {"S": True, "U": False, "U:a": True})
    assert "delta" not in combos and "block" in skipped and "U:a" in combos


def _sums(rng, y, signal, n_groups):
    p = signal * y + rng.standard_normal(y.shape) * np.sqrt(1 - signal ** 2)
    yg, pg = np.split(y, n_groups), np.split(p, n_groups)
    return {"labels": [f"film-{i}" for i in range(n_groups)], "n": np.array([len(a) for a in yg]),
            "sy": np.array([a.sum(0) for a in yg]), "syy": np.array([(a ** 2).sum(0) for a in yg]),
            "res": np.array([((a - b) ** 2).sum(0) for a, b in zip(yg, pg)])}


def test_cells_read_the_contrast_and_the_draws_and_pooling_finds_a_planted_roi(ss):
    from braintwill.encoding import draw_counts, resampled_r2

    rng = np.random.default_rng(0)
    n_groups, per, cols = 30, 40, 60
    objs = _objects(n=4, members=(), empty=())
    p = ss.plan(objs)
    combos, _ = ss.combinations(p, {"S": True, "U": True})
    rois = {"r1": np.arange(0, 20), "r2": np.arange(20, 40), "r3": np.arange(40, 60)}
    cells = []
    for sub in ("03", "04", "05"):
        y = rng.standard_normal((n_groups * per, cols))
        strength = np.r_[np.full(20, 0.6), np.full(40, 0.3)]  # S beats U in r1 only
        groups = {}
        for name in objs:
            s = strength if name.startswith("S") else np.full(cols, 0.3)
            groups[name] = _sums(rng, y, s, n_groups)
        counts = draw_counts(n_groups, 300, 0)
        got = ss.cell_stats(groups, rois, counts, combos, 0.95)
        by = {(r["read"], r["roi"]): r for r in got}
        ones = np.ones(n_groups)
        hand = np.mean([np.nanmean(resampled_r2(groups[f"S@d{i:03d}"], ones)[rois["r1"]])
                        - np.nanmean(resampled_r2(groups[f"U@d{i:03d}"], ones)[rois["r1"]]) for i in range(4)])
        assert by[("delta", "r1")]["value"] == pytest.approx(hand, abs=1e-12)
        assert by[("delta", "r1")]["lo"] > 0 and by[("delta", "r1")]["sd"] > 0
        d = pd.DataFrame(ss.draw_deltas(groups, rois, p))
        assert d.groupby("roi")["delta"].mean()["r1"] == pytest.approx(hand, abs=1e-12)
        cells += [{"block": "L.x", "sub": sub, **r} for r in got]
    pooled = ss.pool(pd.DataFrame(cells), 0.90).set_index("roi")
    assert bool(pooled.loc["r1", "above_zero"]) and not bool(pooled.loc["r2", "above_zero"])
    assert "(mean over ROIs)" in pooled.index

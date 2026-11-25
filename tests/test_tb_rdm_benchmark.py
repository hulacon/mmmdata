"""rdm_benchmark.py: the fraction recovers shared item structure, sits at the
null for noise, and the co-run mask covers exactly the same-run item pairs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tb_item_signal"))
import rdm_benchmark as rb  # noqa: E402

N_ITEMS, N_RUNS, K = 60, 12, 5


def make_cache(path, subject, signal, seed, F, n_vox=80):
    """3 exposures per item over N_RUNS runs in 2 sessions, one sequence shared
    by every cache (as NSD's subjects share one); patterns = signal * F @ W
    (cache-specific W) + noise; plus 2 anchors."""
    rng = np.random.default_rng(seed)
    design = np.random.default_rng(0)
    W = rng.standard_normal((K, n_vox))
    rows = []
    for rep in range(3):
        order = design.permutation(N_ITEMS)
        for pos, i in enumerate(order):
            run = (pos * N_RUNS // N_ITEMS + rep) % N_RUNS + 1
            rows.append((f"ses-{1 + (run > N_RUNS // 2):02d}", run, str(i + 1), 0, float(rep * 100 + pos)))
    for a in ("901", "902"):
        for ses in ("ses-01", "ses-02"):
            rows.append((ses, 1 if ses == "ses-01" else N_RUNS, a, 1, 999.0))
    t = pd.DataFrame(rows, columns=["session", "run", "mmmId", "sharedId", "onset"])
    item = t["mmmId"].astype(int).clip(upper=N_ITEMS) - 1
    P = signal * (F[item] @ W) + rng.standard_normal((len(t), n_vox))
    np.savez(path, subject=subject, arm="syn", roi_names=np.array(["R"]),
             patterns_R=P.T.astype(np.float32),
             session=t["session"].to_numpy().astype(object), run=t["run"].to_numpy(),
             task=np.array(["x"] * len(t), dtype=object),
             mmmId=t["mmmId"].to_numpy().astype(object), sharedId=t["sharedId"].to_numpy(float),
             enCon=np.full(len(t), np.nan), onset=t["onset"].to_numpy())
    return path


@pytest.fixture
def caches(tmp_path):
    F = np.random.default_rng(0).standard_normal((N_ITEMS, K))
    refs = [make_cache(tmp_path / f"ref{k}.npz", f"ref{k}", 1.0, 10 + k, F) for k in range(3)]
    good = make_cache(tmp_path / "good.npz", "good", 1.0, 20, F)
    noise = make_cache(tmp_path / "noise.npz", "noise", 0.0, 21, F)
    return refs, good, noise


def run(tmp_path, targets, refs, *extra):
    out = tmp_path / "out.tsv"
    rb.main(["--target", *map(str, targets), "--reference", *map(str, refs), "--rois", "R",
             "--n-boot", "50", "--n-perm", "100", *extra, "--out", str(out)])
    return pd.read_csv(out, sep="\t")


def get(df, metric, label):
    return df[(df.metric == metric) & (df.label == label)].iloc[0]


def test_fraction_recovers_signal_and_noise_sits_at_null(tmp_path, caches):
    refs, good, noise = caches
    df = run(tmp_path, [good, noise], refs)
    assert (df.n_items == N_ITEMS).all()                       # anchors dropped
    g, n = get(df, "fraction", "syn:good"), get(df, "fraction", "syn:noise")
    assert 0.8 < g.value < 1.2
    assert abs(n.value) < 0.2
    assert get(df, "target_ref", "syn:good").p_perm < 0.05
    assert get(df, "target_ref", "syn:noise").p_perm > 0.05
    assert get(df, "ref_ref", "all").value > 0.3
    assert g.boot_lo < g.value < g.boot_hi


def test_corun_mask_marks_exactly_same_run_pairs(tmp_path, caches):
    refs, _, _ = caches
    d = np.load(refs[0], allow_pickle=True)
    t = rb.load_trials(d)
    items = sorted(t.loc[~t.anchor, "mmmId"].unique(), key=int)
    M = rb.corun(t, items)
    runs = t[~t.anchor].groupby("mmmId")["run_key"].apply(set)
    for a in range(0, N_ITEMS, 7):
        for b in range(0, N_ITEMS, 5):
            assert M[a, b] == bool(runs[items[a]] & runs[items[b]])


def test_needs_two_references(tmp_path, caches):
    refs, good, _ = caches
    with pytest.raises(SystemExit):
        run(tmp_path, [good], refs[:1])


def test_session_mask_is_wider_than_run_mask(tmp_path, caches):
    refs, good, _ = caches
    run_ = run(tmp_path, [good], refs)
    ses = run(tmp_path, [good], refs, "--mask-on", "session")
    none = run(tmp_path, [good], refs, "--mask-on", "none")
    assert none.n_cells.iloc[0] == N_ITEMS * (N_ITEMS - 1) // 2
    assert ses.n_cells.iloc[0] < run_.n_cells.iloc[0] < none.n_cells.iloc[0]

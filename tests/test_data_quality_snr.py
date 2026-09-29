"""data-quality tier 2 snr view (T2.8): run table, task summary, per-session view, coverage, build + diff."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroimaging import data_quality_snr as dqs

REGIMES = ("none", "gsr")
#: (sub, ses, task, run, tSNR under `none`); `gsr` adds 2
RUNS = [
    ("03", "01", "TBencoding", "1", 40.0),
    ("03", "01", "TBencoding", "2", 42.0),
    ("03", "01", "TBmath", "n/a", 20.0),
    ("03", "02", "TBencoding", "1", 50.0),
    ("03", "02", "TBmath", "n/a", 25.0),
    ("03", "02", "INITresting", "n/a", 60.0),
    ("06", "01", "TBencoding", "1", 30.0),
]


def _tier1(absent: tuple = ()) -> pd.DataFrame:
    """tier1_runs.tsv as `load_tier1_runs` returns it: every column a string, `absent` a bool."""
    rows = []
    for sub, ses, task, run, tsnr in RUNS:
        for i, regime in enumerate(REGIMES):
            gone = (sub, ses, task, run, regime) in absent
            val = (lambda x: "n/a" if gone else str(x))
            rows.append({"sub": sub, "ses": ses, "task": task, "run": run, "regime": regime, "absent": gone,
                         "n_vol": "200", "n_regressors": val(1 + 5 * i), "dof_resid": val(198 - 5 * i),
                         "dof_loss": val(round((2 + 5 * i) / 200, 6)), "tsnr_median_mask": val(tsnr + 2 * i),
                         "mask_n_voxels": val(1000)})
    return pd.DataFrame(rows)


def _motion() -> pd.DataFrame:
    keys = {(s, e, t, r.replace("n/a", "")) for s, e, t, r, _ in RUNS}
    return pd.DataFrame([{"sub": s, "ses": e, "task": t, "run": r, "fd_mean": 0.1 + 0.1 * (e == "02"),
                          "fdf_mean": 0.05, "fd_frac_gt_0.2": 0.01} for s, e, t, r in sorted(keys)])


def _coverage() -> pd.DataFrame:
    m = _motion()[dqs.KEYS]
    return m.assign(cov_median=0.99, n_cov_below_floor=3, hipp_cov_min=0.95)


@pytest.fixture
def runs():
    return dqs.run_table(_tier1(), _motion(), _coverage(), provisional=["06"])


# ---------------------------------------------------------------------------
# Run table
# ---------------------------------------------------------------------------

def test_run_table_has_one_row_per_run_and_regime(runs):
    assert len(runs) == len(RUNS) * len(REGIMES)
    assert set(runs["run"]) == {"1", "2", ""}
    assert runs.loc[runs["sub"] == "06", "provisional"].all() and not runs.loc[runs["sub"] == "03", "provisional"].any()


def test_tsnr_rel_is_relative_to_the_subject_task_regime_median(runs):
    r = runs[(runs["regime"] == "none") & (runs["sub"] == "03")].set_index(["ses", "task", "run"])["tsnr_rel"]
    # TBencoding median 42 (40, 42, 50); TBmath median 22.5 (20, 25)
    assert r[("01", "TBencoding", "1")] == pytest.approx(40 / 42)
    assert r[("02", "TBencoding", "1")] == pytest.approx(50 / 42)
    assert r[("01", "TBmath", "")] == pytest.approx(20 / 22.5)


def test_tsnr_rel_is_na_for_a_task_run_in_only_one_session(runs):
    # INITresting occurs once, sub-06's TBencoding in one session: no reference outside the session
    one = runs[(runs["task"] == "INITresting") | (runs["sub"] == "06")]
    assert len(one) == 2 * len(REGIMES) and one["tsnr_rel"].isna().all()
    s = dqs.sessions(runs).set_index(["sub", "ses", "regime"])
    assert s.loc[("03", "02", "none"), "n_runs"] == 3 and s.loc[("03", "02", "none"), "n_runs_rel"] == 2
    assert np.isnan(s.loc[("06", "01", "none"), "tsnr_rel_median"]) and s.loc[("06", "01", "none"), "n_runs_rel"] == 0


def test_run_table_is_loud_about_a_run_without_motion_or_coverage():
    with pytest.raises(KeyError, match="tier1_motion"):
        dqs.run_table(_tier1(), _motion().iloc[1:], _coverage())
    with pytest.raises(KeyError, match="tier1_parcels"):
        dqs.run_table(_tier1(), _motion(), _coverage().iloc[1:])


def test_absent_cell_is_counted_never_filled_and_keeps_its_mask_size():
    runs = dqs.run_table(_tier1(absent={("03", "01", "TBencoding", "1", "gsr")}), _motion(), _coverage())
    cell = runs[(runs["ses"] == "01") & (runs["run"] == "1") & (runs["regime"] == "gsr") & (runs["sub"] == "03")]
    assert cell["absent"].item() and np.isnan(cell["tsnr_rel"].item())
    assert cell["mask_n_voxels"].item() == 1000
    ts = dqs.task_summary(runs).set_index(["scope", "task", "regime"])
    row = ts.loc[("sub-03", "TBencoding", "gsr")]
    assert row["n_runs"] == 3 and row["n_absent"] == 1
    assert row["tsnr_median"] == pytest.approx(48.0)           # median of 44, 52; the absent 42 is out
    ses = dqs.sessions(runs).set_index(["sub", "ses", "regime"]).loc[("03", "01", "gsr")]
    assert ses["n_absent"] == 1 and ses["n_runs"] == 3


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def test_task_summary_scopes_tasks_and_values(runs):
    ts = dqs.task_summary(runs)
    assert set(ts["scope"]) == {"sub-03", "sub-06", "pooled", "pooled_confirmed"}
    assert set(ts["task"]) == {"all", "INITresting", "TBencoding", "TBmath"}
    row = ts.set_index(["scope", "task", "regime"]).loc[("sub-03", "TBencoding", "none")]
    assert row["tsnr_median"] == 42 and row["tsnr_q25"] == 41 and row["tsnr_q75"] == 46
    assert row["dof_resid_min"] == 198 and row["n_regressors_median"] == 1
    pooled = ts.set_index(["scope", "task", "regime"])
    assert pooled.loc[("pooled", "all", "none"), "n_runs"] == 7 and pooled.loc[("pooled", "all", "none"), "provisional"]
    assert pooled.loc[("pooled_confirmed", "all", "none"), "n_runs"] == 6
    assert not pooled.loc[("pooled_confirmed", "all", "none"), "provisional"]


def test_no_pooled_confirmed_scope_without_provisional_subjects():
    runs = dqs.run_table(_tier1(), _motion(), _coverage())
    assert "pooled_confirmed" not in set(dqs.task_summary(runs)["scope"])


def test_sessions_carry_the_regime_free_columns_on_every_regime(runs):
    s = dqs.sessions(runs)
    assert len(s) == 3 * len(REGIMES)                            # sub-03 ses-01/02, sub-06 ses-01
    s03 = s[s["sub"] == "03"].set_index(["ses", "regime"])
    assert s03.loc[("01", "none"), "tasks"] == "TBencoding,TBmath"
    assert s03.loc[("02", "none"), "fd_mean_median"] == pytest.approx(0.2)
    assert s03.loc[("02", "none"), "fd_mean_median"] == s03.loc[("02", "gsr"), "fd_mean_median"]
    # ses-02 has the high-tSNR TBencoding run AND a TBmath run; relative tSNR reads above 1, raw is task-mixed
    assert s03.loc[("02", "none"), "tsnr_rel_median"] > 1 > s03.loc[("01", "none"), "tsnr_rel_median"]


# ---------------------------------------------------------------------------
# Coverage and I/O
# ---------------------------------------------------------------------------

def _parcels_tsv(path):
    rows = []
    for regime, scale in (("none", 1.0), ("gsr", 0.0)):          # only `none` may be read
        for run, covs in (("1", (1.0, 0.9, 0.5)), ("n/a", (0.7, 0.95, 1.0))):
            for i, c in enumerate(covs):
                rows.append({"sub": "03", "ses": "01", "task": "TBencoding", "run": run, "atlas": dqs.SEG,
                             "regime": regime, "parcel": f"P{i}", "coverage": c * scale})
            for side, c in (("Left Hippocampus", 0.9), ("Right Hippocampus", 0.6), ("Left Thalamus", 0.1)):
                rows.append({"sub": "03", "ses": "01", "task": "TBencoding", "run": run, "atlas": dqs.SUBCORTICAL,
                             "regime": regime, "parcel": side, "coverage": c * scale})
    pd.DataFrame(rows).to_csv(path / "tier1_parcels.tsv", sep="\t", index=False)


def test_load_coverage_reads_the_none_regime_and_both_atlases(tmp_path):
    _parcels_tsv(tmp_path)
    cov = dqs.load_coverage(tmp_path).set_index("run")
    assert cov.loc["1", "cov_median"] == pytest.approx(0.9)       # Schaefer only
    assert cov.loc["1", "n_cov_below_floor"] == 3                 # 0.5, right hippocampus 0.6, thalamus 0.1
    assert cov.loc["", "n_cov_below_floor"] == 3                  # 0.7, 0.6, 0.1
    assert cov.loc["1", "hipp_cov_min"] == pytest.approx(0.6)


def test_load_coverage_is_loud_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="tier1_parcels.tsv"):
        dqs.load_coverage(tmp_path)


def test_rebuild_is_identical_and_diff_catches_a_change(runs, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    dqs.write(dqs.compute(runs), a, {"created": "x"})
    dqs.write(dqs.compute(runs), b, {"created": "y"})
    assert dqs.diff(a, b) == []
    changed = runs.copy()
    changed.loc[0, "tsnr_median_mask"] += 1
    dqs.write(dqs.compute(changed), b, {"created": "y"})
    assert dqs.diff(a, b) == ["task_summary.tsv differs", "sessions.tsv differs"]
    (b / "sessions.tsv").unlink()
    assert "sessions.tsv missing in b" in dqs.diff(a, b)

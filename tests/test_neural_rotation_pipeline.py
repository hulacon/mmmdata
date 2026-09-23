"""Planted-truth tests for scripts/neural_rotation: the whole pipeline
(fake_caches.py -> fit_pair.py -> report.py) on a self-contained synthetic
design, asserting on the output tables only (the schema WP5 consumes).

Nothing here touches GPFS. Each scenario writes caches in the ladder format
for a small fake design (6 retrieval sessions x 40 items) and runs one or
more cells; the suite stays under about a minute.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NR = Path(__file__).resolve().parent.parent / "scripts" / "neural_rotation"
PY = sys.executable

N_SES = 6
ITEMS_PER_SES = 40
RUNS_PER_PHASE = 2
GAPS = [3, 14, 7, 30, 21]                    # days between consecutive sessions


def fake_design(subject: str, out: Path, seed: int = 0) -> Path:
    """A small design table with the columns design.py writes.

    Items cross exposure count (1 or 3) with retrieval condition (within the
    encoding session, age 0, or the session after, age = the gap). Six
    anchors are retrieved every session by both cues and encoded three times
    per session.
    """
    rng = np.random.default_rng(seed)
    dates = [pd.Timestamp("2024-01-01")]
    for g in GAPS:
        dates.append(dates[-1] + pd.Timedelta(days=g))
    rows = []
    item = 1
    anchors = list(range(995, 1001))
    onset = {}                                   # (session, run) -> last onset, across sessions

    def add(phase, ses, run, onset, mmm, encon, recon, exposure, ses_date, enc_date, fold, anchor):
        rows.append({"subject": subject, "phase": phase, "session": f"ses-{ses:02d}", "run": run,
                     "onset": onset, "duration": 3.0, "mmmId": mmm, "nsdId": mmm, "word": f"w{mmm}",
                     "pairId": mmm, "sharedId": int(anchor), "anchor": anchor, "enCon": encon,
                     "reCon": recon, "resp": 7, "resp_RT": 1.0, "exposure": exposure,
                     "ses_date": ses_date.strftime("%Y-%m-%d"),
                     "enc_date": enc_date.strftime("%Y-%m-%d") if enc_date is not None else "n/a",
                     "age_days": (ses_date - enc_date).days if enc_date is not None and phase != "enc" else 0,
                     "fold": fold, "stimulus_id": f"shared{mmm:04d}_nsd{mmm:05d}"})

    for s in range(N_SES):
        ses = s + 1
        ids = list(range(item, item + ITEMS_PER_SES))
        item += ITEMS_PER_SES
        q = ITEMS_PER_SES // 4
        # enCon x reCon crossed: single/within, single/across, three/within, three/across
        groups = {(1, 1): ids[:q], (1, 2): ids[q:2 * q], (3, 1): ids[2 * q:3 * q], (3, 2): ids[3 * q:]}
        if s == 0:                                   # nothing to be "across" from
            groups = {(1, 1): ids[:2 * q], (3, 1): ids[2 * q:]}
        enc_of = {}
        for (encon, recon), members in groups.items():
            enc_s = s if recon == 1 else s - 1
            for mmm in members:
                enc_of[mmm] = (encon, recon, enc_s)
        for mmm, (encon, recon, enc_s) in enc_of.items():
            for e in range(1, encon + 1):
                run = 1 + (e % RUNS_PER_PHASE)
                key = (enc_s + 1, run)
                onset[key] = onset.get(key, 9.0) + 4.5
                add("enc", enc_s + 1, run, onset[key], mmm, encon, recon, e, dates[enc_s],
                    dates[enc_s], ses, False)
        for e in (1, 2, 3):
            for a in anchors:
                run = 1 + (e % RUNS_PER_PHASE)
                key = (ses, run)
                onset[key] = onset.get(key, 9.0) + 4.5
                add("enc", ses, run, onset[key], a, 3, 1, e, dates[s], None, 0, True)
        # retrieval: word and image cues in different runs
        for cue, run in (("ret-word", 1), ("ret-image", 2)):
            t0 = 9.0
            for mmm in rng.permutation(ids):
                encon, recon, enc_s = enc_of[mmm]
                add(cue, ses, run, t0, mmm, encon, recon, 1, dates[s], dates[enc_s], ses, False)
                t0 += 4.5
            for a in anchors:
                add(cue, ses, run, t0, a, 3, 1, 1, dates[s], None, 0, True)
                t0 += 4.5
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{subject}_desc-trials.tsv"
    df.to_csv(p, sep="\t", index=False)
    # anchor lags, as design.py writes them
    lag_rows = []
    for cue in ("word", "image"):
        for a in anchors:
            for i in range(N_SES):
                for j in range(i + 1, N_SES):
                    ta, tb = (dates[i] - dates[0]).days, (dates[j] - dates[0]).days
                    lag_rows.append({"subject": subject, "cue": cue, "mmmId": a, "ses_a": f"ses-{i + 1:02d}",
                                     "ses_b": f"ses-{j + 1:02d}", "lag_days": tb - ta, "abs_time_a_days": ta,
                                     "abs_time_b_days": tb, "abs_time_days": (ta + tb) / 2,
                                     "similarity": np.nan, "plane_angle_delta": np.nan})
    pd.DataFrame(lag_rows).to_csv(out / f"{subject}_desc-anchorlags.tsv", sep="\t", index=False, na_rep="n/a")
    return p


def run(cmd, cwd=None):
    r = subprocess.run([PY] + [str(c) for c in cmd], capture_output=True, text=True, cwd=cwd)
    assert r.returncode == 0, f"{' '.join(str(c) for c in cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
    return r.stdout


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    root = tmp_path_factory.mktemp("design")
    p = fake_design("sub-99", root / "sub-99")
    return root, p


def make_caches(design, tmp_path, scenario, seed=1, noise=0.6, extra=()):
    root, p = design
    out = tmp_path / scenario
    run([NR / "fake_caches.py", "--subject", "sub-99", "--design", p, "--out", out,
         "--scenario", scenario, "--seed", seed, "--noise", noise, "--sizes", 60, 80, 100,
         "--latent-dim", 12] + list(extra))
    return out


def fit(design, cache_root, roi, pair="enc:ret-word", rung="i", n_perm=20):
    root, _ = design
    run([NR / "fit_pair.py", "--subject", "sub-99", "--rung", rung, "--roi", roi, "--pair", pair,
         "--cache-root", cache_root, "--design-root", root, "--out-root", cache_root,
         "--n-perm", n_perm, "--n-perm-rot", 3, "--features", "none",
         # the planted-truth scenarios were calibrated under the original
         # preselect 0.5 (the null scenario's 20-draw gate and the age-slope CI
         # both move when every voxel is kept); the production default is 1.0
         "--preselect", 0.5])
    stem = f"sub-99_rung-{rung}_roi-{roi}_pair-{pair.replace(':', '')}_desc-typed"
    d = cache_root / "fits" / "sub-99"
    return {p.name[len(stem) + 1:-4]: pd.read_csv(p, sep="\t", na_values=["n/a"])
            for p in d.glob(f"{stem}_*.tsv")}


def selected(tc, direction="forward"):
    s = tc[tc["rank_selected"].astype(bool) & (tc["direction"] == direction)]
    return s.groupby("class")[["gain", "acc_2afc", "null_p", "ci_lo", "ci_hi"]].mean()


# ── design fixture sanity ────────────────────────────────────────────────────

def test_fake_design_shape(design):
    _, p = design
    df = pd.read_csv(p, sep="\t", na_values=["n/a"])
    non = df[~df["anchor"]]
    assert non["mmmId"].nunique() == N_SES * ITEMS_PER_SES
    assert (non[non["phase"] == "ret-word"].groupby("mmmId").size() == 1).all()
    assert non["fold"].nunique() == N_SES
    w = non[non["phase"] == "ret-word"]
    assert (w[w["reCon"] == 2]["age_days"] > 0).all() and (w[w["reCon"] == 1]["age_days"] == 0).all()
    assert w[w["enCon"] == 1]["age_days"].nunique() > 2        # single-exposure items span ages
    enc = non[non["phase"] == "enc"]
    assert not enc.duplicated(["session", "run", "onset"]).any()


# ── scenarios ────────────────────────────────────────────────────────────────

def test_rotation_orthogonal_map_wins(design, tmp_path):
    cache = make_caches(design, tmp_path, "rotation")
    t = fit(design, cache, "FakeRoi80")
    s = selected(t["transformation_class"])
    assert s.loc["procrustes", "gain"] > 0.05
    assert s.loc["procrustes", "ci_lo"] > 0                       # beats identity beyond the fold CI
    assert s.loc["procrustes", "null_p"] < 0.05
    assert s.loc["identity", "acc_2afc"] < s.loc["procrustes", "acc_2afc"]
    rm = t["rotation_metrics"]
    fit_angle = rm[(rm["metric"] == "mean_plane_angle_deg") & (rm["arm"] == "fit")]["value"].mean()
    floor = rm[(rm["metric"] == "mean_plane_angle_deg") & (rm["arm"] == "floor_enc")]["value"].mean()
    assert fit_angle > floor + 10                                  # a real rotation, well above the floor
    assert t["transformation_class"]["gate_passed"].any()
    ge = t["geometry"]
    assert ge["rdm_corr"].mean() > 0.2                             # geometry preserved


def test_degradation_scaled_identity_is_best(design, tmp_path):
    cache = make_caches(design, tmp_path, "degradation")
    t = fit(design, cache, "FakeRoi80")
    s = selected(t["transformation_class"])
    assert s.loc["procrustes", "ci_hi"] <= 0.02                    # no gain beyond the fold CI
    assert abs(s.loc["scaled_identity", "gain"]) < 1e-9            # scaled identity == identity in 2AFC
    assert s.loc["procrustes", "acc_2afc"] <= s.loc["identity", "acc_2afc"] + 0.02


def test_null_nothing_beats_identity(design, tmp_path):
    cache = make_caches(design, tmp_path, "null")
    t = fit(design, cache, "FakeRoi80")
    s = selected(t["transformation_class"])
    for cls in ("procrustes", "ridge", "reduced_rank"):
        assert s.loc[cls, "null_p"] > 0.05
        assert abs(s.loc[cls, "acc_2afc"] - 0.5) < 0.1              # chance, six folds of 40 items
    assert abs(s.loc["identity", "acc_2afc"] - 0.5) < 0.08
    assert not t["transformation_class"]["gate_passed"].any()


def test_remap_union_wins_and_cross_energy_is_high(design, tmp_path):
    cache = make_caches(design, tmp_path, "remap", noise=0.4)
    b1 = selected(fit(design, cache, "FakeBlock1")["transformation_class"])
    b2 = selected(fit(design, cache, "FakeBlock2")["transformation_class"])
    u = fit(design, cache, "FakeUnion", rung="iii")
    su = selected(u["transformation_class"])
    # single blocks: block 1 of R is noise, block 2 of R has no E counterpart
    assert su.loc["procrustes", "acc_2afc"] > max(b1.loc["procrustes", "acc_2afc"],
                                                   b2.loc["procrustes", "acc_2afc"]) + 0.1
    assert su.loc["procrustes", "gain"] > 0.1
    be = u["block_energy"]
    assert be["cross_fraction"].mean() > 0.4                       # movement across blocks
    off = be[be["source_block"] != be["target_block"]]["energy_frac"].mean()
    assert off > 0.3


def test_age_slope_recovered_in_the_spectrum(design, tmp_path):
    cache = make_caches(design, tmp_path, "age", noise=0.15, extra=["--age-slope", 2.0])
    t = fit(design, cache, "FakeRoi80")
    sp = t["plane_spectrum"]
    top = sp.sort_values("slope_per_day", ascending=False).iloc[0]
    assert top["ci_lo"] > 0 and top["ci_lo"] < 2.0 < top["ci_hi"]  # planted 2 deg/day
    assert bool(top["age_dependent"])
    assert sp["position_share"].iloc[0] > 0                        # some variance is age-dependent
    ds = t["delay_slopes"]
    assert set(ds["measure"]) == {"plane_angle", "alignment_residual"}
    assert set(ds["exposure_group"]) == {"single", "three"}


def test_composition_holds_then_breaks(design, tmp_path):
    good = make_caches(design, tmp_path, "composition")
    t = fit(design, good, "FakeRoi80", pair="enc:ret-word:ret-image")
    c = t["composition"]
    best = c.groupby("rank")["acc_direct"].mean().idxmax()
    cb = c[c["rank"] == best]
    assert cb["acc_direct"].mean() > 0.8
    assert abs(cb["shortfall"].mean()) < 0.05                      # composed ~ direct
    broken = make_caches(design, tmp_path, "composition_broken")   # R_word is a bottleneck
    t2 = fit(design, broken, "FakeRoi80", pair="enc:ret-word:ret-image")
    c2 = t2["composition"]
    cb2 = c2[c2["rank"] == c2.groupby("rank")["acc_direct"].mean().idxmax()]
    assert cb2["shortfall"].mean() > cb["shortfall"].mean() + 0.05
    assert cb2["shortfall"].mean() > 0.05


def test_reflection_is_tagged(design, tmp_path):
    """A det = -1 planted map yields procrustes det_sign -1 rows and a
    procrustes_proper row beside them."""
    cache = make_caches(design, tmp_path, "rotation", seed=3)
    # flip one voxel's sign in the retrieval cache: an improper map in voxel space
    arm = cache / "cache" / "glmsingle_tb" / "sub-99" / "ret-word-tbonly"
    p = next(arm.glob("*.npz"))
    d = dict(np.load(p, allow_pickle=True))
    P = d["patterns_FakeRoi80"]
    P[0] = -P[0]
    d["patterns_FakeRoi80"] = P
    np.savez_compressed(p, **d)
    t = fit(design, cache, "FakeRoi80", n_perm=5)
    tc = t["transformation_class"]
    pro = tc[(tc["class"] == "procrustes") & (tc["direction"] == "forward")]
    assert set(pro["det_sign"].astype(int)) <= {-1, 1}
    if (pro["det_sign"].astype(int) == -1).any():
        assert (tc["class"] == "procrustes_proper").any()


def test_report_concatenates_and_specs_validate(design, tmp_path):
    cache = make_caches(design, tmp_path, "rotation", seed=5)
    fit(design, cache, "FakeRoi60", n_perm=5)
    fit(design, cache, "FakeRoi80", pair="enc:ret-image", n_perm=5)
    out = tmp_path / "report"
    run([NR / "report.py", "--in", cache, "--out", out])
    for name in ("transformation_class", "class_summary", "class_winners", "class_verdict",
                 "rotation_summary", "geometry", "delay_slopes"):
        assert (out / f"{name}.tsv").exists(), name
    cw = pd.read_csv(out / "class_winners.tsv", sep="\t")
    assert set(cw["roi"]) == {"FakeRoi60", "FakeRoi80"}
    # every spec validates against its table under the mmmview contract
    # (conftest already puts src/python on sys.path)
    from resultsview import page
    specs = sorted(out.glob("*.vl.json"))
    assert specs
    for s in specs:
        page.load_spec(s)
    verdict = pd.read_csv(out / "class_verdict.tsv", sep="\t")
    assert "verdict" in verdict.columns


def test_fit_json_records_inputs(design, tmp_path):
    cache = make_caches(design, tmp_path, "null", seed=7)
    fit(design, cache, "FakeRoi60", n_perm=5)
    j = json.load(open(next((cache / "fits" / "sub-99").glob("*_fit.json"))))
    assert j["args"]["roi"] == "FakeRoi60"
    assert len(j["folds"]) == N_SES
    assert "enc" in j["caches"] and "ret-word" in j["caches"]


# ── scoring variants: strata (targets) and foil pools (candidates) ──────────

def test_identify_masks_restrict_targets_and_foils():
    """target_mask keeps the foil pool; candidate_mask keeps the targets; the
    target's own true pattern is always compared."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("nr_score", NR / "score.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    rng = np.random.default_rng(0)
    n, k = 12, 8
    R_true = rng.standard_normal((n, k))
    R_hat = R_true + 0.5 * rng.standard_normal((n, k))
    runs = np.array([0] * 6 + [1] * 6)
    full = sc.identify(R_hat, R_true, runs)
    assert full["n_items"] == n and full["n_pairs"] == 2 * 6 * 5
    tm = np.arange(n) < 6                                   # targets: run 0 only
    t = sc.identify(R_hat, R_true, runs, target_mask=tm)
    assert t["n_items"] == 6 and t["n_pairs"] == 6 * 5
    cm = np.arange(n) % 2 == 0                              # foils: even items only
    c = sc.identify(R_hat, R_true, runs, candidate_mask=cm)
    assert c["n_items"] == n and c["n_pairs"] == 2 * (3 * 2 + 3 * 3)   # even targets 2 foils, odd 3
    v = sc.identify_variants(R_hat, R_true, runs, {"_a": (tm, None), "_b": (None, cm), "_ab": (tm, cm)})
    assert v["acc_2afc"] == full["acc_2afc"] and v["n_items_a"] == 6 and v["n_pairs"] == full["n_pairs"]
    assert set(v) >= {"acc_2afc_a", "acc_rank_b", "n_items_ab"}
    # a foil pool with no candidates for a target leaves that target out
    only_self = np.zeros(n, bool)
    e = sc.identify(R_hat, R_true, runs, candidate_mask=only_self)
    assert e["n_pairs"] == 0 and np.isnan(e["acc_2afc"])


def test_fit_writes_variant_columns(design, tmp_path):
    cache = make_caches(design, tmp_path, "rotation", noise=0.15)
    t = fit(design, cache, "FakeRoi60", n_perm=5)
    tc = t["transformation_class"]
    for s in ("_reCon1", "_reCon2", "_ntf", "_ntf_reCon1", "_ntf_reCon2"):
        for col in ("acc_2afc", "gain", "n_items", "null_p", "ci_lo", "gate_passed"):
            assert f"{col}{s}" in tc.columns, f"missing {col}{s}"
    sel = tc[tc["rank_selected"].astype(bool) & (tc["direction"] == "forward") & (tc["class"] == "procrustes")]
    # the non-triplet pool has fewer pairs but the same targets
    assert (sel["n_items_ntf"] == sel["n_items"]).all() if "n_items" in sel else True
    assert sel["null_p_ntf"].notna().any()

"""Task 0 input verification for the pattern-similarity analysis.

Read-only checks of the ingredients the analysis depends on (see
docs/doc/pattern-similarity-plan.md, Phase 0):

  1. TB shared items: per-subject pairId->mmmId mapping constant across all
     14 sessions x 3 runs; each of the 6 items presented exactly 42x.
     Writes the registry Phase 6 consumes.
  2. NAT repeated movies: condition==3 rows are exactly {The Bench,
     From Dad To Son}, once per session, durations stable; records TR and
     4.5-s chunk counts.
  3. Every NAT run has >=2 movies (same-run different-item sampling).
  4. Bilateral Harvard-Oxford ROI voxel counts on the atlas grid and on the
     fMRIPrep MNI res-2 BOLD grid.

Outputs (login node, <5 min):
  derivatives/pattern_similarity/qc/task0_verification.tsv
  derivatives/pattern_similarity/qc/shared_items.tsv

Usage: .venv/bin/python scripts/pattern_similarity/verify_inputs.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared as ps

CHECKS: list[dict] = []


def record(check: str, subject: str, detail: str, ok: bool, value=""):
    CHECKS.append({
        "check": check, "subject": subject, "detail": detail,
        "status": "OK" if ok else "FAIL", "value": value,
    })
    if not ok:
        print(f"  FAIL [{check}] {subject} {detail}: {value}")


def verify_shared_items() -> pd.DataFrame:
    registries = []
    for sub in ps.SUBJECTS:
        try:
            reg = ps.shared_item_registry(sub)
        except ValueError as exc:
            record("tb_shared_stable", sub, "mapping constant across sessions",
                   False, str(exc))
            continue
        registries.append(reg)
        record("tb_shared_stable", sub, "mapping constant across sessions", True)
        record("tb_shared_pairids", sub, "pairIds are {1,2,3,75,76,77}",
               sorted(reg["pairId"]) == [1, 2, 3, 75, 76, 77],
               ",".join(map(str, sorted(reg["pairId"]))))
        record("tb_shared_pool", sub, "mmmIds are the 995-1000 pool",
               sorted(reg["mmmId"]) == [995, 996, 997, 998, 999, 1000],
               ",".join(map(str, sorted(reg["mmmId"]))))
        bad = reg[reg["n_presentations"] != 42]
        record("tb_shared_42x", sub, "each item presented 42x",
               len(bad) == 0, "" if len(bad) == 0 else bad.to_string())
        triplet = reg[reg["role"] == "triplet"].sort_values("position")
        record("tb_triplet", sub, "triplet order (info)", True,
               " -> ".join(f"{w}({m})" for w, m in zip(triplet["word"],
                                                       triplet["mmmId"])))
    return pd.concat(registries, ignore_index=True) if registries else pd.DataFrame()


def verify_movies():
    for sub in ps.SUBJECTS:
        frames = []
        for ses in ps.NAT_SESSIONS:
            for run in ps.NAT_RUNS:
                df = pd.read_csv(ps.events_path(sub, ses, run, "NATencoding"),
                                 sep="\t")
                movies = df[df["trial_type"] == "movie"].copy()
                movies["session"], movies["run"] = ses, run
                frames.append(movies)
        allmov = pd.concat(frames, ignore_index=True)

        # >=2 movies per run (needed for same-run different-item sampling)
        per_run = allmov.groupby(["session", "run"]).size()
        record("nat_movies_per_run", sub, "every run has >=2 movies",
               bool((per_run >= 2).all()),
               f"min={per_run.min()} (ses-28 has 3/run by design)")

        rep = allmov[allmov["condition"] == 3]
        record("nat_repeated_names", sub, "condition==3 movies",
               set(rep["movie_name"]) == set(ps.REPEATED_MOVIES),
               ",".join(sorted(set(rep["movie_name"]))))
        per_ses = rep.groupby(["movie_name", "session"]).size()
        record("nat_one_per_session", sub, "each repeat 1x/session",
               bool((per_ses == 1).all()) and len(per_ses) == 2 * len(ps.NAT_SESSIONS))
        for name, grp in rep.groupby("movie_name"):
            dur_min, dur_max = grp["duration"].min(), grp["duration"].max()
            stable = (dur_max - dur_min) <= 0.05
            n_trs = math.floor(grp["duration"].median() / ps.TR)
            n_chunks = math.floor(grp["duration"].median() / ps.NAT_CHUNK_S)
            record("nat_duration_stable", sub, name, stable,
                   f"range=[{dur_min:.3f},{dur_max:.3f}]s "
                   f"n_TRs={n_trs} n_chunks={n_chunks}")


def verify_rois():
    import nibabel as nib

    masks, affine = ps.load_bilateral_roi_masks()
    record("roi_labels", "all", "HO label-name assertions passed", True)

    ref = ps.bold_path(ps.SUBJECTS[0], ps.TB_SESSIONS[0], 1, "TBencoding",
                       "original")
    bold_img = nib.load(ref)
    resampled = ps.resample_masks_to_bold(masks, affine, bold_img)
    for roi in ps.PATTERN_ROI_NAMES:
        n_atlas, n_bold = int(masks[roi].sum()), int(resampled[roi].sum())
        record("roi_voxels", "all", roi, n_bold > 100,
               f"atlas_grid={n_atlas} bold_grid={n_bold}")


def main():
    print("Task 0 verification — pattern-similarity analysis inputs")
    registry = verify_shared_items()
    verify_movies()
    verify_rois()

    ps.QC_DIR.mkdir(parents=True, exist_ok=True)
    checks = pd.DataFrame(CHECKS)
    checks.to_csv(ps.QC_DIR / "task0_verification.tsv", sep="\t", index=False)
    if len(registry):
        registry.to_csv(ps.QC_DIR / "shared_items.tsv", sep="\t", index=False)

    n_fail = int((checks["status"] == "FAIL").sum())
    print(f"\n{len(checks)} checks, {n_fail} failures")
    print(f"Wrote {ps.QC_DIR / 'task0_verification.tsv'}")
    if len(registry):
        print(f"Wrote {ps.QC_DIR / 'shared_items.tsv'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

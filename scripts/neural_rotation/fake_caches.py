#!/usr/bin/env python3
"""Synthetic ROI caches in the ladder cache format, for the end-to-end run.

Writes one npz per arm (enc, ret-word-tbonly, ret-image-tbonly) exactly as
extract_roi_betas.py --roi-set ladder does, using the REAL design table of a
subject for the trial columns so folds, runs, exposures and item ages are
real, and planting a known relation between the phases per scenario:

  rotation     R = rotate(E) + noise; every latent plane rotated by a large
               known angle, so identity fails and an orthogonal map wins
  degradation  R = 0.3 E + noise (scaled identity is the truth)
  remap        two blocks: R block 2 = rotate(E block 1), R block 1 = noise
               (single-block fits degrade, the union fit wins, cross-block
               energy is high)
  null         R independent of E
  age          rotation whose angle in one plane grows 0.5 deg/day with item
               age (the delay test)
  composition  three phases with Q_word Q_wi = Q_direct planted
  composition_broken
               the word phase is a quarter-rank bottleneck, so the composed
               path underperforms the direct map

Three fake ROIs (sizes given by --sizes) named to look like ladder rows,
plus a two-block union ROI. Item signal S ~ N(0, I) in a d-dim latent, E =
S A + noise with A orthonormal rows so a latent rotation is an orthogonal
map in voxel space.

Usage:
    python fake_caches.py --subject sub-## --design <trials.tsv> --out <cache_root> \\
        --scenario rotation --seed 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = {"enc": "enc", "ret-word": "ret-word-tbonly", "ret-image": "ret-image-tbonly"}
ROI_SET = "ladder"
TRIAL_COLS = ["session", "run", "task", "subgroup", "mmmId", "condition_id", "col_index",
              "onset", "duration", "word", "pairId", "sharedId", "enCon", "reCon", "resp", "resp_RT"]


def rotation_matrix(d: int, angles_deg, rng) -> np.ndarray:
    """Block-diagonal rotation in a random orthonormal frame of R^d: plane j
    gets angles_deg[j]; leftover dims are fixed."""
    Q = np.eye(d)
    for j, deg in enumerate(angles_deg):
        i0, i1 = 2 * j, 2 * j + 1
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        Q[i0, i0], Q[i0, i1], Q[i1, i0], Q[i1, i1] = c, -s, s, c
    U = np.linalg.qr(rng.standard_normal((d, d)))[0]
    return U @ Q @ U.T


def make_roi(V: int, d: int, items: np.ndarray, design: pd.DataFrame, scenario: str,
             rng, noise: float, blocks: np.ndarray | None = None,
             age_slope: float = 0.5) -> dict:
    """Latent S per item, phase patterns per trial. Returns {phase: (N, V)}."""
    n = len(items)
    S = rng.standard_normal((n, d))
    S2 = rng.standard_normal((n, d))                     # remap: block 2's own content
    pos = {it: i for i, it in enumerate(items)}
    A = np.linalg.qr(rng.standard_normal((V, d)))[0].T           # (d, V) orthonormal rows
    angles = rng.uniform(30, 90, d // 2)
    Q = rotation_matrix(d, angles, rng)
    Q2 = rotation_matrix(d, rng.uniform(30, 90, d // 2), rng)
    age = design[design["phase"] == "ret-word"].drop_duplicates("mmmId").set_index("mmmId")["age_days"]
    latent = {"enc": S}
    if scenario == "rotation":
        latent["ret-word"] = S @ Q
        latent["ret-image"] = S @ Q2
    elif scenario == "degradation":
        latent["ret-word"] = 0.3 * S
        latent["ret-image"] = 0.3 * S
    elif scenario == "null":
        latent["ret-word"] = rng.standard_normal((n, d))
        latent["ret-image"] = rng.standard_normal((n, d))
    elif scenario == "age":
        # plane 0 (latent dims 0,1 in the identity frame) rotates by 20 + age_slope deg/day
        Rw = np.zeros_like(S)
        for i, it in enumerate(items):
            deg = 20 + age_slope * float(age.get(it, 0))
            c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
            v = S[i].copy()
            v[0], v[1] = c * S[i, 0] - s * S[i, 1], s * S[i, 0] + c * S[i, 1]
            Rw[i] = v
        latent["ret-word"] = Rw
        latent["ret-image"] = S @ Q2
    elif scenario == "composition":
        latent["ret-word"] = S @ Q
        latent["ret-image"] = S @ Q @ Q2          # Q_direct = Q Q2 exactly
    elif scenario == "composition_broken":
        # the word phase keeps only half the latent (a bottleneck), so the
        # composed path E -> R_word -> R_image loses what the direct map keeps
        keep = np.r_[np.ones(max(1, d // 4)), np.zeros(d - max(1, d // 4))]
        latent["ret-word"] = (S * keep) @ Q
        latent["ret-image"] = S @ Q2
    elif scenario == "remap":
        latent["ret-word"] = S @ Q                # block handling below
        latent["ret-image"] = S @ Q2
    else:
        raise ValueError(scenario)

    out = {}
    for phase, arm in ARMS.items():
        t = design[design["phase"] == phase]
        P = np.zeros((len(t), V))
        for r, it in enumerate(t["mmmId"].astype(int)):
            i = pos.get(it)
            base = latent[phase][i] @ A if i is not None else rng.standard_normal(V) * 0.5
            P[r] = base + noise * rng.standard_normal(V)
        if scenario == "remap" and blocks is not None:
            # encoding: block 1 carries S, block 2 an INDEPENDENT latent S2;
            # retrieval: block 2 carries S rotated (block 1's content moved
            # across), block 1 is noise. Within either block alone no map
            # relates the phases; only the union sees E block 1 -> R block 2.
            b1, b2 = blocks == 1, blocks == 2
            assert b1.sum() == b2.sum(), "remap needs equal-sized blocks"
            if phase == "enc":
                for r, it in enumerate(t["mmmId"].astype(int)):
                    i = pos.get(it)
                    if i is not None:
                        P[r, b2] = (S2[i] @ A)[b2] + noise * rng.standard_normal(b2.sum())
            else:
                moved = np.zeros((len(t), b1.sum()))
                for r, it in enumerate(t["mmmId"].astype(int)):
                    i = pos.get(it)
                    moved[r] = (latent["enc"][i] @ Q @ A)[b1] if i is not None else 0
                P[:, b2] = moved + noise * rng.standard_normal((len(t), b2.sum()))
                P[:, b1] = noise * rng.standard_normal((len(t), b1.sum()))
        out[phase] = P
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--design", required=True, help="<sub>_desc-trials.tsv from design.py")
    ap.add_argument("--out", required=True, help="cache root (the derivatives/pattern_similarity analogue)")
    ap.add_argument("--scenario", required=True,
                    choices=["rotation", "degradation", "remap", "null", "age", "composition",
                             "composition_broken"])
    ap.add_argument("--sizes", nargs=3, type=int, default=[150, 300, 600])
    ap.add_argument("--latent-dim", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--age-slope", type=float, default=0.5, help="age scenario: deg/day")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--beta-type", default="D")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    design = pd.read_csv(args.design, sep="\t", na_values=["n/a"])
    design["run"] = design["run"].astype(int)
    items = np.array(sorted(design.loc[~design["anchor"].astype(bool), "mmmId"].astype(int).unique()))
    grid = np.array([97, 115, 97])
    affine = np.array([[2, 0, 0, -96.5], [0, 2, 0, -132.5], [0, 0, 2, -78.5], [0, 0, 0, 1.0]])

    rois = {f"FakeRoi{V}": (V, None) for V in args.sizes}
    Vb = args.sizes[0]                                   # two equal blocks
    blocks_union = np.r_[np.ones(Vb, int), 2 * np.ones(Vb, int)]
    rois["FakeUnion"] = (2 * Vb, blocks_union)
    patterns = {}
    for name, (V, blk) in rois.items():
        d = min(args.latent_dim, V // 4)
        patterns[name] = make_roi(V, d, items, design, args.scenario, rng, args.noise, blk,
                                  age_slope=args.age_slope)
    # the union's two blocks as single ROIs, so "single-block fits degrade,
    # union fit wins" is testable on the same voxels
    for b in (1, 2):
        cols = blocks_union == b
        rois[f"FakeBlock{b}"] = (int(cols.sum()), None)
        patterns[f"FakeBlock{b}"] = {ph: P[:, cols] for ph, P in patterns["FakeUnion"].items()}

    for phase, arm in ARMS.items():
        t = design[design["phase"] == phase].reset_index(drop=True)
        out = {"subject": args.subject, "arm": arm, "beta_type": f"TYPE{args.beta_type}",
               "roi_set": ROI_SET, "roi_names": np.array(list(rois)),
               "roi_rungs": np.array(["i", "i", "i", "iii", "i", "i"]),
               "grid_shape": grid, "affine": affine, "source_dir": f"synthetic:{args.scenario}:{args.seed}",
               "trial_index": np.arange(len(t)), "run": t["run"].to_numpy(),
               "session": t["session"].astype(str).to_numpy(),
               "task": np.array(["TBencoding" if phase == "enc" else "TBretrieval"] * len(t)),
               "subgroup": np.array([phase if phase == "enc" else phase[4:]] * len(t)),
               "mmmId": t["mmmId"].astype(int).astype(str).to_numpy(),
               "condition_id": t["mmmId"].astype(int).astype(str).to_numpy(),
               "col_index": np.arange(len(t)), "onset": t["onset"].to_numpy(float),
               "duration": t["duration"].to_numpy(float), "word": t["word"].astype(str).to_numpy(),
               "pairId": t["pairId"].to_numpy(float), "sharedId": t["sharedId"].to_numpy(float),
               "enCon": t["enCon"].to_numpy(float), "reCon": t["reCon"].to_numpy(float),
               "resp": t["resp"].to_numpy(float), "resp_RT": t["resp_RT"].to_numpy(float)}
        offset = 0
        for name, (V, blk) in rois.items():
            P = patterns[name][phase].astype(np.float32).T             # (V, N)
            out[f"patterns_{name}"] = P
            out[f"voxidx_{name}"] = np.arange(offset, offset + V)
            out[f"meanvol_{name}"] = np.full(V, 1000.0, np.float32)
            out[f"R2_{name}"] = np.full(V, 10.0, np.float32)
            if blk is not None:
                out[f"blocks_{name}"] = blk.astype(np.int16)
                out[f"blocknames_{name}"] = np.array(["FakeBlock1", "FakeBlock2"])
            offset += V
        d = Path(args.out) / "cache" / "glmsingle_tb" / args.subject / arm
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{args.subject}_arm-{arm}_set-{ROI_SET}_desc-type{args.beta_type.lower()}_roipatterns.npz"
        np.savez_compressed(p, **out)
        print(f"wrote {p}")
    with open(Path(args.out) / "cache" / "glmsingle_tb" / args.subject / "synthetic.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()

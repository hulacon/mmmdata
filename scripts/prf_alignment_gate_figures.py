#!/usr/bin/env python3
"""Figures for the pRF alignment gate: the offset curve and the split-half map.

Reads what `prf_alignment_gate.py` wrote (its JSON plus `r_split_half.npy`,
`r_diff_setnum.npy`, `mask.npy`) and renders two panels. Separate from the gate
itself so the figures can be regenerated without re-reading the BOLD volumes.

Usage:
    python prf_alignment_gate_figures.py --gate-dir DIR --mask-ref NIFTI \
        --out-dir DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--mask-ref", required=True,
                    help="any NIfTI on the same grid, for the affine")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import nibabel as nb
    gate_dir = Path(args.gate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    js = next(gate_dir.glob("*_gate.json"))
    res = json.loads(js.read_text())
    shifts = np.array(res["shifts"])

    # ---- panel 1: the offset curve -----------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name in sorted(res["runs"]):
        r = res["runs"][name]
        ax.plot(shifts, r["curve"], marker="o", ms=3.5, lw=1.4,
                label=f"{name} (setnum {r['setnum']})")
    ax.axvline(0, color="k", lw=1, ls="--", alpha=.6)
    ax.set_xlabel("stimulus shifted relative to the volumes (TR)")
    ax.set_ylabel("99.9th-percentile voxel correlation")
    ax.set_title(f"pRF alignment gate — sub-{res['subject']} ses-{res['session']}\n"
                 "peak at 0 TR is the pass condition", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    p1 = out_dir / "alignment-offset-curve.png"
    fig.savefig(p1, dpi=120)
    print("wrote", p1)

    # ---- panel 2: split-half map -------------------------------------------
    sh_path = gate_dir / "r_split_half.npy"
    if not sh_path.exists():
        print("no split-half map to draw")
        return
    mask = np.load(gate_dir / "mask.npy")
    ref = nb.load(args.mask_ref)
    vol = np.zeros(mask.shape, dtype=float)
    vol[mask] = np.load(sh_path)

    # slices chosen at the peak of the map so the figure shows the finding
    pk = np.unravel_index(np.argmax(vol), vol.shape)
    fig2, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    vmax = float(np.percentile(vol[mask], 99.9))
    for ax, (axis, idx, name) in zip(axes, [(0, pk[0], "sagittal"),
                                            (1, pk[1], "coronal"),
                                            (2, pk[2], "axial")]):
        sl = np.take(vol, idx, axis=axis)
        ax.imshow(np.rot90(sl), cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"{name} @ voxel {idx}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    com = res["split_half"]["cluster"]["com_mni"] if res.get("split_half", {}).get("cluster") else None
    sub = (f"split-half r, run-{res['split_half']['runs'][0]:02d} vs "
           f"run-{res['split_half']['runs'][1]:02d} (identical apertures); "
           f"peak r = {vol.max():.2f}")
    if com:
        sub += f"; cluster COM MNI ({com[0]:+.0f}, {com[1]:+.0f}, {com[2]:+.0f})"
    fig2.suptitle(sub, fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    p2 = out_dir / "alignment-split-half.png"
    fig2.savefig(p2, dpi=120)
    print("wrote", p2)


if __name__ == "__main__":
    main()

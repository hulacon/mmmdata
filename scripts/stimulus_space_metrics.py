#!/usr/bin/env python
"""Projection-free stimulus-space metrics: nearest-neighbor similarity violins.

The companion diagnostic to plot_stimulus_space.py. 2-D projections of the
joint embedding spaces disagree by construction (MDS keeps global distances,
UMAP keeps neighbor graphs), so sampling claims should rest on statistics
computed in the ambient space. For every item in a source set this script
finds its nearest neighbor in a target set (cosine similarity, unit-norm
embeddings) and plots the seven distributions that describe how the three
MMMData stimulus sets relate:

    within-set   image -> nearest other image
                 word  -> nearest other word
                 movie frame -> nearest other frame, same movie
    across sets  movie frame -> nearest frame, different movie
                 movie frame -> nearest image
                 movie frame -> nearest word
                 image -> nearest word

Movies enter as their 0.5 s-grid frames (vision arm); words are the text arm
of the same joint space. Outputs land next to the projection figures:
nn_similarity_violins.png and nn_similarity_distributions.csv (long format).

Usage:
    python stimulus_space_metrics.py --model ebind
Run with the stimfeat env:
    /gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — use config if importable, else fall back to well-known path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
try:
    from core.config import load_config

    _config = load_config(config_dir=_REPO_ROOT / "config")
    BIDS_ROOT = Path(_config["paths"]["bids_project_dir"])
except Exception:
    BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")

FEAT_ROOT = BIDS_ROOT / "derivatives" / "stimuli_features"

# CVD-validated set palette, shared with plot_stimulus_space.py
COLORS = {"images": "#1f77b4", "words": "#1b9e77", "movies": "#d95f02"}


def _load(path: Path, prefix: str) -> np.ndarray:
    df = pd.read_csv(path)
    cols = sorted(c for c in df.columns
                  if c.startswith(prefix) and c[len(prefix):].isdigit())
    X = df[cols].to_numpy(np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def nn_sim(A: np.ndarray, B: np.ndarray, mask=None, chunk: int = 2000) -> np.ndarray:
    """For each row of A, max cosine similarity over rows of B.

    mask(i0, i1) may return a boolean (rows i0:i1 of A, all of B) array of
    pairs to EXCLUDE (self pairs, same-movie pairs).
    """
    out = np.empty(len(A))
    for i0 in range(0, len(A), chunk):
        i1 = min(i0 + chunk, len(A))
        S = A[i0:i1] @ B.T
        if mask is not None:
            S[mask(i0, i1)] = -np.inf
        out[i0:i1] = S.max(axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["ebind", "clip"], default="ebind")
    ap.add_argument("--space", choices=["vision", "text"], default="vision",
                    help="vision: image/frame embeddings. text: BLIP-caption "
                         "embeddings through the text arm (stimfeat_captions "
                         "outputs) — one modality, no gap.")
    ap.add_argument("--out-dir", type=Path,
                    default=FEAT_ROOT / "figures" / "stimulus_space")
    args = ap.parse_args()
    m = args.model

    wrd = _load(FEAT_ROOT / "twp1000" / f"{m}_text_chunks.csv", f"{m}_text_")
    if args.space == "vision":
        vis_file, vis_prefix = f"{m}.csv", f"{m}_"
        img_noun, frame_noun = "image", "movie frame"
        stem, title_arm = m, f"{m.upper()}"
    else:
        vis_file, vis_prefix = f"caption_{m}_text_chunks.csv", f"{m}_text_"
        img_noun, frame_noun = "image caption", "frame caption"
        stem, title_arm = f"{m}_captions", f"{m.upper()} text arm (BLIP captions)"
    img = _load(FEAT_ROOT / "shared1000" / vis_file, vis_prefix)
    movie_dirs = sorted(p for p in (FEAT_ROOT / "movies").iterdir()
                        if (p / vis_file).exists())
    Xs = [_load(p / vis_file, vis_prefix) for p in movie_dirs]
    mov = np.vstack(Xs)
    movie_of = np.repeat(np.arange(len(Xs)), [len(X) for X in Xs])
    print(f"{stem}: {len(img)} images, {len(wrd)} words, "
          f"{len(Xs)} movies ({len(mov)} frames)")

    eye = lambda i0, i1: np.eye(len(img), dtype=bool)[i0:i1]
    same_movie = lambda i0, i1: movie_of[i0:i1, None] == movie_of[None, :]
    diff_movie = lambda i0, i1: movie_of[i0:i1, None] != movie_of[None, :]
    # within-movie NN excludes self only, so temporal neighbors dominate —
    # that redundancy is part of what the figure is meant to show
    self_only = lambda i0, i1: (np.arange(i0, i1)[:, None]
                                == np.arange(len(mov))[None, :])

    relations = [
        # (label, target set for color, values)
        (f"{img_noun} → nearest {img_noun}",
         "images", nn_sim(img, img, eye)),
        ("word → nearest word",
         "words", nn_sim(wrd, wrd, lambda i0, i1: np.eye(len(wrd), dtype=bool)[i0:i1])),
        (f"{frame_noun} → same movie",
         "movies", nn_sim(mov, mov, lambda i0, i1: self_only(i0, i1) | diff_movie(i0, i1))),
        (f"{frame_noun} → different movie",
         "movies", nn_sim(mov, mov, same_movie)),
        (f"{frame_noun} → nearest {img_noun}",
         "images", nn_sim(mov, img)),
        (f"{frame_noun} → nearest word",
         "words", nn_sim(mov, wrd)),
        (f"{img_noun} → nearest word",
         "words", nn_sim(img, wrd)),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long = pd.concat(
        [pd.DataFrame({"relation": label, "target_set": tgt, "nn_cosine_sim": v})
         for label, tgt, v in relations], ignore_index=True)
    csv_path = args.out_dir / f"{stem}_nn_similarity_distributions.csv"
    long.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
    fig.subplots_adjust(left=0.30)
    n = len(relations)
    # top-to-bottom in list order, with a group gap after the within-set block
    ys = [n - i + (0.6 if i < 3 else 0) for i in range(n)]
    for (label, tgt, v), y in zip(relations, ys):
        parts = ax.violinplot([v], positions=[y], vert=False, widths=0.82,
                              showextrema=False)
        body = parts["bodies"][0]
        body.set_facecolor(COLORS[tgt])
        body.set_alpha(0.75)
        body.set_edgecolor("none")
        med, p10, p90 = np.percentile(v, [50, 10, 90])
        ax.plot([p10, p90], [y, y], color="#1C2733", lw=1, solid_capstyle="butt")
        ax.plot(med, y, "o", color="#1C2733", ms=4.5)
        ax.annotate(f"{med:.2f}", (med, y + 0.30), ha="center", fontsize=8.5,
                    color="#1C2733")
        # row labels live in the figure margin so no violin can collide
        ax.annotate(label, (-0.015, y), xycoords=("axes fraction", "data"),
                    ha="right", va="center", fontsize=10, color="#1C2733",
                    annotation_clip=False)
        ax.annotate(f"n={len(v):,}", (-0.015, y - 0.28),
                    xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=7.5, color="#5A6B7A", annotation_clip=False)
    gap_y = ys[2] - (ys[2] - ys[3]) / 2
    ax.axhline(gap_y, color="#DCE3E9", lw=1)
    ax.text(0.995, ys[1], "within set", ha="right", va="center",
            fontsize=8.5, color="#5A6B7A", style="italic",
            transform=ax.get_yaxis_transform())
    ax.text(0.995, ys[4], "across sets", ha="right", va="center",
            fontsize=8.5, color="#5A6B7A", style="italic",
            transform=ax.get_yaxis_transform())

    ax.set_xlim(0, 1)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.75)
    ax.set_yticks([])
    ax.set_xlabel("cosine similarity to nearest neighbor (ambient space, unit-norm)")
    ax.xaxis.grid(True, color="#DCE3E9", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{title_arm} nearest-neighbor similarity — "
                 "no 2-D projection involved", fontsize=12)
    fig.tight_layout()
    png_path = args.out_dir / f"{stem}_nn_similarity_violins.png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"  wrote {png_path}")


if __name__ == "__main__":
    main()

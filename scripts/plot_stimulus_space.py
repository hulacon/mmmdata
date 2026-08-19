#!/usr/bin/env python
"""Joint stimulus-space figures: images, words, and movies in one embedding.

Projects all three MMMData stimulus sets into a shared 2-D view from a
cross-modal embedding model (EBind 1024-d or CLIP ViT-B-32 512-d), to
illustrate stimulus-sampling differences between trial-based (TB: shared1000
images, twp1000 words) and naturalistic (NAT: movie clips) paradigms.

Two figure families per model:
  points  — MDS on cosine distances over images + words + movie centroids
            (every stimulus is one point; ~2,060 points)
  clouds  — UMAP over the same plus per-frame movie embeddings (a TB
            stimulus is a point, a NAT stimulus is a cloud/trajectory)

Each family is rendered raw and modality-centered (per-modality mean removed
before projection, to look past the CLIP-family modality gap).

Inputs are the Contract B feature files under derivatives/stimuli_features/
(recipes: stimfeat_ebind.sbatch, stimfeat_ebind_movies.sbatch,
stimfeat_clip.sbatch). Movie clips missing their feature file are skipped
with a warning, so the script degrades gracefully mid-campaign.

Usage:
    python plot_stimulus_space.py --model ebind
    python plot_stimulus_space.py --model clip --frame-stride 4
Run with the stimfeat env (needs umap-learn + kaleido):
    /gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
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
REGISTRY = BIDS_ROOT / "stimuli" / "stimulus_registry"

# Display attributes per stimulus set (TB = trial-based, NAT = naturalistic).
# Palette is CVD-validated (deutan/tritan dE >= 8 on adjacent pairs) — keep the
# blue/teal/orange trio together if you change any one of them.
SETS = {
    "images": {"label": "shared1000 images (TB)", "color": "#1f77b4"},
    "words": {"label": "twp1000 words (TB)", "color": "#1b9e77"},
    "movies": {"label": "movie clips (NAT)", "color": "#d95f02"},
}
CENTROID_COLOR = "#8a3c00"  # darker step of the movie orange


def _embedding_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = sorted(c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit())
    if not cols:
        raise ValueError(f"no '{prefix}NNNN' columns found (have: {list(df.columns)[:8]}...)")
    return df[cols].to_numpy(dtype=np.float64)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load_features(model: str) -> dict:
    """Return {images, words, frames} as {"df": metadata, "X": unit-norm rows}."""
    out = {}

    img = pd.read_csv(FEAT_ROOT / "shared1000" / f"{model}.csv")
    out["images"] = {"df": pd.DataFrame({"stimulus_id": img["stimulus_id"]}),
                     "X": _l2(_embedding_matrix(img, f"{model}_"))}

    wrd = pd.read_csv(FEAT_ROOT / "twp1000" / f"{model}_text_chunks.csv")
    out["words"] = {"df": pd.DataFrame({"stimulus_id": wrd["stimulus_id"]}),
                    "X": _l2(_embedding_matrix(wrd, f"{model}_text_"))}

    with open(REGISTRY / "movies.tsv") as f:
        movie_ids = [r["stimulus_id"] for r in csv.DictReader(f, delimiter="\t")]
    dfs, Xs, missing = [], [], []
    for mid in movie_ids:
        path = FEAT_ROOT / "movies" / mid / f"{model}.csv"
        if not path.exists():
            missing.append(mid)
            continue
        mdf = pd.read_csv(path)
        dfs.append(pd.DataFrame({"stimulus_id": mdf["stimulus_id"], "time": mdf["time"]}))
        Xs.append(_l2(_embedding_matrix(mdf, f"{model}_")))
    if missing:
        warnings.warn(
            f"{model}: {len(missing)}/{len(movie_ids)} movie clips have no "
            f"feature file yet, skipping: {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
        )
    if not dfs:
        raise SystemExit(f"{model}: no movie feature files found under {FEAT_ROOT}/movies/")
    out["frames"] = {"df": pd.concat(dfs, ignore_index=True), "X": np.vstack(Xs)}
    return out


def load_captions(model: str) -> dict | None:
    """Text-arm embeddings of BLIP captions (stimfeat_captions.sbatch), or None.

    Returns {"images": {df, X}, "frames": {df, X}} where frame ids
    "<movie_id>@<time>" are parsed back into stimulus_id + time columns.
    """
    def _read(path: Path) -> dict | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        id_col = "stimulus_id" if "stimulus_id" in df.columns else "caption_id"
        return {"df": pd.DataFrame({"stimulus_id": df[id_col].astype(str)}),
                "X": _l2(_embedding_matrix(df, f"{model}_text_"))}

    images = _read(FEAT_ROOT / "shared1000" / f"caption_{model}_text_chunks.csv")
    dfs, Xs = [], []
    for sub in sorted((FEAT_ROOT / "movies").iterdir()):
        got = _read(sub / f"caption_{model}_text_chunks.csv")
        if got is None:
            continue
        parts = got["df"]["stimulus_id"].str.rsplit("@", n=1)
        dfs.append(pd.DataFrame({"stimulus_id": parts.str[0],
                                 "time": parts.str[1].astype(float)}))
        Xs.append(got["X"])
    if images is None or not dfs:
        return None
    return {"images": images,
            "frames": {"df": pd.concat(dfs, ignore_index=True), "X": np.vstack(Xs)}}


def movie_centroids(frames: dict) -> dict:
    X = frames["X"]
    ids = frames["df"]["stimulus_id"].to_numpy()
    uniq = list(dict.fromkeys(ids))
    C = np.vstack([X[ids == u].mean(axis=0) for u in uniq])
    return {"df": pd.DataFrame({"stimulus_id": uniq}), "X": _l2(C)}


def center_modalities(blocks: list[np.ndarray]) -> list[np.ndarray]:
    return [b - b.mean(axis=0, keepdims=True) for b in blocks]


def project(blocks: list[np.ndarray], method: str, seed: int = 0) -> list[np.ndarray]:
    """Joint 2-D projection; returns one (n_i, 2) array per input block."""
    X = np.vstack(blocks)
    if method == "mds":
        from sklearn.manifold import MDS
        from sklearn.metrics import pairwise_distances

        D = pairwise_distances(X, metric="cosine")
        Y = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
                normalized_stress="auto", n_init=1).fit_transform(D)
    elif method == "umap":
        import umap

        Y = umap.UMAP(n_components=2, metric="cosine",
                      random_state=seed).fit_transform(X)
    elif method == "pca":
        from sklearn.decomposition import PCA

        Y = PCA(n_components=2, random_state=seed).fit_transform(X)
    else:
        raise ValueError(f"unknown method: {method}")
    splits = np.cumsum([len(b) for b in blocks])[:-1]
    return np.split(Y, splits)


def make_figure(traces: list[dict], title: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    for tr in traces:
        fig.add_trace(go.Scattergl(
            x=tr["Y"][:, 0], y=tr["Y"][:, 1], mode="markers",
            name=tr["name"], text=tr.get("text"),
            hovertemplate="%{text}<extra>" + tr["name"] + "</extra>",
            marker={"color": tr["color"], "size": tr.get("size", 6),
                    "opacity": tr.get("opacity", 0.8),
                    "line": {"width": tr.get("line_width", 0), "color": "white"}},
        ))
    fig.update_layout(
        title=title, template="plotly_white", width=1000, height=800,
        xaxis={"title": "dim 1", "showticklabels": False},
        yaxis={"title": "dim 2", "showticklabels": False, "scaleanchor": "x"},
        legend={"orientation": "h", "y": -0.06},
    )
    return fig


def save(fig, traces: list[dict], title: str, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_dir / f"{stem}.html", include_plotlyjs="cdn")
    # PNG via matplotlib: kaleido needs a Chrome binary the cluster lacks
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(figsize=(10, 8), dpi=200)
    for tr in traces:
        ax.scatter(tr["Y"][:, 0], tr["Y"][:, 1], label=tr["name"],
                   s=tr.get("size", 6) ** 2 / 2, c=tr["color"],
                   alpha=tr.get("opacity", 0.8),
                   edgecolors="white" if tr.get("line_width") else "none",
                   linewidths=0.5)
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              frameon=False, markerscale=2)
    f.tight_layout()
    f.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(f)
    print(f"  wrote {out_dir / stem}.html (+.png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["ebind", "clip"], default="ebind")
    ap.add_argument("--frame-stride", type=int, default=2,
                    help="Keep every Nth movie frame in the clouds figure (default 2 = 1 s)")
    ap.add_argument("--out-dir", type=Path,
                    default=FEAT_ROOT / "figures" / "stimulus_space")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feats = load_features(args.model)
    cents = movie_centroids(feats["frames"])
    n_movies = len(cents["df"])
    print(f"{args.model}: {len(feats['images']['df'])} images, "
          f"{len(feats['words']['df'])} words, "
          f"{n_movies} movies ({len(feats['frames']['df'])} frames)")

    stride = feats["frames"]["df"].iloc[:: args.frame_stride]
    Xf = feats["frames"]["X"][:: args.frame_stride]

    for centered in (False, True):
        tag = "centered" if centered else "raw"
        suffix = " (modality means removed)" if centered else ""

        # points: every stimulus is one point (movie = frame centroid), MDS
        blocks = [feats["images"]["X"], feats["words"]["X"], cents["X"]]
        if centered:
            blocks = center_modalities(blocks)
        Yi, Yw, Yc = project(blocks, "mds", args.seed)
        traces = [
            {"Y": Yi, "name": SETS["images"]["label"], "color": SETS["images"]["color"],
             "text": feats["images"]["df"]["stimulus_id"]},
            {"Y": Yw, "name": SETS["words"]["label"], "color": SETS["words"]["color"],
             "text": feats["words"]["df"]["stimulus_id"]},
            {"Y": Yc, "name": SETS["movies"]["label"], "color": SETS["movies"]["color"],
             "text": cents["df"]["stimulus_id"], "size": 12, "line_width": 1},
        ]
        title = f"{args.model.upper()} stimulus space — MDS, one point per stimulus{suffix}"
        save(make_figure(traces, title), traces, title,
             args.out_dir, f"{args.model}_points_mds_{tag}")

        # clouds: TB stimuli are points, movie frames are clouds, UMAP
        blocks = [feats["images"]["X"], feats["words"]["X"], Xf, cents["X"]]
        if centered:
            # frames and centroids share the movie modality mean
            mu = Xf.mean(axis=0, keepdims=True)
            blocks = center_modalities(blocks[:2]) + [Xf - mu, cents["X"] - mu]
        Yi, Yw, Yf, Yc = project(blocks, "umap", args.seed)
        frame_text = stride["stimulus_id"] + " @ " + stride["time"].round(1).astype(str) + "s"
        traces = [
            {"Y": Yi, "name": SETS["images"]["label"], "color": SETS["images"]["color"],
             "text": feats["images"]["df"]["stimulus_id"], "size": 5},
            {"Y": Yw, "name": SETS["words"]["label"], "color": SETS["words"]["color"],
             "text": feats["words"]["df"]["stimulus_id"], "size": 5},
            {"Y": Yf, "name": "movie frames (NAT)", "color": SETS["movies"]["color"],
             "text": frame_text, "size": 3, "opacity": 0.25},
            {"Y": Yc, "name": "movie centroids", "color": CENTROID_COLOR,
             "text": cents["df"]["stimulus_id"], "size": 12, "line_width": 1},
        ]
        title = f"{args.model.upper()} stimulus space — UMAP, TB points vs NAT frame clouds{suffix}"
        save(make_figure(traces, title), traces, title,
             args.out_dir, f"{args.model}_clouds_umap_{tag}")

    # text-arm family: BLIP captions of images and movie frames alongside the
    # words — one modality, so set separation is content, not modality gap
    caps = load_captions(args.model)
    if caps is None:
        print("no caption embeddings yet (stimfeat_captions.sbatch) — "
              "skipping the text-arm figures")
        print(f"done: 4 figures in {args.out_dir}")
        return
    cap_cents = movie_centroids(caps["frames"])
    stride_c = caps["frames"]["df"].iloc[:: args.frame_stride]
    Xc = caps["frames"]["X"][:: args.frame_stride]

    blocks = [feats["words"]["X"], caps["images"]["X"], cap_cents["X"]]
    Yw, Yi, Yc = project(blocks, "mds", args.seed)
    traces = [
        {"Y": Yw, "name": SETS["words"]["label"], "color": SETS["words"]["color"],
         "text": feats["words"]["df"]["stimulus_id"]},
        {"Y": Yi, "name": "image captions (TB)", "color": SETS["images"]["color"],
         "text": caps["images"]["df"]["stimulus_id"]},
        {"Y": Yc, "name": "movie caption centroids (NAT)", "color": SETS["movies"]["color"],
         "text": cap_cents["df"]["stimulus_id"], "size": 12, "line_width": 1},
    ]
    title = (f"{args.model.upper()} text arm — words vs BLIP captions, "
             "MDS, one point per stimulus")
    save(make_figure(traces, title), traces, title,
         args.out_dir, f"{args.model}_captions_mds")

    blocks = [feats["words"]["X"], caps["images"]["X"], Xc, cap_cents["X"]]
    Yw, Yi, Yf, Yc = project(blocks, "umap", args.seed)
    frame_text = stride_c["stimulus_id"] + " @ " + stride_c["time"].round(1).astype(str) + "s"
    traces = [
        {"Y": Yw, "name": SETS["words"]["label"], "color": SETS["words"]["color"],
         "text": feats["words"]["df"]["stimulus_id"], "size": 5},
        {"Y": Yi, "name": "image captions (TB)", "color": SETS["images"]["color"],
         "text": caps["images"]["df"]["stimulus_id"], "size": 5},
        {"Y": Yf, "name": "movie frame captions (NAT)", "color": SETS["movies"]["color"],
         "text": frame_text, "size": 3, "opacity": 0.25},
        {"Y": Yc, "name": "movie caption centroids", "color": CENTROID_COLOR,
         "text": cap_cents["df"]["stimulus_id"], "size": 12, "line_width": 1},
    ]
    title = (f"{args.model.upper()} text arm — words vs BLIP captions, "
             "UMAP, TB points vs NAT frame clouds")
    save(make_figure(traces, title), traces, title,
         args.out_dir, f"{args.model}_captions_umap")

    print(f"done: 6 figures in {args.out_dir}")


if __name__ == "__main__":
    main()

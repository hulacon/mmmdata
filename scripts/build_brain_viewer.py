#!/usr/bin/env python3
"""
build_brain_viewer.py — self-contained NiiVue viewer bundles for brain maps.

One command turns an underlay + overlay spec into a single .html file that
opens over file:// with no network (NiiVue vendored, data base64-inlined) and
exports a .nvd for FreeBrowse from an in-page button. Library:
src/python/neuroimaging/viewer.py (masking + sentinel conventions documented
there). Bundles are findings — they land beside the data (qc/ dirs), never in
a repo; the claude.ai artifact route caps at 16 MB (surfaces ship per hemi).

Generic subcommands take explicit files; overlay/layer specs are
PATH[:key=value[,key=value...]] with keys name,label,colormap,cal_min,
cal_max,opacity,visible,shade,angle_legend:

    python build_brain_viewer.py volume --underlay boldref.nii.gz \\
        --overlay zmap.nii.gz:colormap=viridis,cal_min=3,cal_max=8 \\
        --out view.html
    python build_brain_viewer.py surface --mesh surf/lh.inflated \\
        --layer surf/lh.curv:shade=1,colormap=gray,cal_min=0.3,cal_max=0.8 \\
        --layer maps/lh_stat.shape.gii:cal_min=0,cal_max=5 --out lh.html

The prf subcommand knows the dataset layout (config-driven, --deriv-root to
override off-cluster, e.g. at a checkout of the remote-cache branch):

    python build_brain_viewer.py prf --subject 03 --session 02 --mode volume
    python build_brain_viewer.py prf --subject 03 --session 02 --mode surface \\
        --hemi L --polarity prf
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from neuroimaging import viewer  # noqa: E402

ARTIFACT_CAP_MB = 16.0
R2_FLOOR_DEFAULT = 10.0  # percent, the pilot's supra-threshold cut

# per-parameter display defaults for the prf subcommand; cal_max=None means
# "99th percentile of surviving samples"
PRF_PARAMS = {
    "R2": {"colormap": "viridis", "cal_max": None},
    "angle": {"colormap": "hsv", "cal_min": 0.0, "cal_max": 360.0,
              "angle_legend": True},
    "eccentricity": {"colormap": "turbo", "cal_min": 0.0, "cal_max": None},
    "size": {"colormap": "plasma", "cal_min": 0.0, "cal_max": None},
}
HEMIS = {"L": "lh", "R": "rh"}


def deriv_root(override):
    if override:
        return Path(override)
    try:
        from core.config import load_config
        cfg = load_config(config_dir=_REPO_ROOT / "config")
        return Path(cfg["paths"]["output_dir"])
    except Exception:
        sys.exit("ERROR: could not resolve derivatives root from config; "
                 "pass --deriv-root explicitly")


# ---------------------------------------------------------------------------
# spec parsing
# ---------------------------------------------------------------------------

_BOOL_KEYS = ("visible", "shade", "angle_legend")
_SPEC_KEYS = ("name", "label", "colormap", "cal_min", "cal_max", "opacity",
              "visible", "shade", "angle_legend")


def parse_spec(text):
    """PATH[:key=value[,key=value...]] -> dict. The first ':' whose tail
    contains '=' starts the options (paths with literal ':' must be renamed)."""
    path, opts = text, ""
    if ":" in text:
        head, tail = text.split(":", 1)
        if "=" in tail:
            path, opts = head, tail
    spec = {"path": path}
    if opts:
        for kv in opts.split(","):
            if "=" not in kv:
                sys.exit(f"ERROR: bad spec option {kv!r} in {text!r} "
                         "(expected key=value)")
            k, v = kv.split("=", 1)
            if k not in _SPEC_KEYS:
                sys.exit(f"ERROR: unknown spec key {k!r} in {text!r} "
                         f"(known: {', '.join(_SPEC_KEYS)})")
            if k in _BOOL_KEYS:
                spec[k] = v.lower() in ("1", "true", "yes")
            else:
                spec[k] = v
    if not Path(spec["path"]).exists():
        sys.exit(f"ERROR: no such file: {spec['path']}")
    return spec


def report(out, mode):
    mb = out.stat().st_size / 1e6
    note = ("" if mb <= ARTIFACT_CAP_MB
            else f"  [exceeds the {ARTIFACT_CAP_MB:.0f} MB artifact cap — "
                 "file:// only]")
    print(f"wrote {out}  ({mb:.1f} MB, {mode} mode){note}")


def provenance(sources, extra=""):
    lines = [f"built {datetime.date.today().isoformat()} by "
             "mmmdata/scripts/build_brain_viewer.py (workbench brain-viewer)"]
    if extra:
        lines.append(extra)
    lines += [f"  {Path(s).name}" for s in sources]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_volume(args):
    underlay = parse_spec(args.underlay)
    overlays = [parse_spec(s) for s in args.overlay]
    if args.mask:
        mask = Path(args.mask)
        if not mask.exists():
            sys.exit(f"ERROR: no such mask: {mask}")
        for spec in overlays:
            spec["image"] = viewer.masked_volume(spec["path"], mask,
                                                 args.mask_floor)
    srcs = [underlay["path"]] + [o["path"] for o in overlays]
    out = viewer.build_volume_viewer(
        underlay, overlays, args.out, title=args.title,
        notes=provenance(srcs, f"mask: {args.mask} > {args.mask_floor}"
                         if args.mask else ""))
    report(out, "volume")


def cmd_surface(args):
    layers = [parse_spec(s) for s in args.layer]
    out = viewer.build_surface_viewer(
        args.mesh, layers, args.out, title=args.title,
        notes=provenance([args.mesh] + [l["path"] for l in layers]))
    report(out, "surface")


def _prf_dir(root, subject, session):
    d = root / "prf" / f"sub-{subject}" / f"ses-{session}"
    if not d.is_dir():
        sys.exit(f"ERROR: no pRF session directory at {d}")
    return d


def _p99(values):
    good = values[np.isfinite(values) & (values != viewer.MASK_SENTINEL)]
    if good.size == 0:
        sys.exit("ERROR: a masked pRF map has no surviving samples — "
                 "check --r2-floor against the fit's R2 range")
    return float(np.percentile(good, 99))


def _prf_volume(args, root, out_dir):
    import nibabel as nib
    prf = _prf_dir(root, args.subject, args.session)
    base = f"sub-{args.subject}_ses-{args.session}_task-prf"
    underlay = (root / "fmriprep" / f"sub-{args.subject}"
                / f"ses-{args.session}" / "func"
                / f"{base}_run-01_desc-coreg_boldref.nii.gz")
    if not underlay.exists():
        sys.exit(f"ERROR: no boldref underlay at {underlay}")
    r2 = prf / f"{base}_space-func_desc-R2_{args.polarity}.nii.gz"

    overlays = []
    for param, disp in PRF_PARAMS.items():
        path = prf / f"{base}_space-func_desc-{param}_{args.polarity}.nii.gz"
        if not path.exists():
            sys.exit(f"ERROR: no parameter volume at {path}")
        img = viewer.masked_volume(path, r2, args.r2_floor)
        data = np.asarray(img.dataobj)
        spec = {"image": img, "name": path.name, "label": param, **disp}
        spec.setdefault("cal_min", args.r2_floor if param == "R2" else 0.0)
        if spec["cal_max"] is None:
            spec["cal_max"] = round(_p99(data), 2)
        overlays.append(spec)

    out = out_dir / f"{base}_space-func_desc-viewer_{args.polarity}.html"
    out = viewer.build_volume_viewer(
        {"path": underlay, "label": "boldref"}, overlays, out,
        title=f"pRF {args.polarity} sub-{args.subject} ses-{args.session} (func)",
        notes=provenance([underlay, r2],
                         f"all maps masked to R2 > {args.r2_floor}%"))
    report(out, "volume")


def _prf_surface(args, root, out_dir, hemi):
    prf = _prf_dir(root, args.subject, args.session)
    fs = (root / "fmriprep" / "sourcedata" / "freesurfer"
          / f"sub-{args.subject}" / "surf")
    mesh = fs / f"{HEMIS[hemi]}.{args.surf}"
    curv = fs / f"{HEMIS[hemi]}.curv"
    for p in (mesh, curv):
        if not p.exists():
            sys.exit(f"ERROR: no FreeSurfer file at {p}")
    base = (f"sub-{args.subject}_ses-{args.session}_task-prf"
            f"_space-fsnative_hemi-{hemi}")
    r2 = prf / f"{base}_desc-R2_{args.polarity}.shape.gii"
    if not r2.exists():
        sys.exit(f"ERROR: no projected R2 map at {r2} — run "
                 "project_prf_fsnative.py first")

    layers = [{"path": curv, "label": "curvature", "shade": True,
               "colormap": "gray", "cal_min": 0.3, "cal_max": 0.8,
               "opacity": 0.7}]
    for param, disp in PRF_PARAMS.items():
        path = prf / f"{base}_desc-{param}_{args.polarity}.shape.gii"
        if not path.exists():
            sys.exit(f"ERROR: no projected map at {path}")
        vals = viewer.masked_shape_values(path, r2, args.r2_floor)
        spec = {"values": vals, "name": path.name, "label": param, **disp}
        spec.setdefault("cal_min", args.r2_floor if param == "R2" else 0.0)
        if spec["cal_max"] is None:
            spec["cal_max"] = round(_p99(vals), 2)
        layers.append(spec)

    out = out_dir / f"{base}_desc-viewer_{args.polarity}.html"
    out = viewer.build_surface_viewer(
        mesh, layers, out,
        title=(f"pRF {args.polarity} sub-{args.subject} ses-{args.session} "
               f"hemi-{hemi} ({args.surf})"),
        notes=provenance([mesh, curv, r2],
                         f"vertices masked to R2 > {args.r2_floor}%"))
    report(out, "surface")


def cmd_prf(args):
    root = deriv_root(args.deriv_root)
    out_dir = (Path(args.out_dir) if args.out_dir
               else _prf_dir(root, args.subject, args.session) / "qc")
    if args.mode in ("volume", "both"):
        _prf_volume(args, root, out_dir)
    if args.mode in ("surface", "both"):
        hemis = ("L", "R") if args.hemi == "both" else (args.hemi,)
        for hemi in hemis:
            _prf_surface(args, root, out_dir, hemi)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("volume", help="NIfTI underlay + stat-map overlays")
    v.add_argument("--underlay", required=True, help="spec (see module doc)")
    v.add_argument("--overlay", action="append", default=[], help="spec; repeatable")
    v.add_argument("--mask", help="NIfTI whose value must exceed --mask-floor")
    v.add_argument("--mask-floor", type=float, default=0.0)
    v.add_argument("--title", default="brain viewer")
    v.add_argument("--out", required=True, type=Path)
    v.set_defaults(func=cmd_volume)

    s = sub.add_parser("surface", help="FreeSurfer mesh + per-vertex layers")
    s.add_argument("--mesh", required=True, help="e.g. surf/lh.inflated")
    s.add_argument("--layer", action="append", default=[],
                   help="spec; repeatable; shade=1 marks the always-on "
                        "shading layer (curv/sulc)")
    s.add_argument("--title", default="brain viewer")
    s.add_argument("--out", required=True, type=Path)
    s.set_defaults(func=cmd_surface)

    p = sub.add_parser("prf", help="dataset-aware pRF bundles")
    p.add_argument("--subject", required=True, help="bare label, e.g. 03")
    p.add_argument("--session", required=True, help="bare label, e.g. 02")
    p.add_argument("--polarity", choices=("prf", "negprf"), default="prf")
    p.add_argument("--mode", choices=("volume", "surface", "both"),
                   default="both")
    p.add_argument("--hemi", choices=("L", "R", "both"), default="both")
    p.add_argument("--surf", default="inflated",
                   help="FreeSurfer mesh flavor (inflated, white, pial)")
    p.add_argument("--r2-floor", type=float, default=R2_FLOOR_DEFAULT)
    p.add_argument("--deriv-root", help="override config derivatives root")
    p.add_argument("--out-dir", help="override <prf session>/qc/")
    p.set_defaults(func=cmd_prf)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

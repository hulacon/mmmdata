#!/usr/bin/env python3
"""
mmmview.py — one path into the dataset in, the right interactive view out.

    mmmview PATH [--no-open] [--out-dir DIR] [--underlay NII | --mesh SURF]
                 [--surf inflated|pial|white] [--r2-floor F] [--force]

mmmview is a dispatcher, not a viewer. The viewers exist (the NiiVue bundle
builder in src/python/neuroimaging/viewer.py, the three *2psy dashboards);
this script's only job is the resolution rule from a path's BIDS entities and
Contract B sidecar to (renderer, underlay or mesh, display profile). Three
stages, each a pure function with its own tests (tests/test_mmmview.py):

    classify(path)          -> [Target]   what the file is, from its name
    resolve(target, roots)  -> Plan       underlay/mesh/command + display
    render(plan)            -> out_path   build, then open unless --no-open

Exit codes: 0 built (and opened if it could); 2 could not place the file
(the message names the flag that would place it); 3 the renderer failed
(its stderr passed through).

What it places:

    *.nii.gz with BIDS entities          volume bundle; underlay by space-
    *.shape.gii / *.func.gii with hemi-  surface bundle; fsnative mesh by hemi-
    a directory of either                one bundle per (space, hemi, suffix)
                                         group, every map a toggle layer
    *.csv / *.parquet + *.meta.json      the dashboard of the sidecar's
                                         extractor (viz2psy, aud2psy, word2psy)
    *_events.tsv, *_beh.tsv              not here; the MCP plot tools cover them

Display profiles live in DISPLAY_PROFILES below, keyed on the entity that
names the quantity (desc- for pRF parameters, stat- for GLM maps). They are
recreated in the viewer per family and never written to disk (decided
2026-09-11). A z/t/effect map's negative tail is a second toggle layer
(the template has no two-tailed colormap); the pRF profile extends the
table in build_brain_viewer.py.

Bundles are findings: they land in <sub-##>/qc/ beside the data, never in a
repo. A second call on unchanged inputs reuses the bundle (a sha256 key over
the inputs and profile is written into the provenance note).

Roots come from config/base.toml (+ local.toml): paths.output_dir,
paths.bids_project_dir, paths.stimfeat_env.
"""

import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
# Guarded like build_brain_viewer.py: tests import this module with
# src/python already on sys.path, and a duplicate entry fails the conftest
# idempotency check in tests/test_portability.py.
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from neuroimaging import viewer  # noqa: E402

EXIT_UNPLACEABLE = 2
EXIT_RENDER = 3
R2_FLOOR_DEFAULT = 10.0   # percent, as build_brain_viewer.py
ARTIFACT_CAP_MB = 16.0    # claude.ai artifact route; larger bundles are file:// only
PROFILE_VERSION = 1       # bump when a display profile changes: keys change,
                          # existing bundles rebuild

HEMIS = {"L": "lh", "R": "rh"}
MAP_EXTS = (".nii.gz", ".nii")
SURF_EXTS = (".shape.gii", ".func.gii")
FEATURE_EXTS = (".csv", ".parquet")
TABLE_SUFFIXES = ("_events.tsv", "_beh.tsv")
_ALL_EXTS = MAP_EXTS + SURF_EXTS + FEATURE_EXTS + (".gii", ".tsv", ".json", ".html")

FEATURE_EXTRACTORS = ("viz2psy", "aud2psy", "word2psy")

# Stimulus registry fallback (contracts §4.2): store subdirectory -> the
# stimulus set's image or audio directory under <bids>/stimuli. Used only
# when the sidecar's own recorded input path no longer exists.
STORE_IMAGE_DIRS = {
    "shared1000": ("shared1000", "images"),
    "movie_cues": ("movies", "movie_cues"),
    "movies": ("movies", "movie_files"),
}


class Unplaceable(Exception):
    """The path cannot be placed; message names the flag that would. Exit 2."""


class RenderError(Exception):
    """A renderer failed; message carries its stderr. Exit 3."""


# ---------------------------------------------------------------------------
# filename grammar (key-value_ entities only; no pybids, no catalog)
# ---------------------------------------------------------------------------

def split_name(name):
    """'sub-01_task-x_bold.nii.gz' -> ('sub-01_task-x_bold', '.nii.gz')."""
    for ext in _ALL_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)], ext
    return Path(name).stem, Path(name).suffix


def parse_entities(name):
    """Return (entities as an ordered dict, suffix, ext) from a filename.

    The grammar is the BIDS ``key-value_`` chain; the last dash-less token
    is the suffix. Anything else in the name is ignored.
    """
    stem, ext = split_name(name)
    ents, suffix = {}, None
    for part in stem.split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            if k.isalnum() and v:
                ents[k] = v
                continue
        suffix = part
    return ents, suffix, ext


@dataclass
class Target:
    kind: str                       # volume | surface | features
    maps: list                      # Path(s); one for a file, many for a dir
    entities: dict = field(default_factory=dict)   # shared, filename order
    suffix: str = None
    sidecar: Path = None            # features only
    extractor: str = None           # features only
    source: Path = None             # what the user pointed at


PRF_DESCS = ("R2", "angle", "eccentricity", "size", "sigma", "gain",
             "exponent")


def family_of(entities, suffix):
    """Display family from the entity that names the quantity."""
    if entities.get("desc") in PRF_DESCS and suffix and suffix.endswith("prf"):
        return "prf"
    stat = entities.get("stat")
    if stat in ("z", "t", "effect"):
        return f"glm-{stat}"
    return "unknown"


# ---------------------------------------------------------------------------
# stage 1 — classify
# ---------------------------------------------------------------------------

def classify(path):
    """Return a list of Targets for *path* (one for a file, one per group
    for a directory). Raises Unplaceable with a message naming the flag or
    tool that would place the file."""
    path = Path(path)
    if path.is_dir():
        return _classify_dir(path)
    if not path.exists():
        raise Unplaceable(f"no such file: {path}")
    name = path.name
    if name.endswith(MAP_EXTS):
        return [_classify_map(path)]
    if name.endswith(SURF_EXTS):
        return [_classify_surface(path)]
    if name.endswith(FEATURE_EXTS):
        return [_classify_features(path)]
    if name.endswith(TABLE_SUFFIXES):
        raise Unplaceable(
            f"{path} is a behavioral/events table; tables are not in "
            "mmmview's first cut — use the MCP plot tools "
            "(plot_timeline_responses, plot_accuracy_by_condition, "
            "plot_rt_distribution, ...) on it")
    raise Unplaceable(f"cannot place {path}; pass --underlay/--mesh, or see "
                      "build_brain_viewer.py")


def _classify_map(path):
    ents, suffix, _ = parse_entities(path.name)
    return Target("volume", [path], ents, suffix, source=path)


def _classify_surface(path):
    ents, suffix, _ = parse_entities(path.name)
    if "hemi" not in ents:
        raise Unplaceable(f"{path.name} carries no hemi- entity, so no mesh "
                          "can be chosen; pass --mesh SURF")
    if ents["hemi"] not in HEMIS:
        raise Unplaceable(f"hemi-{ents['hemi']} in {path.name} is not L/R; "
                          "pass --mesh SURF")
    space = ents.get("space")
    if space is not None and space != "fsnative":
        raise Unplaceable(
            f"space-{space} in {path.name}: only fsnative surfaces are "
            "placed in the first cut (no template mesh is vendored); "
            "build_brain_viewer.py surface --mesh takes an explicit mesh")
    return Target("surface", [path], ents, suffix, source=path)


def _group_key(target):
    e = target.entities
    return (target.kind, e.get("space"), e.get("hemi"), e.get("ses"),
            e.get("task"), e.get("run"), target.suffix)


def _classify_dir(path):
    singles = []
    for p in sorted(path.iterdir()):
        if not p.is_file() or "_desc-viewer" in p.name:
            continue
        if p.name.endswith(MAP_EXTS):
            t = _classify_map(p)
            if "sub" in t.entities:     # entity-less volumes cannot be grouped
                singles.append(t)
        elif p.name.endswith(SURF_EXTS):
            try:
                singles.append(_classify_surface(p))
            except Unplaceable:
                continue        # a directory skips what it cannot place
    if not singles:
        raise Unplaceable(f"no brain maps (*.nii.gz, *.shape.gii, "
                          f"*.func.gii) in {path}; pass one file")
    subs = {t.entities.get("sub") for t in singles}
    if len(subs) > 1:
        raise Unplaceable(f"maps in {path} span several subjects "
                          f"({', '.join(sorted(s or '?' for s in subs))}); "
                          "point mmmview at one sub-## directory")
    groups = {}
    for t in singles:
        groups.setdefault(_group_key(t), []).append(t)
    out = []
    for members in groups.values():
        common = dict(members[0].entities)
        for t in members[1:]:
            common = {k: v for k, v in common.items()
                      if t.entities.get(k) == v}
        out.append(Target(members[0].kind, [t.maps[0] for t in members],
                          common, members[0].suffix, source=path))
    return out


def find_sidecar(path):
    """Contract B sidecar for a feature file: <stem>.meta.json, else the
    prefix family (clap_text_chunks.csv -> clap_text.meta.json)."""
    stem, _ = split_name(path.name)
    parts = stem.split("_")
    for i in range(len(parts), 0, -1):
        cand = path.parent / ("_".join(parts[:i]) + ".meta.json")
        if cand.exists():
            return cand
    return None


def _classify_features(path):
    sidecar = find_sidecar(path)
    if sidecar is None:
        raise Unplaceable(
            f"no Contract B sidecar (*.meta.json) beside {path}; mmmview "
            "dispatches feature files by the sidecar's extractor")
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError) as exc:
        raise Unplaceable(f"cannot read {sidecar}: {exc}")
    extractor = meta.get("extractor")
    if extractor == "psytwill":
        raise Unplaceable(
            f"{path.name} is a psytwill aggregate ({sidecar.name} says "
            "extractor psytwill); psytwill has no browse verb yet, so "
            "there is nothing to dispatch to")
    if extractor not in FEATURE_EXTRACTORS:
        raise Unplaceable(
            f"{sidecar.name} names extractor {extractor!r}; mmmview knows "
            f"{', '.join(FEATURE_EXTRACTORS)}")
    return Target("features", [path], {}, None, sidecar=sidecar,
                  extractor=extractor, source=path)


# ---------------------------------------------------------------------------
# stage 2 — resolve
# ---------------------------------------------------------------------------

@dataclass
class Roots:
    deriv: Path
    bids: Path
    stimfeat_env: Path = None
    catalog_db: Path = None


def load_roots(deriv_override=None):
    try:
        from core.config import load_config
        cfg = load_config(config_dir=_REPO_ROOT / "config")
    except Exception as exc:
        if deriv_override:
            return Roots(Path(deriv_override), Path(deriv_override).parent)
        sys.exit(f"ERROR: could not load config/base.toml ({exc}); pass "
                 "--deriv-root")
    paths = cfg["paths"]
    deriv = Path(deriv_override) if deriv_override else Path(paths["output_dir"])
    bids = Path(paths["bids_project_dir"])
    env = paths.get("stimfeat_env")
    return Roots(deriv, bids, Path(env) if env else None,
                 bids / "inventory" / "catalog.duckdb")


@dataclass
class Opts:
    underlay: str = None
    mesh: str = None
    surf: str = "inflated"
    r2_floor: float = R2_FLOOR_DEFAULT
    out_dir: str = None
    force: bool = False


@dataclass
class Plan:
    renderer: str                  # volume | surface | features
    out: Path
    title: str
    inputs: dict = field(default_factory=dict)   # named source paths
    display: list = field(default_factory=list)  # one entry per map
    command: list = None           # features: argv to run
    messages: list = field(default_factory=list)
    key: str = ""                  # idempotence key (bundles)
    notes: str = ""


# display profiles — the table Ben asked to live in the dispatcher.
# cal_max None = 99th percentile of the surviving samples.
PRF_DISPLAY = {
    "R2": {"colormap": "viridis", "cal_max": None},
    "angle": {"colormap": "hsv", "cal_min": 0.0, "cal_max": 360.0,
              "angle_legend": True},
    "eccentricity": {"colormap": "turbo", "cal_min": 0.0, "cal_max": None},
    "size": {"colormap": "plasma", "cal_min": 0.0, "cal_max": None},
    "sigma": {"colormap": "plasma", "cal_min": 0.0, "cal_max": None},
    "gain": {"colormap": "viridis", "cal_min": 0.0, "cal_max": None},
    "exponent": {"colormap": "viridis", "cal_min": 0.0, "cal_max": None},
}
DISPLAY_PROFILES = {
    # masked at R2 > floor; per-parameter colormap from PRF_DISPLAY
    "prf": {"mask": "R2", "tails": False},
    # positive tail warm, negative tail (as |z|) winter; cal_min = the
    # displayed floor, the conventional threshold is in the label
    "glm-z": {"tails": True, "cal_min": 2.3, "threshold": 3.1,
              "pos": "warm", "neg": "winter"},
    "glm-t": {"tails": True, "cal_min": 2.0, "threshold": None,
              "pos": "warm", "neg": "winter"},
    "glm-effect": {"tails": True, "cal_min": 0.0, "threshold": None,
                   "pos": "warm", "neg": "winter"},
    "unknown": {"tails": False, "colormap": "viridis", "cal_min": 0.0},
}
_FAMILY_ORDER = {"prf": 0, "glm-z": 1, "glm-t": 2, "glm-effect": 3,
                 "unknown": 4}


def display_for(map_path, entities, suffix, r2_floor):
    """One display entry for a map: family, profile, mask, label."""
    family = family_of(entities, suffix)
    prof = dict(DISPLAY_PROFILES[family])
    entry = {"map": Path(map_path), "family": family, "profile": prof,
             "mask": None, "label": _label(entities, suffix, family),
             "message": None}
    if family == "prf":
        param = entities["desc"]
        prof.update(PRF_DISPLAY[param])
        prof.setdefault("cal_min", r2_floor if param == "R2" else 0.0)
        prof["floor"] = r2_floor
        r2 = Path(map_path).with_name(
            Path(map_path).name.replace(f"_desc-{param}_", "_desc-R2_"))
        if r2.exists():
            entry["mask"] = r2
        else:
            entry["message"] = (f"no R2 map beside {Path(map_path).name}; "
                                "drawn unmasked")
    elif family == "unknown":
        entry["message"] = (f"no display profile for {Path(map_path).name} "
                            "(desc/stat names no known family), using default")
    return entry


def _label(entities, suffix, family):
    if family == "prf":
        return entities["desc"]
    bits = [f"{k}-{v}" for k, v in entities.items()
            if k in ("contrast", "desc", "stat", "label")]
    return " ".join(bits) or (suffix or "map")


def _sorted_display(entries):
    return sorted(entries, key=lambda e: (_FAMILY_ORDER[e["family"]],
                                          str(e["map"])))


def find_underlay(entities, roots):
    """fMRIPrep underlay for a volume map by its space- entity."""
    sub = entities.get("sub")
    if not sub:
        raise Unplaceable("no sub- entity in the filename, so no underlay "
                          "can be chosen; pass --underlay NII")
    fp = roots.deriv / "fmriprep"
    anat = fp / f"sub-{sub}" / "anat"
    space = entities.get("space")
    if space == "T1w":
        cands = [p for p in sorted(anat.glob(f"sub-{sub}*_desc-preproc_T1w.nii.gz"))
                 if "_space-" not in p.name and "_ses-" not in p.name]
        where = f"{anat}/sub-{sub}*_desc-preproc_T1w.nii.gz (no space-)"
    elif space == "MNI152NLin2009cAsym":
        res = entities.get("res")
        pat = (f"sub-{sub}*_space-{space}_"
               + (f"res-{res}_" if res else "*") + "desc-preproc_T1w.nii.gz")
        cands = [p for p in sorted(anat.glob(pat)) if "_ses-" not in p.name]
        where = f"{anat}/{pat}"
    elif space is None:
        ses, task, run = (entities.get(k) for k in ("ses", "task", "run"))
        if not (ses and task):
            raise Unplaceable("a space-less map needs ses- and task- to find "
                              "its coreg boldref; pass --underlay NII")
        func = fp / f"sub-{sub}" / f"ses-{ses}" / "func"
        base = f"sub-{sub}_ses-{ses}_task-{task}"
        cands = sorted(func.glob(f"{base}*_desc-coreg_boldref.nii.gz"))
        if run:
            exact = [p for p in cands if f"_run-{run}_" in p.name]
            cands = exact or [p for p in cands if "_run-" not in p.name]
        else:
            cands = [p for p in cands if "_run-" not in p.name] or cands
        where = f"{func}/{base}*_desc-coreg_boldref.nii.gz"
    else:
        raise Unplaceable(f"no underlay rule for space-{space}; pass "
                          "--underlay NII")
    if len(cands) == 1:
        return cands[0]
    narrowed = _catalog_underlay(entities, roots)
    if len(narrowed) == 1:
        return narrowed[0]
    if not cands and not narrowed:
        raise Unplaceable(f"no underlay for sub-{sub} space-{space or 'func'} "
                          f"at {where}; pass --underlay NII")
    names = ", ".join(p.name for p in (narrowed or cands))
    raise Unplaceable(f"underlay for sub-{sub} space-{space or 'func'} is "
                      f"ambiguous ({names}); pass --underlay NII")


def _catalog_underlay(entities, roots):
    """Read-only catalog lookup, used only when the glob is empty or
    ambiguous (the catalog knows about deleted variants). Any failure is an
    empty result — never a fallback to another subject's anatomy."""
    db = roots.catalog_db
    if not db or not Path(db).exists():
        return []
    try:
        from core import catalog
        space = entities.get("space")
        if space is None:
            sql = ("SELECT root, path FROM files WHERE sub = ? AND ses = ? "
                   "AND task = ? AND \"desc\" = 'coreg' AND suffix = 'boldref' "
                   "AND ext = '.nii.gz' AND space IS NULL")
            params = [entities["sub"], entities["ses"], entities["task"]]
        else:
            sql = ("SELECT root, path FROM files WHERE sub = ? AND ses IS NULL "
                   "AND \"desc\" = 'preproc' AND suffix = 'T1w' AND ext = "
                   "'.nii.gz' AND space IS NOT DISTINCT FROM ?")
            params = [entities["sub"], None if space == "T1w" else space]
            if entities.get("res"):
                sql += " AND res = ?"
                params.append(entities["res"])
        _, rows = catalog.run_select(db, sql, params)
    except Exception:
        return []
    fp = (roots.deriv / "fmriprep").resolve()
    out = []
    for r in rows:
        p = Path(r["root"]) / r["path"] if r.get("root") else Path(r["path"])
        if p.exists() and str(p.resolve()).startswith(str(fp)):
            out.append(p)
    return sorted(set(out))


def find_mesh(entities, roots, surf):
    sub, hemi = entities.get("sub"), entities.get("hemi")
    if not sub:
        raise Unplaceable("no sub- entity, so no FreeSurfer mesh can be "
                          "chosen; pass --mesh SURF")
    d = roots.deriv / "fmriprep" / "sourcedata" / "freesurfer" / f"sub-{sub}" / "surf"
    mesh, curv = d / f"{HEMIS[hemi]}.{surf}", d / f"{HEMIS[hemi]}.curv"
    if not mesh.exists():
        raise Unplaceable(f"no FreeSurfer mesh at {mesh}; pass --mesh SURF")
    return mesh, (curv if curv.exists() else None)


def qc_dir_for(path):
    """<sub-##>/qc when the path sits under a subject directory, else
    <parent>/qc (a directory input uses itself as the parent)."""
    path = Path(path)
    start = path if path.is_dir() else path.parent
    for anc in (start, *start.parents):
        if anc.name.startswith("sub-"):
            return anc / "qc"
    return start / "qc"


def bundle_name(entities, suffix):
    bits = [f"{k}-{v}" for k, v in entities.items() if k != "desc"]
    bits.append("desc-viewer")
    name = "_".join(bits)
    return name + (f"_{suffix}" if suffix else "") + ".html"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(inputs, display, opts):
    h = hashlib.sha256()
    h.update(f"v{PROFILE_VERSION}".encode())
    for p in sorted(str(p) for p in inputs if p):
        h.update(p.encode())
        h.update(_sha(p).encode())
    prof = [(str(e["map"]), e["family"], json.dumps(e["profile"], sort_keys=True))
            for e in display]
    h.update(json.dumps(prof).encode())
    h.update(f"{opts.r2_floor}|{opts.surf}".encode())
    return h.hexdigest()


def resolve(target, roots, opts=None):
    """Turn a Target into a Plan: where the underlay/mesh/command comes from,
    which profile each map gets, and where the output lands."""
    opts = opts or Opts()
    if target.kind == "features":
        return _resolve_features(target, roots, opts)

    out_dir = Path(opts.out_dir) if opts.out_dir else qc_dir_for(target.source)
    out = out_dir / bundle_name(target.entities, target.suffix)
    display = _sorted_display([
        display_for(m, parse_entities(m.name)[0], target.suffix, opts.r2_floor)
        for m in target.maps])
    messages = [e["message"] for e in display if e["message"]]
    inputs = {"maps": list(target.maps)}
    sub = target.entities.get("sub", "?")
    if target.kind == "volume":
        if opts.underlay:
            underlay = Path(opts.underlay)
            if not underlay.exists():
                raise Unplaceable(f"--underlay {underlay} does not exist")
        else:
            underlay = find_underlay(target.entities, roots)
        inputs["underlay"] = underlay
        space = target.entities.get("space") or "func"
        title = f"sub-{sub} {_title_bits(target)} ({space})"
        srcs = [underlay] + list(target.maps)
    else:
        if opts.mesh:
            mesh = Path(opts.mesh)
            if not mesh.exists():
                raise Unplaceable(f"--mesh {mesh} does not exist")
            curv = mesh.with_name(mesh.name.split(".")[0] + ".curv")
            curv = curv if curv.exists() else None
        else:
            mesh, curv = find_mesh(target.entities, roots, opts.surf)
        inputs["mesh"], inputs["curv"] = mesh, curv
        title = (f"sub-{sub} {_title_bits(target)} hemi-"
                 f"{target.entities.get('hemi')} ({mesh.name})")
        srcs = [mesh, curv] + list(target.maps)
    masks = sorted({e["mask"] for e in display if e["mask"]})
    key = _key(srcs + masks, display, opts)
    notes = _provenance(srcs + masks, display, key, opts)
    return Plan(target.kind, out, title, inputs, display, None, messages,
                key, notes)


def _title_bits(target):
    e = target.entities
    bits = [f"{k}-{v}" for k, v in e.items()
            if k in ("ses", "task", "run", "contrast", "stat", "desc")]
    if target.suffix:
        bits.append(target.suffix)
    return " ".join(bits)


def _provenance(sources, display, key, opts):
    fams = sorted({e["family"] for e in display})
    lines = [f"built {datetime.date.today().isoformat()} by "
             "mmmdata/scripts/mmmview.py (workbench brain-viewer)",
             f"families: {', '.join(fams)}; r2 floor {opts.r2_floor}% "
             "(pRF maps masked to R2 > floor); z/t/effect negative tails "
             "are a separate layer",
             f"mmmview-key: {key}"]
    lines += [f"  {Path(s).name}" for s in sources if s]
    return "\n".join(lines)


def _resolve_features(target, roots, opts):
    csv = target.maps[0]
    meta = json.loads(target.sidecar.read_text())
    out_dir = Path(opts.out_dir) if opts.out_dir else qc_dir_for(csv)
    out = out_dir / (split_name(csv.name)[0] + "_desc-viewer.html")
    env = roots.stimfeat_env
    bindir = Path(env) / "bin" if env else None
    messages = []
    inputs_meta = meta.get("input") or {}
    paths = inputs_meta.get("paths") or []
    first = Path(paths[0]) if paths else None
    subdir = csv.parent.name

    def exe(name):
        return str(bindir / name) if bindir else name

    if target.extractor == "viz2psy":
        cmd = [exe("viz2psy-viz"), "dashboard", str(csv), "-o", str(out)]
        root = first.parent if first and first.parent.is_dir() else None
        if root is None and subdir in STORE_IMAGE_DIRS:
            cand = roots.bids / "stimuli" / Path(*STORE_IMAGE_DIRS[subdir])
            root = cand if cand.is_dir() else None
        if root is not None:
            cmd += ["--image-root", str(root)]
        else:
            cmd.append("--no-images")
            messages.append("image root not resolvable from the sidecar or "
                            "the stimulus registry; dashboard builds without "
                            "thumbnails")
    elif target.extractor == "aud2psy":
        cmd = [exe("aud2psy"), "viz", "browse", str(csv), "-o", str(out)]
        if first is not None and first.exists():
            cmd += ["--audio", str(first)]
        else:
            messages.append("audio input from the sidecar is missing; "
                            "dashboard builds without playback (pass "
                            "--audio to aud2psy viz browse yourself)")
    else:
        cmd = [exe("word2psy"), "viz", "browse", str(csv), "-o", str(out)]
    if bindir is None:
        messages.append("paths.stimfeat_env is not set in config; running "
                        f"{cmd[0]} from PATH")
    title = f"{target.extractor} {csv.name}"
    return Plan("features", out, title,
                {"csv": csv, "sidecar": target.sidecar}, [], cmd, messages)


# ---------------------------------------------------------------------------
# stage 3 — render and open
# ---------------------------------------------------------------------------

def is_current(plan):
    """True when the output exists and was built from these inputs."""
    if not plan.out.exists():
        return False
    if plan.renderer == "features":
        newest = max(Path(p).stat().st_mtime for p in plan.inputs.values())
        return plan.out.stat().st_mtime >= newest
    with open(plan.out, "r", errors="replace") as f:
        return f"mmmview-key: {plan.key}" in f.read()


def _p99(values):
    good = values[np.isfinite(values) & (values != viewer.MASK_SENTINEL)]
    if good.size == 0:
        return None
    return round(float(np.percentile(good, 99)), 2)


def _volume_specs(entry, floor):
    import nibabel as nib
    prof, path = entry["profile"], entry["map"]
    if entry["family"] == "prf" and entry["mask"] is not None:
        img = viewer.masked_volume(path, entry["mask"], floor)
        data = np.asarray(img.dataobj)
        spec = {"image": img, "name": path.name, "label": entry["label"],
                "colormap": prof["colormap"], "cal_min": prof["cal_min"],
                "cal_max": prof["cal_max"], "angle_legend":
                prof.get("angle_legend", False)}
        if spec["cal_max"] is None:
            spec["cal_max"] = _p99(data)
        return [spec]
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if not prof.get("tails"):
        keep = np.isfinite(data)
        out = np.where(keep, data, np.float32(viewer.MASK_SENTINEL))
        new = nib.Nifti1Image(out, img.affine, img.header)
        new.header.set_data_dtype(np.float32)
        return [{"image": new, "name": path.name, "label": entry["label"],
                 "colormap": prof.get("colormap", "viridis"),
                 "cal_min": prof.get("cal_min", 0.0),
                 "cal_max": prof.get("cal_max") or _p99(out),
                 "angle_legend": prof.get("angle_legend", False)}]
    return _tail_specs(entry, data, lambda arr: _nifti_like(img, arr))


def _nifti_like(img, arr):
    import nibabel as nib
    new = nib.Nifti1Image(arr, img.affine, img.header)
    new.header.set_data_dtype(np.float32)
    return new


def _tail_specs(entry, data, wrap):
    """Positive and negative tails as two layers (|value| each)."""
    prof, path = entry["profile"], entry["map"]
    finite = np.isfinite(data)
    # ceiling = p99 of the samples that survive the floor (as the pRF
    # convention); p99 over the whole volume sits just above the floor
    # because background dominates, which saturates the map
    mag = np.abs(data)
    above = finite & (mag >= prof["cal_min"])
    absmax = (_p99(np.where(above, mag, viewer.MASK_SENTINEL))
              or _p99(np.where(finite, mag, viewer.MASK_SENTINEL)) or 1.0)
    pos = np.where(finite & (data >= 0), data, np.float32(viewer.MASK_SENTINEL))
    neg = np.where(finite & (data < 0), -data, np.float32(viewer.MASK_SENTINEL))
    thr = (f" (threshold {prof['threshold']})" if prof.get("threshold")
           else "")
    specs = [{"image" if wrap else "values": wrap(pos) if wrap else pos,
              "name": path.name, "label": f"{entry['label']} +{thr}",
              "colormap": prof["pos"], "cal_min": prof["cal_min"],
              "cal_max": absmax}]
    if np.any(neg != viewer.MASK_SENTINEL):
        specs.append({"image" if wrap else "values": wrap(neg) if wrap else neg,
                      "name": "neg_" + path.name,
                      "label": f"{entry['label']} − (as |value|){thr}",
                      "colormap": prof["neg"], "cal_min": prof["cal_min"],
                      "cal_max": absmax})
    return specs


def _surface_specs(entry, floor):
    import nibabel as nib
    prof, path = entry["profile"], entry["map"]
    if entry["family"] == "prf":
        vals = viewer.masked_shape_values(path, entry["mask"], floor)
        spec = {"values": vals, "name": path.name, "label": entry["label"],
                "colormap": prof["colormap"], "cal_min": prof["cal_min"],
                "cal_max": prof["cal_max"],
                "angle_legend": prof.get("angle_legend", False)}
        if spec["cal_max"] is None:
            spec["cal_max"] = _p99(vals)
        return [spec]
    if not prof.get("tails"):
        vals = viewer.masked_shape_values(path)
        return [{"values": vals, "name": path.name, "label": entry["label"],
                 "colormap": prof.get("colormap", "viridis"),
                 "cal_min": prof.get("cal_min", 0.0),
                 "cal_max": prof.get("cal_max") or _p99(vals)}]
    data = nib.load(str(path)).darrays[0].data.astype(np.float32)
    return _tail_specs(entry, data, None)


def render(plan, force=False):
    """Build the plan's output. Returns (out_path, built) where built is
    False when an up-to-date output was reused."""
    if not force and is_current(plan):
        return plan.out, False
    plan.out.parent.mkdir(parents=True, exist_ok=True)
    if plan.renderer == "features":
        try:
            res = subprocess.run(plan.command, capture_output=True, text=True)
        except OSError as exc:
            raise RenderError(f"cannot run {plan.command[0]}: {exc}")
        if res.returncode != 0:
            raise RenderError(f"{' '.join(shlex.quote(c) for c in plan.command)}"
                              f"\n{res.stderr.strip()}")
        if not plan.out.exists():
            raise RenderError(f"{plan.command[0]} exited 0 but wrote no "
                              f"{plan.out}")
        return plan.out, True
    floor = _floor_from(plan)
    try:
        if plan.renderer == "volume":
            overlays = []
            for e in plan.display:
                overlays += _volume_specs(e, floor)
            for i, o in enumerate(overlays):
                o["visible"] = i == 0
            viewer.build_volume_viewer(
                {"path": plan.inputs["underlay"], "label": "underlay"},
                overlays, plan.out, title=plan.title, notes=plan.notes)
        else:
            layers = []
            if plan.inputs.get("curv"):
                layers.append({"path": plan.inputs["curv"], "label": "curvature",
                               "shade": True, "colormap": "gray",
                               "cal_min": 0.3, "cal_max": 0.8, "opacity": 0.7})
            for e in plan.display:
                layers += _surface_specs(e, floor)
            viewer.build_surface_viewer(plan.inputs["mesh"], layers, plan.out,
                                        title=plan.title, notes=plan.notes)
    except Exception as exc:   # renderer failure, not a placement failure
        raise RenderError(f"{type(exc).__name__}: {exc}")
    return plan.out, True


def _floor_from(plan):
    for e in plan.display:
        if "floor" in e["profile"]:
            return e["profile"]["floor"]
    return R2_FLOOR_DEFAULT


def open_view(path):
    """Try $BROWSER, then xdg-open under a display. Returns a note, or None
    when nothing could open it (the caller prints the path and a hint)."""
    browser = os.environ.get("BROWSER")
    uri = Path(path).resolve().as_uri()     # helpers want a URI, not a path
    if browser:
        try:
            subprocess.Popen(shlex.split(browser) + [uri],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return "opened via $BROWSER"
        except OSError:
            pass
    if os.environ.get("DISPLAY") and shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", uri], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return "opened via xdg-open"
    return None


OPEN_HINT = ("nothing here can open a browser (no $BROWSER, no $DISPLAY). "
             "The bundle is self-contained: in VS Code Remote-SSH right-click "
             "it in the Explorer and Download, then open it locally; or use "
             "the Live Preview extension on it; or port-forward a "
             "`python -m http.server` from its directory")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mmmview", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="a map, a directory of maps, "
                    "or a Contract B feature file")
    ap.add_argument("--no-open", action="store_true",
                    help="build only; print the output path")
    ap.add_argument("--out-dir", help="override <sub-##>/qc/")
    ap.add_argument("--underlay", help="explicit NIfTI underlay (volume maps)")
    ap.add_argument("--mesh", help="explicit FreeSurfer mesh (surface maps)")
    ap.add_argument("--surf", default="inflated",
                    help="FreeSurfer mesh flavor when resolving by hemi- "
                         "(inflated, pial, white)")
    ap.add_argument("--r2-floor", type=float, default=R2_FLOOR_DEFAULT,
                    help="pRF maps are masked to R2 > this (percent)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when the output is current")
    ap.add_argument("--deriv-root", help="override config derivatives root")
    args = ap.parse_args(argv)

    opts = Opts(args.underlay, args.mesh, args.surf, args.r2_floor,
                args.out_dir, args.force)
    roots = load_roots(args.deriv_root)
    try:
        targets = classify(args.path)
        plans = [resolve(t, roots, opts) for t in targets]
    except Unplaceable as exc:
        print(f"mmmview: cannot place {args.path}: {exc}", file=sys.stderr)
        return EXIT_UNPLACEABLE

    outs = []
    for plan in plans:
        for m in plan.messages:
            print(f"mmmview: note: {m}", file=sys.stderr)
        try:
            out, built = render(plan, force=opts.force)
        except RenderError as exc:
            print(f"mmmview: renderer failed for {plan.out.name}:\n{exc}",
                  file=sys.stderr)
            return EXIT_RENDER
        mb = out.stat().st_size / 1e6
        cap = ("" if mb <= ARTIFACT_CAP_MB else
               f"  [exceeds the {ARTIFACT_CAP_MB:.0f} MB artifact cap — "
               "file:// only]")
        print(f"{'wrote' if built else 'current'} {out}  ({mb:.1f} MB, "
              f"{plan.renderer}){cap}")
        outs.append(out)

    if not args.no_open and outs:
        if len(outs) > 1:
            print(f"opening the first of {len(outs)} outputs")
        note = open_view(outs[0])
        print(note if note else OPEN_HINT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

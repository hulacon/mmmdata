#!/usr/bin/env python3
"""
mmmview.py — one path into the dataset in, the right interactive view out.

    mmmview PATH [--no-open] [--out-dir DIR] [--underlay NII | --mesh SURF]
                 [--surf inflated|pial|white] [--r2-floor F] [--force]
                 [--films-dir DIR]

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
                                         group, every map a toggle layer;
                                         pRF fit variants (prf, negprf,
                                         motion6prf, ...) merge into ONE
                                         bundle per (space, hemi) with a
                                         variant selector, named
                                         *_desc-viewer_prfvariants.html
    *.csv / *.parquet + *.meta.json      the dashboard of the sidecar's
                                         extractor (viz2psy, aud2psy, word2psy)
    movies_*_features.csv/.parquet       psytwill's movies timeline viewer,
    (one table, or a directory of them)  over the whole table directory; the
                                         per-film media tree is the sibling
                                         movies/ directory (or --films-dir).
                                         Covers composed TB runs laid out the
                                         same way (workbench tb-timelines)
    *_events.tsv, *_beh.tsv              not here; the MCP plot tools cover them

Display profiles live in DISPLAY_PROFILES below, keyed on the entity that
names the quantity (desc- for pRF parameters, stat- for GLM maps). They are
recreated in the viewer per family and never written to disk (decided
2026-09-11). A z/t/effect map's negative tail is a second toggle layer
(the template has no two-tailed colormap); the pRF profile extends the
table in build_brain_viewer.py.

Bundles are findings: they land in <sub-##>/viz/ beside the data, never in a
repo. A second call on unchanged inputs reuses the bundle (a sha256 key over
the inputs and profile is written into the provenance note). A viz dir
holding more than one bundle also gets a data-free index.html — a pulldown
over every bundle present, shown in an iframe — regenerated from the
directory listing whenever a bundle lands, and opened (preselecting the
bundle just built) instead of "the first of N outputs".

Roots come from config/base.toml (+ local.toml): paths.output_dir,
paths.bids_project_dir, paths.stimfeat_env.
"""

import argparse
import datetime
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass, field
from html import escape as _escape
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
PROFILE_VERSION = 3       # bump when a display profile changes: keys change,
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
    kind: str                       # volume | surface | features | movies
    maps: list                      # Path(s); one for a file, many for a dir
    entities: dict = field(default_factory=dict)   # shared, filename order
    suffix: str = None              # None when variants spans several
    sidecar: Path = None            # features only
    extractor: str = None           # features only
    source: Path = None             # what the user pointed at
    variants: tuple = None          # >1 pRF fit variants merged in one bundle


PRF_DESCS = ("R2", "angle", "eccentricity", "size", "sigma", "gain",
             "exponent")


def _variant_rank(v):
    """Stable variant order: the plain fit first, its negative second,
    everything else (confound variants etc.) alphabetically after."""
    return ({"prf": 0, "negprf": 1}.get(v, 2), v or "")


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
        # a movies feature-table directory, or a composed-run root holding
        # one as features/ (the tb-timelines compose layout: features/ +
        # movies/ side by side)
        for cand in (path, path / "features"):
            if cand.is_dir() and any(is_movies_table(p.name)
                                     for p in cand.iterdir() if p.is_file()):
                return [_movies_target(cand, source=path)]
        raise Unplaceable(f"no brain maps (*.nii.gz, *.shape.gii, "
                          f"*.func.gii) and no movies feature tables "
                          f"(movies_*_features.*) in {path}; pass one file")
    subs = {t.entities.get("sub") for t in singles}
    if len(subs) > 1:
        raise Unplaceable(f"maps in {path} span several subjects "
                          f"({', '.join(sorted(s or '?' for s in subs))}); "
                          "point mmmview at one sub-## directory")
    groups = {}
    for t in singles:
        groups.setdefault(_group_key(t), []).append(t)
    # pRF fit variants (same space/hemi/ses/task/run, different suffix —
    # prf/negprf/motion6prf/...) merge into ONE bundle carrying a variant
    # selector; every other key keeps a bundle per suffix.
    buckets = {}
    for key, members in groups.items():
        is_prf = family_of(members[0].entities, key[-1]) == "prf"
        bucket = (key[:-1], True) if is_prf else (key, False)
        buckets.setdefault(bucket, []).append((key[-1], members))
    out = []
    for variant_groups in buckets.values():
        variant_groups.sort(key=lambda kv: _variant_rank(kv[0]))
        members = [t for _, ms in variant_groups for t in ms]
        common = dict(members[0].entities)
        for t in members[1:]:
            common = {k: v for k, v in common.items()
                      if t.entities.get(k) == v}
        vnames = tuple(v for v, _ in variant_groups)
        out.append(Target(members[0].kind, [t.maps[0] for t in members],
                          common,
                          suffix=vnames[0] if len(vnames) == 1 else None,
                          source=path,
                          variants=vnames if len(vnames) > 1 else None))
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


def is_movies_table(name):
    """A psytwill movie-schema table: movies_<grain>_features.csv/.parquet.
    The naming is the dispatch signal itself — composed TB runs (workbench
    tb-timelines) carry no sidecar but use the same schema and names."""
    stem, ext = split_name(name)
    return (ext in FEATURE_EXTS and stem.startswith("movies_")
            and stem.endswith("_features"))


def _movies_target(features_dir, source):
    tables = sorted(p for p in features_dir.iterdir()
                    if p.is_file() and is_movies_table(p.name))
    return Target("movies", tables, {}, None, source=source)


def _classify_features(path):
    sidecar = find_sidecar(path)
    if sidecar is None:
        if is_movies_table(path.name):
            return _movies_target(path.parent, source=path)
        raise Unplaceable(
            f"no Contract B sidecar (*.meta.json) beside {path}; mmmview "
            "dispatches feature files by the sidecar's extractor")
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError) as exc:
        raise Unplaceable(f"cannot read {sidecar}: {exc}")
    extractor = meta.get("extractor")
    if extractor == "psytwill":
        if is_movies_table(path.name):
            return _movies_target(path.parent, source=path)
        raise Unplaceable(
            f"{path.name} is a psytwill aggregate ({sidecar.name} says "
            "extractor psytwill); psytwill's browse verb covers the movies "
            "set only (movies_*_features tables -> psytwill viz movies), "
            "and this is not one")
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
    films_dir: str = None          # movies: per-film media tree override


@dataclass
class Plan:
    renderer: str                  # volume | surface | features | movies
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
        if param == "R2":
            # R2 thresholds itself, which is what cal_min already does in
            # display — so embed it unthresholded and let the viewer's
            # threshold slider walk the floor down to 0
            pass
        elif r2.exists():
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
                                          _variant_rank(e.get("variant")),
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


def viz_dir_for(path):
    """<sub-##>/viz when the path sits under a subject directory, else
    <parent>/viz (a directory input uses itself as the parent). viz, not qc:
    a bundle is a view of the data, not a review of its quality."""
    path = Path(path)
    start = path if path.is_dir() else path.parent
    for anc in (start, *start.parents):
        if anc.name.startswith("sub-"):
            return anc / "viz"
    return start / "viz"


def bundle_name(entities, suffix):
    bits = [f"{k}-{v}" for k, v in entities.items() if k != "desc"]
    bits.append("desc-viewer")
    name = "_".join(bits)
    return name + (f"_{suffix}" if suffix else "") + ".html"


_SHA_CACHE = {}


def _sha(path):
    """Content hash, memoized on (path, mtime, size). A reap sweep resolves
    every map in a directory and they share one underlay — without this the
    underlay is re-hashed once per map."""
    try:
        st = os.stat(path)
        ck = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        ck = None
    if ck is not None and ck in _SHA_CACHE:
        return _SHA_CACHE[ck]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if ck is not None:
        _SHA_CACHE[ck] = digest
    return digest


def _key(inputs, display, opts):
    h = hashlib.sha256()
    h.update(f"v{PROFILE_VERSION}".encode())
    for p in sorted(str(p) for p in inputs if p):
        h.update(p.encode())
        h.update(_sha(p).encode())
    prof = [(str(e["map"]), e["family"], e.get("variant"),
             json.dumps(e["profile"], sort_keys=True))
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
    if target.kind == "movies":
        return _resolve_movies(target, roots, opts)

    out_dir = Path(opts.out_dir) if opts.out_dir else viz_dir_for(target.source)
    out = out_dir / bundle_name(
        target.entities, "prfvariants" if target.variants else target.suffix)
    display = []
    for m in target.maps:
        ents, sfx, _ = parse_entities(m.name)
        entry = display_for(m, ents, sfx, opts.r2_floor)
        entry["variant"] = sfx if target.variants else None
        display.append(entry)
    display = _sorted_display(display)
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
    if target.variants:
        bits.append("+".join(target.variants))
    elif target.suffix:
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
    variants = [v for v in dict.fromkeys(e.get("variant") for e in display) if v]
    if variants:
        lines.insert(2, f"variants: {', '.join(variants)} — each masked to "
                        "its own R2; selector in the viewer")
    lines += [f"  {Path(s).name}" for s in sources if s]
    return "\n".join(lines)


def _resolve_features(target, roots, opts):
    csv = target.maps[0]
    meta = json.loads(target.sidecar.read_text())
    out_dir = Path(opts.out_dir) if opts.out_dir else viz_dir_for(csv)
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


def _resolve_movies(target, roots, opts):
    """psytwill's movies timeline viewer over a directory of movie-schema
    tables. The per-film media tree (frames/, audio, transcripts) is the
    sibling movies/ directory in both layouts that exist — the real set
    (stimuli_features/{psytwill,movies}) and a composed TB run
    (<run>/{features,movies}, workbench tb-timelines)."""
    features_dir = target.maps[0].parent
    if opts.films_dir:
        films = Path(opts.films_dir)
        if not films.is_dir():
            raise Unplaceable(f"--films-dir {films} does not exist")
    else:
        films = features_dir.parent / "movies"
        if not films.is_dir():
            raise Unplaceable(
                f"no per-film media tree beside the tables (expected "
                f"{films}); pass --films-dir DIR")
    # default where psytwill puts it: relative media links stay valid there
    out_dir = Path(opts.out_dir) if opts.out_dir else films / "viz" / "timeline"
    env = roots.stimfeat_env
    exe = str(Path(env) / "bin" / "psytwill") if env else "psytwill"
    cmd = [exe, "viz", "movies", "--features-dir", str(features_dir),
           "--films-dir", str(films), "-o", str(out_dir)]
    messages = []
    registry = roots.bids / "stimuli" / "stimulus_registry"
    if registry.is_dir():
        cmd += ["--registry", str(registry)]
    else:
        messages.append("no stimulus registry at "
                        f"{registry}; film titles fall back to slugs")
    if env is None:
        messages.append("paths.stimfeat_env is not set in config; running "
                        "psytwill from PATH")
    title = f"psytwill movies timeline ({features_dir})"
    inputs = {t.name: t for t in target.maps}
    return Plan("movies", out_dir / "index.html", title, inputs, [], cmd,
                messages)


# ---------------------------------------------------------------------------
# stage 3 — render and open
# ---------------------------------------------------------------------------

def is_current(plan):
    """True when the output exists and was built from these inputs."""
    if not plan.out.exists():
        return False
    if plan.renderer in ("features", "movies"):
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
    if entry["family"] == "prf":
        if entry["mask"] is not None:
            img = viewer.masked_volume(path, entry["mask"], floor)
            data = np.asarray(img.dataobj)
        else:
            raw = nib.load(str(path))
            arr = np.asarray(raw.dataobj, dtype=np.float32)
            data = np.where(np.isfinite(arr), arr,
                            np.float32(viewer.MASK_SENTINEL))
            img = _nifti_like(raw, data)
        spec = {"image": img, "name": path.name, "label": entry["label"],
                "colormap": prof["colormap"], "cal_min": prof["cal_min"],
                "cal_max": prof["cal_max"], "angle_legend":
                prof.get("angle_legend", False)}
        if spec["cal_max"] is None:
            # unthresholded R2: ceiling from the samples above the floor,
            # as the masked maps get by construction
            ref = (np.where(data > prof["floor"], data,
                            np.float32(viewer.MASK_SENTINEL))
                   if entry["label"] == "R2" and entry["mask"] is None
                   else data)
            spec["cal_max"] = _p99(ref) or _p99(data)
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
        vals = viewer.masked_shape_values(
            path, entry["mask"], floor if entry["mask"] is not None else None)
        spec = {"values": vals, "name": path.name, "label": entry["label"],
                "colormap": prof["colormap"], "cal_min": prof["cal_min"],
                "cal_max": prof["cal_max"],
                "angle_legend": prof.get("angle_legend", False)}
        if spec["cal_max"] is None:
            ref = (np.where(vals > prof["floor"], vals,
                            np.float32(viewer.MASK_SENTINEL))
                   if entry["label"] == "R2" and entry["mask"] is None
                   else vals)
            spec["cal_max"] = _p99(ref) or _p99(vals)
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
    if plan.renderer in ("features", "movies"):
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
                specs = _volume_specs(e, floor)
                for s in specs:
                    s["variant"] = e.get("variant")
                overlays += specs
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
                specs = _surface_specs(e, floor)
                for s in specs:
                    s["variant"] = e.get("variant")
                layers += specs
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


# ---------------------------------------------------------------------------
# viz-dir index — one pulldown over every bundle in the directory
# ---------------------------------------------------------------------------

_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — mmmview</title>
<style>
  :root { --bg:#101014; --panel:#1a1a22; --ink:#e8e8ee; --edge:#2c2c38; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.45 system-ui, sans-serif;
         display:flex; flex-direction:column; height:100vh; }
  header { padding:8px 14px; border-bottom:1px solid var(--edge);
           background:var(--panel); display:flex; align-items:center;
           gap:12px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  select { background:var(--panel); color:var(--ink);
           border:1px solid var(--edge); border-radius:6px;
           padding:5px 8px; font:inherit; max-width:70vw; }
  iframe { flex:1; border:0; width:100%; }
</style>
</head>
<body>
<header><h1>__TITLE__</h1><select id="pick">
__OPTIONS__
</select></header>
<iframe id="view" src=""></iframe>
<script>
"use strict";
const pick = document.getElementById("pick"), view = document.getElementById("view");
// #<bundle-filename> preselects a view (mmmview passes the one just built)
if (location.hash) {
  const want = decodeURIComponent(location.hash.slice(1));
  for (const o of pick.options) if (o.value === want) pick.value = want;
}
const go = () => { view.src = pick.value; };
pick.addEventListener("change", go);
go();
</script>
</body>
</html>
"""


def _index_label(name):
    """Display label for a bundle filename: its entities minus the
    constant-per-directory sub- and the desc-viewer marker."""
    ents, suffix, _ = parse_entities(name)
    bits = [f"{k}-{v}" for k, v in ents.items() if k not in ("sub", "desc")]
    if suffix:
        bits.append(suffix)
    return " ".join(bits) or name


def write_index(viz_dir):
    """(Re)generate viz_dir/index.html: a pulldown over every viewer bundle
    in the directory, shown in an iframe. Data-free — it holds filenames
    only, so it is rebuilt from the directory listing every time a bundle
    lands and stays current for bundles from earlier runs too. Returns the
    index path, or None when fewer than two bundles exist (a lone bundle
    needs no index)."""
    bundles = sorted(p.name for p in viz_dir.iterdir()
                     if p.is_file() and "_desc-viewer" in p.name
                     and p.name.endswith(".html"))
    agents = agent_artifacts(viz_dir)
    if len(bundles) + len(agents) < 2:
        return None
    options = "\n".join(f'<option value="{n}">{_index_label(n)}</option>'
                        for n in bundles)
    if agents:
        # agent-written pages are views too, but never mmmview's output —
        # kept in their own group so the distinction survives the pulldown
        options += ('\n<optgroup label="Agent-generated">\n'
                    + "\n".join(
                        f'<option value="{a["name"]}">{a["label"]} — '
                        f'{_agent_caption(a)}</option>' for a in agents)
                    + "\n</optgroup>")
    html = (_INDEX_TEMPLATE.replace("__TITLE__", viz_dir.parent.name)
                           .replace("__OPTIONS__", options))
    idx = viz_dir / "index.html"
    if not idx.exists() or idx.read_text() != html:
        idx.write_text(html)
    return idx


# ---------------------------------------------------------------------------
# browse pages — a directory mmmview cannot place is still navigable
# ---------------------------------------------------------------------------

AGENT_MARK = "_desc-agent"
PROV_EXT = ".prov.json"


def agent_artifacts(viz_dir):
    """Agent-written pages in *viz_dir*: `*_desc-agent_<slug>.html`, each
    described by a `<stem>.prov.json` sidecar (author, date, inputs,
    command). A page without a readable sidecar still lists — flagged
    unattributed — because hiding it would be worse than not knowing who
    wrote it. These are never mmmview's output: reap ignores them (they
    carry no mmmview-key) and the sidecar owns them."""
    out = []
    if not Path(viz_dir).is_dir():
        return out
    for p in sorted(Path(viz_dir).iterdir()):
        if not (p.is_file() and AGENT_MARK in p.name
                and p.name.endswith(".html")):
            continue
        author = date = None
        prov = p.with_name(split_name(p.name)[0] + PROV_EXT)
        if prov.exists():
            try:
                meta = json.loads(prov.read_text())
                author, date = meta.get("author"), meta.get("date")
            except (OSError, ValueError):
                pass
        out.append({"name": p.name, "path": p, "label": _index_label(p.name),
                    "author": author, "date": date,
                    "attributed": bool(author or date)})
    return out


def _agent_caption(a):
    if not a["attributed"]:
        return "unattributed"
    return " ".join(str(b) for b in (a["author"], a["date"]) if b)


BROWSE_PAGE = "browse.html"
# The reaper keys on this exact string; every generated page carries it.
BROWSE_SIGNATURE = "<!-- mmmview-browse: generated by mmmdata/scripts/mmmview.py -->"
SOURCEDATA_DIR = "mmmsourcedata"


def sourcedata_refusal(path):
    """The refusal message when *path* is in the PII tree, else None.

    Checked on the raw AND the resolved path: the sibling layout
    (<bids>/../mmmsourcedata) puts it one typo away, and a symlink into it
    is the same leak by another name.
    """
    raw = Path(path)
    for cand in (raw, raw.resolve()):
        if SOURCEDATA_DIR in cand.parts:
            return (f"refusing {path}: {SOURCEDATA_DIR} is outside mmmview's "
                    "scope (PII tree)")
    return None


def _rel_to_root(path, roots):
    """Display path: relative to whichever configured root contains it
    (shortest wins), absolute when neither does."""
    best = str(path)
    for root in (r for r in (roots.bids, roots.deriv) if r):
        try:
            rel = str(Path(path).relative_to(Path(root)))
        except ValueError:
            continue
        if len(rel) < len(best):
            best = rel
    return best


def _child_page(child):
    """An existing page of a child directory: its own viz/browse.html or
    viz/index.html. Never built here — one page per invocation."""
    for name in (BROWSE_PAGE, "index.html"):
        cand = child / "viz" / name
        if cand.is_file():
            return cand
    return None


def _viewable_entries(child, roots, opts):
    """What mmmview could show for one child path (zero entries when it
    cannot place it). `out` is set only when the bundle already exists."""
    try:
        targets = classify(child)
    except Unplaceable:
        return []
    entries = []
    for t in targets:
        label = _label(t.entities, t.suffix, family_of(t.entities, t.suffix))
        out = None
        try:
            plan = resolve(t, roots, opts)
        except Unplaceable:
            plan = None
        if plan is not None and plan.out.exists():
            out = plan.out
        entries.append({"name": child.name, "path": child, "kind": t.kind,
                        "label": label, "out": out})
    return entries


def browse_model(directory, roots, opts=None):
    """The data behind a browse page: subdirectories, what is viewable
    here, and the bundles already in this directory's viz dir. Pure data —
    static mode and serve mode render it with different link policies, so
    the two can never drift."""
    directory = Path(directory)
    opts = opts or Opts()
    model = {"dir": directory, "rel": _rel_to_root(directory, roots),
             "date": datetime.date.today().isoformat(),
             "command": f"mmmview {directory}",
             "subdirs": [], "viewable": [], "bundles": [], "agents": []}
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return model
    viz_here = viz_dir_for(directory)
    for child in children:
        # the viz dir this page lives in is the "Existing bundles" section,
        # not a subdirectory row — listing it makes the page reference
        # itself and never settle (it appears only after the first write)
        if child == viz_here or not child.is_dir():
            continue
        if SOURCEDATA_DIR not in child.parts:
            model["subdirs"].append({"name": child.name, "path": child,
                                     "page": _child_page(child)})
    for child in children:
        model["viewable"] += _viewable_entries(child, roots, opts)
    viz = viz_dir_for(directory)
    if viz.is_dir():
        for p in sorted(viz.iterdir()):
            if p.is_file() and "_desc-viewer" in p.name and p.name.endswith(".html"):
                model["bundles"].append({"name": p.name, "path": p,
                                         "label": _index_label(p.name)})
    model["agents"] = agent_artifacts(viz)
    return model


_BROWSE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — mmmview</title>
__SIGNATURE__
<style>
  :root { --bg:#101014; --panel:#1a1a22; --ink:#e8e8ee; --edge:#2c2c38;
          --dim:#9a9aae; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.45 system-ui, sans-serif; padding:0 0 40px; }
  header { padding:10px 18px; border-bottom:1px solid var(--edge);
           background:var(--panel); }
  header h1 { font-size:15px; margin:0 0 4px; font-weight:600; }
  header p { margin:2px 0; color:var(--dim); font-size:12px; }
  main { padding:0 18px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em;
       color:var(--dim); margin:22px 0 6px; font-weight:600; }
  ul { list-style:none; margin:0; padding:0; }
  li { padding:4px 0; border-bottom:1px solid var(--edge); }
  a { color:#8ab4f8; text-decoration:none; }
  a:hover { text-decoration:underline; }
  code { background:var(--panel); border:1px solid var(--edge);
         border-radius:4px; padding:1px 5px; color:var(--dim);
         font:12px/1.4 ui-monospace, monospace; }
  .kind { color:var(--dim); font-size:12px; margin-left:6px; }
</style>
</head>
<body>
<header>
<h1>__TITLE__</h1>
<p>static snapshot — may be stale; regenerate with: <code>__COMMAND__</code></p>
<p>generated __DATE__ by mmmview</p>
</header>
<main>
__SECTIONS__
</main>
</body>
</html>
"""


def _recipe(path):
    return f"<code>mmmview {_escape(str(path))}</code>"


def _browse_sections(model, link, build_link=None, dir_link=None):
    """HTML for the page body. `link(Path) -> href` places an existing
    file; `build_link(Path) -> href` offers to build one on demand (serve
    mode) — when it is None an unbuilt target shows its recipe instead;
    `dir_link(Path) -> href` addresses a subdirectory live (serve mode),
    which must win over any page already on disk, or serving would hand
    back yesterday's static snapshot."""
    out = []

    def section(title, rows):
        if rows:
            out.append(f"<h2>{title}</h2>\n<ul>\n"
                       + "\n".join(f"<li>{r}</li>" for r in rows) + "\n</ul>")

    rows = []
    for sd in model["subdirs"]:
        name = _escape(sd["name"])
        if dir_link is not None:
            rows.append(f'<a href="{_escape(dir_link(sd["path"]))}">{name}/</a>')
        elif sd["page"] is not None:
            rows.append(f'<a href="{_escape(link(sd["page"]))}">{name}/</a>')
        else:
            rows.append(f"{name}/ {_recipe(sd['path'])}")
    section("Subdirectories", rows)

    rows = []
    for v in model["viewable"]:
        label = _escape(f"{v['name']} — {v['label']}")
        kind = f'<span class="kind">{_escape(v["kind"])}</span>'
        if v["out"] is not None:
            rows.append(f'<a href="{_escape(link(v["out"]))}">{label}</a>{kind}')
        elif build_link is not None:
            rows.append(f'<a href="{_escape(build_link(v["path"]))}">'
                        f'{label}</a>{kind}')
        else:
            rows.append(f"{label}{kind} {_recipe(v['path'])}")
    section("Viewable here", rows)

    rows = [f'<a href="{_escape(link(b["path"]))}">{_escape(b["label"])}</a>'
            for b in model["bundles"]]
    section("Existing bundles", rows)

    rows = [f'<a href="{_escape(link(a["path"]))}">{_escape(a["label"])}</a>'
            f'<span class="kind">{_escape(_agent_caption(a))}</span>'
            for a in model.get("agents", [])]
    section("Agent-generated", rows)
    return "\n".join(out)


def render_browse(model, link, build_link=None, dir_link=None):
    """Render a browse model to HTML. Data-free by construction: filenames
    and entity labels only, never imaging data or participant values."""
    return (_BROWSE_TEMPLATE
            .replace("__TITLE__", _escape(model["rel"]))
            .replace("__SIGNATURE__", BROWSE_SIGNATURE)
            .replace("__COMMAND__", _escape(model["command"]))
            .replace("__DATE__", model["date"])
            .replace("__SECTIONS__",
                     _browse_sections(model, link, build_link, dir_link)))


def write_browse(directory, roots, opts=None):
    """Write <viz dir>/browse.html for *directory* and return its path. Not
    index.html (that stays the bundle pulldown) and never a *desc-viewer*
    name (the deface report globs those)."""
    directory = Path(directory)
    model = browse_model(directory, roots, opts)
    viz = viz_dir_for(directory)
    html = render_browse(model, lambda p: os.path.relpath(p, viz))
    page = viz / BROWSE_PAGE
    if not page.exists() or page.read_text() != html:
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(html)
    return page


def open_view(path, fragment=None):
    """Open a local file in a browser. Returns a note, or None when nothing
    could open it (the caller prints the path and a hint)."""
    uri = Path(path).resolve().as_uri()     # helpers want a URI, not a path
    if fragment:
        uri += "#" + fragment
    return open_url(uri)


def open_url(uri):
    """Try $BROWSER, then the platform opener: `open` on macOS (a GUI is
    always there — no $DISPLAY to gate on), xdg-open under a display
    elsewhere."""
    browser = os.environ.get("BROWSER")
    if browser:
        try:
            subprocess.Popen(shlex.split(browser) + [uri],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return "opened via $BROWSER"
        except OSError:
            pass
    if sys.platform == "darwin" and shutil.which("open"):
        subprocess.Popen(["open", uri], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return "opened via open"
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


# ---------------------------------------------------------------------------
# serve mode — the same browse model, generated per request, with live
# build links and Range support (Chrome media elements stall without 206s)
# ---------------------------------------------------------------------------

BUILD_PATH = "/__build"


class _BrowseHandler(SimpleHTTPRequestHandler):
    """Serves the dataset tree: directories as browse pages generated in
    memory (never written, so never stale), files with Range support."""

    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".m4a": "audio/mp4"}

    def log_message(self, fmt, *args):
        sys.stderr.write("mmmview serve: %s\n" % (fmt % args))

    # -- path validation, on every request ----------------------------------

    def safe_path(self, urlpath):
        """(Path, None) for a request inside the served tree, else
        (None, reason). Containment is checked BEFORE symlink resolution:
        the staged tree symlinks internally by design, so
        resolve-then-check would refuse its own layout."""
        rel = urllib.parse.unquote(urlpath.split("?", 1)[0]).lstrip("/")
        if ".." in Path(rel).parts:
            return None, "path traversal is refused"
        root = Path(self.server.mmm_root)
        full = Path(os.path.normpath(str(root / rel))) if rel else root
        if full != root and root not in full.parents:
            return None, "outside the served root"
        if (SOURCEDATA_DIR in full.parts
                or SOURCEDATA_DIR in full.resolve().parts):
            return None, (f"{SOURCEDATA_DIR} is outside mmmview's scope "
                          "(PII tree)")
        return full, None

    def url_for(self, path):
        """URL path for a file inside the served tree ('' when outside)."""
        root = Path(self.server.mmm_root)
        try:
            rel = Path(path).relative_to(root)
        except ValueError:
            return ""
        return "/" + urllib.parse.quote(str(rel))

    def build_url_for(self, path):
        rel = self.url_for(path).lstrip("/")
        return f"{BUILD_PATH}?path={rel}" if rel else ""

    # -- routing ------------------------------------------------------------

    def do_GET(self):
        split = urllib.parse.urlsplit(self.path)
        if split.path == BUILD_PATH:
            return self.serve_build(split.query)
        path, reason = self.safe_path(split.path)
        if reason:
            return self.send_error(403, reason)
        if path.is_dir():
            return self.serve_dir(path)
        return super().do_GET()

    def serve_dir(self, path):
        """A directory that classifies goes to its bundle (building it if
        absent, through the same endpoint the links use); anything else
        renders the browse model live."""
        plans = self._plans(path)
        if plans:
            target = self._existing_output(path, plans)
            return self.redirect(target or self.build_url_for(path))
        model = browse_model(path, self.server.mmm_roots,
                             self.server.mmm_opts)
        body = render_browse(model, self.url_for,
                             build_link=self.build_url_for,
                             dir_link=self.url_for).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_build(self, query):
        rel = (urllib.parse.parse_qs(query).get("path") or [""])[0]
        path, reason = self.safe_path("/" + rel)
        if reason:
            return self.send_error(403, reason)
        try:
            targets = classify(path)
            outs = []
            for t in targets:
                plan = resolve(t, self.server.mmm_roots, self.server.mmm_opts)
                out, _ = render(plan)
                outs.append(out)
        except (Unplaceable, RenderError) as exc:
            # 422: the request was well formed, the thing it names cannot be
            # built. Never a traceback page.
            return self.send_error(422, str(exc).splitlines()[0])
        for d in sorted({o.parent for o in outs}):
            write_index(d)
        url = self.url_for(outs[0]) if outs else ""
        if not url:
            return self.send_error(422, f"{path} built outside the served "
                                        "root; open it directly")
        return self.redirect(url)

    def redirect(self, url):
        self.send_response(303)
        self.send_header("Location", url)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _plans(self, path):
        try:
            return [resolve(t, self.server.mmm_roots, self.server.mmm_opts)
                    for t in classify(path)]
        except Unplaceable:
            return []

    def _existing_output(self, path, plans):
        """The URL to send a classifiable directory to, or '' when nothing
        is built yet."""
        idx = viz_dir_for(path) / "index.html"
        if len(plans) > 1 and idx.exists():
            return self.url_for(idx)
        if all(p.out.exists() for p in plans):
            return self.url_for(plans[0].out)
        return ""

    # -- files: Range, ported from the stimfeat-viewer prototype ------------

    def send_head(self):
        m = re.match(r"bytes=(\d+)-(\d*)$", self.headers.get("Range") or "")
        if not m:
            return super().send_head()
        path = self.translate_path(self.path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(f.fileno()).st_size
        start = int(m.group(1))
        end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        if start >= size:
            f.close()
            self.send_error(416)
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        self._range_left = None
        while left > 0:
            chunk = source.read(min(65536, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)


def make_server(root, roots, opts, bind="127.0.0.1", port=8471):
    """A server over *root*. Separate from serve_main so tests can drive it
    on an ephemeral port."""
    root = Path(os.path.abspath(str(root)))
    handler = functools.partial(_BrowseHandler, directory=str(root))
    srv = ThreadingHTTPServer((bind, port), handler)
    srv.mmm_root, srv.mmm_roots, srv.mmm_opts = root, roots, opts
    return srv


def serve_main(argv):
    ap = argparse.ArgumentParser(
        prog="mmmview serve",
        description="serve the dataset tree: browse pages per request, "
                    "build-on-demand links, HTTP Range for media")
    ap.add_argument("root", nargs="?", help="directory to serve "
                    "(default: the configured BIDS root)")
    ap.add_argument("--port", type=int, default=8471)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--deriv-root", help="override config derivatives root")
    args = ap.parse_args(argv)

    roots = load_roots(args.deriv_root)
    root = Path(args.root) if args.root else roots.bids
    refusal = sourcedata_refusal(root)
    if refusal:
        print(f"mmmview: {refusal}", file=sys.stderr)
        return EXIT_UNPLACEABLE
    if not Path(root).is_dir():
        print(f"mmmview: not a directory: {root}", file=sys.stderr)
        return EXIT_UNPLACEABLE
    srv = make_server(root, roots, Opts(), args.bind, args.port)
    port = srv.server_address[1]
    url = f"http://{args.bind}:{port}/"
    print(f"serving {Path(os.path.abspath(str(root)))} at {url}")
    print(f"from your laptop: ssh -N -L {port}:localhost:{port} "
          "<user>@login.talapas.uoregon.edu   # then open " + url)
    if not args.no_open:
        open_url(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nmmmview: stopped")
    finally:
        srv.server_close()
    return 0


# ---------------------------------------------------------------------------
# reap — remove bundles whose provenance no longer matches their inputs
# ---------------------------------------------------------------------------

VIZ_DIRS = ("viz", "qc")        # qc/ is the legacy location, still swept
KEY_MARK = "mmmview-key:"


def _read_head(path, limit=1 << 20):
    """Bundles embed their provenance note near the end, but they are tens
    of MB of base64 — read the whole file the same forgiving way is_current
    does, not a slice, or a key would be missed."""
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _viz_dirs_under(path):
    path = Path(path)
    found = []
    if path.name in VIZ_DIRS and path.is_dir():
        found.append(path)
    for root, dirs, _ in os.walk(path):
        dirs.sort()
        for d in dirs:
            if d in VIZ_DIRS:
                found.append(Path(root) / d)
    return sorted(set(found))


def _data_dirs_for(viz):
    """Every directory whose bundles land in this viz dir.

    NOT just `viz.parent`: `viz_dir_for` walks UP to the nearest sub-##
    ancestor, so a subject's maps usually sit deeper (<sub>/func,
    <sub>/ses-##/func) while their bundles all land in <sub>/viz. Asking
    only the parent finds no maps there, which would mark every live
    bundle an orphan — and delete it under --yes. Matched on the anchor
    (`viz.parent`) rather than the dir name so the legacy qc/ location is
    covered by the same walk.
    """
    anchor = viz.parent
    if not anchor.is_dir():
        return []
    found = []
    for root, dirs, _ in os.walk(anchor):
        dirs[:] = sorted(d for d in dirs if d not in VIZ_DIRS)
        d = Path(root)
        if viz_dir_for(d).parent == anchor:
            found.append(d)
    return found


def _claims_for(viz, roots, opts):
    """What SHOULD be in this viz dir: {bundle name: Plan}, recomputed from
    every data directory it serves."""
    claims = {}

    def add(target):
        try:
            plan = resolve(target, roots, opts)
        except Unplaceable:
            return
        claims[plan.out.name] = plan

    for data_dir in _data_dirs_for(viz):
        try:
            for t in classify(data_dir):
                add(t)
        except Unplaceable:
            pass
        # ...and from each file on its own: `mmmview <one map>` is a
        # supported call, and the bundle it writes carries a narrower name
        # (contrast-, stat-) than the directory-level merge claims. Without
        # this pass those bundles look unclaimed and --yes would delete
        # current output.
        for child in sorted(data_dir.iterdir()):
            if not child.is_file():
                continue
            if not child.name.endswith(MAP_EXTS + SURF_EXTS + FEATURE_EXTS):
                continue
            try:
                for t in classify(child):
                    add(t)
            except Unplaceable:
                continue
    return claims


def reap_scan(path, roots, opts=None, paths_from=None):
    """Classify every candidate under *path*. Returns (rows, touched) where
    a row is (verdict, Path, size) and touched is the viz dirs holding at
    least one stale file. Nothing is deleted here."""
    opts = opts or Opts()
    if paths_from:
        starts = [Path(l.strip()) for l in Path(paths_from).read_text().splitlines()
                  if l.strip()]
    else:
        starts = [Path(path)]
    rows, touched = [], set()
    for start in starts:
        for viz in _viz_dirs_under(start):
            claims = _claims_for(viz, roots, opts)
            for f in sorted(viz.iterdir()):
                if not (f.is_file() and "_desc-viewer" in f.name
                        and f.name.endswith(".html")):
                    continue
                size = f.stat().st_size
                text = _read_head(f)
                if KEY_MARK not in text:
                    # not signed by mmmview. A features/movies dashboard the
                    # data still claims is ours but unverifiable (its
                    # currency is mtime-based); anything else is simply not
                    # ours — deface montages live in qc/ too — and stays
                    # invisible to this sweep.
                    plan = claims.get(f.name)
                    if plan is not None and plan.renderer in ("features",
                                                              "movies"):
                        rows.append(("unverifiable (no key sidecar)", f, size))
                    continue
                plan = claims.get(f.name)
                if plan is None:
                    rows.append(("stale-orphan", f, size))
                    touched.add(viz)
                elif f"{KEY_MARK} {plan.key}" in text:
                    rows.append(("keep", f, size))
                else:
                    rows.append(("stale-key", f, size))
                    touched.add(viz)
    return rows, touched


def _regenerate(viz, roots, opts):
    """Rewrite the pages that point at what was just removed. Only pages
    that already exist — a reap builds nothing new."""
    notes = []
    if (viz / "index.html").exists():
        if write_index(viz) is None:
            notes.append(f"note: {viz / 'index.html'} left in place; fewer "
                         "than two bundles remain, so it has no pulldown to "
                         "regenerate — delete it by hand if you want it gone")
    if (viz / BROWSE_PAGE).exists():
        write_browse(viz.parent, roots, opts)
    return notes


def reap_main(argv):
    ap = argparse.ArgumentParser(
        prog="mmmview reap",
        description="remove viewer bundles whose provenance key no longer "
                    "matches their inputs. Dry run unless --yes.")
    ap.add_argument("path", type=Path, help="subtree to sweep")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete (default: report only)")
    ap.add_argument("--paths-from", help="file of paths, one per line, to "
                    "sweep instead of walking PATH (the catalog-diff hook)")
    ap.add_argument("--deriv-root", help="override config derivatives root")
    args = ap.parse_args(argv)

    refusal = sourcedata_refusal(args.path)
    if refusal:
        print(f"mmmview: {refusal}", file=sys.stderr)
        return EXIT_UNPLACEABLE
    roots = load_roots(args.deriv_root)
    opts = Opts()
    rows, touched = reap_scan(args.path, roots, opts, args.paths_from)

    stale = [r for r in rows if r[0].startswith("stale")]
    for verdict, f, size in rows:
        print(f"{verdict}\t{f}\t{size}")
    freed = sum(size for _, _, size in stale)
    print(f"-- {len(rows)} candidates, {len(stale)} stale, "
          f"{freed / 1e6:.1f} MB "
          f"{'deleted' if args.yes else 'reclaimable (dry run; --yes to '
             'delete)'}")
    if not args.yes:
        return 0
    for _, f, _ in stale:
        f.unlink()
    for viz in sorted(touched):
        for note in _regenerate(viz, roots, opts):
            print(note)
    return 0


VERBS = {"serve": serve_main, "reap": reap_main}


def main(argv=None):
    # verb dispatch BEFORE argparse: `mmmview PATH` must keep parsing
    # exactly as it does today, so the parser never becomes subcommands
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in VERBS:
        return VERBS[argv[0]](argv[1:])

    ap = argparse.ArgumentParser(
        prog="mmmview", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="a map, a directory of maps, "
                    "or a Contract B feature file")
    ap.add_argument("--no-open", action="store_true",
                    help="build only; print the output path")
    ap.add_argument("--out-dir", help="override <sub-##>/viz/")
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
    ap.add_argument("--films-dir", help="movies tables: per-film media tree "
                    "(default: the movies/ directory beside the tables)")
    args = ap.parse_args(argv)

    refusal = sourcedata_refusal(args.path)
    if refusal:
        print(f"mmmview: {refusal}", file=sys.stderr)
        return EXIT_UNPLACEABLE

    opts = Opts(args.underlay, args.mesh, args.surf, args.r2_floor,
                args.out_dir, args.force, args.films_dir)
    roots = load_roots(args.deriv_root)
    try:
        targets = classify(args.path)
        plans = [resolve(t, roots, opts) for t in targets]
    except Unplaceable as exc:
        # a FILE keeps the strict contract: exit 2, message naming the flag.
        # a DIRECTORY is total — what cannot be viewed is browsed instead.
        if not Path(args.path).is_dir():
            print(f"mmmview: cannot place {args.path}: {exc}", file=sys.stderr)
            return EXIT_UNPLACEABLE
        print(f"mmmview: note: {exc}", file=sys.stderr)
        page = write_browse(Path(args.path), roots, opts)
        print(f"browse {page}")
        if not args.no_open:
            note = open_view(page)
            print(note if note else OPEN_HINT)
        return 0

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

    # a viz dir holding several bundles gets a pulldown index; open that
    # (preselecting this run's first bundle) instead of "the first of N"
    index = None
    bundle_outs = [p.out for p in plans if p.renderer in ("volume", "surface")]
    for d in sorted({o.parent for o in bundle_outs}):
        got = write_index(d)
        if got:
            index = got
            print(f"index {got}")

    if not args.no_open and outs:
        if index:
            note = open_view(index, fragment=bundle_outs[0].name)
        else:
            if len(outs) > 1:
                print(f"opening the first of {len(outs)} outputs")
            note = open_view(outs[0])
        print(note if note else OPEN_HINT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

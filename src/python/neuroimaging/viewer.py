"""Self-contained NiiVue brain-map viewer bundles (workbench brain-viewer).

One builder per case:

    build_volume_viewer(underlay, overlays, out, ...)   # NIfTI underlay + stat maps
    build_surface_viewer(mesh, layers, out, ...)        # FreeSurfer mesh + per-vertex maps

Each writes ONE .html file with the pinned NiiVue UMD (``vendor/``) and every
volume/mesh/layer base64-inlined — no network, opens over ``file://``, and the
in-page "Save .nvd" button exports a NiiVue document for FreeBrowse. Bundles
are findings: they live on GPFS beside the data (or travel as claude.ai
artifacts, 16 MB cap — surfaces per-hemisphere), never in a repo.

Masking convention (measured, workbench brain-viewer log 2026-08-27):
NiiVue hides a voxel/vertex only when its value is strictly below ``cal_min``
(volumes via ``isAlphaClipDark``, mesh layers via ``isTransparentBelowCalMin``).
NaN fails every comparison and renders as garbage color. So invalid samples
are rewritten to the sentinel ``MASK_SENTINEL`` (-1) and stat maps keep
``cal_min >= 0``: real 0-valued data (polar angle at 0 deg) survives, the
sentinel never shows. Helpers ``masked_volume``/``masked_shape_gii`` apply
exactly this; do not hand nilearn the same files (it thresholds on |value| —
opposite trap, same log entry).
"""

import base64
import gzip
import io
import json
from pathlib import Path

import numpy as np

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
NIIVUE_JS = VENDOR_DIR / "niivue.umd.js"
TEMPLATE = VENDOR_DIR / "viewer_template.html"

MASK_SENTINEL = -1.0

# NiiVue infers every format from the file extension, so embedded payloads
# must carry a recognized name (FreeSurfer meshes: the bare suffix, e.g.
# ``lh.inflated``; morph data ``.curv``; per-vertex GIFTI ``.shape.gii``).


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _nifti_bytes(img) -> bytes:
    """Serialize a nibabel NIfTI image to gzipped .nii.gz bytes in memory."""
    buf = io.BytesIO()
    # mtime=0 keeps output byte-identical across runs (bundle diffing)
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(img.to_bytes())
    return buf.getvalue()


def _script_safe(js: str) -> str:
    """Neutralize sequences that break an inline <script> block.

    ``<!--`` followed later by ``<script`` puts HTML parsers into the
    double-escaped script state, where the real ``</script>`` close tag is
    ignored. Both occur in NiiVue's UMD as JS *string literals*, so a no-op
    backslash escape changes the page bytes without changing the JS value.
    """
    return (js.replace("<!--", "<\\!--")
              .replace("<script", "<\\script")
              .replace("</script", "<\\/script"))


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------

def masked_volume(path, mask, floor):
    """Return (nibabel image) copy of *path* with sentinel where the mask
    volume is not strictly above *floor* (or where the map itself is NaN)."""
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    keep = np.asarray(nib.load(str(mask)).dataobj, dtype=np.float32) > floor
    keep &= np.isfinite(data)
    out = np.where(keep, data, np.float32(MASK_SENTINEL))
    new = nib.Nifti1Image(out, img.affine, img.header)
    new.header.set_data_dtype(np.float32)
    return new


def masked_shape_values(path, mask_path=None, floor=None):
    """Per-vertex values from a .shape.gii with NaN (and optionally vertices
    whose *mask_path* value is not strictly above *floor*) set to sentinel."""
    import nibabel as nib
    vals = nib.load(str(path)).darrays[0].data.astype(np.float32)
    keep = np.isfinite(vals)
    if mask_path is not None:
        mvals = nib.load(str(mask_path)).darrays[0].data.astype(np.float32)
        keep &= np.isfinite(mvals) & (mvals > floor)
    return np.where(keep, vals, np.float32(MASK_SENTINEL))


def shape_gii_bytes(values, structure="CortexLeft") -> bytes:
    import nibabel as nib
    da = nib.gifti.GiftiDataArray(np.asarray(values, dtype=np.float32),
                                  intent="NIFTI_INTENT_SHAPE",
                                  datatype="NIFTI_TYPE_FLOAT32")
    img = nib.gifti.GiftiImage(darrays=[da])
    img.meta["AnatomicalStructurePrimary"] = structure
    return img.to_bytes()


# ---------------------------------------------------------------------------
# spec normalization
# ---------------------------------------------------------------------------

_NUMERIC_KEYS = ("cal_min", "cal_max", "opacity")


def _norm_spec(spec, defaults):
    out = dict(defaults)
    out.update(spec)
    for k in _NUMERIC_KEYS:
        if k in out and out[k] is not None:
            out[k] = float(out[k])
    return out


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_volume_viewer(underlay, overlays, out, title="brain viewer",
                        notes=""):
    """underlay: path or dict spec; overlays: list of dict specs.

    Overlay spec keys: ``path`` OR ``image`` (in-memory nibabel image, e.g.
    from :func:`masked_volume`), ``name``, ``colormap``, ``cal_min``,
    ``cal_max``, ``opacity``, ``visible``, ``angle_legend``.
    """
    if not isinstance(underlay, dict):
        underlay = {"path": underlay}
    underlay = _norm_spec(underlay, {"name": None, "colormap": "gray",
                                     "cal_min": None, "cal_max": None})

    def payload(spec):
        if spec.get("image") is not None:
            return _nifti_bytes(spec["image"])
        return Path(spec["path"]).read_bytes()

    def nii_name(spec, i):
        n = spec.get("name") or (Path(spec["path"]).name if spec.get("path")
                                 else f"overlay{i}")
        return n if n.endswith((".nii", ".nii.gz")) else n + ".nii.gz"

    vols = [{
        "name": nii_name(underlay, 0), "label": "underlay",
        "base64": _b64(payload(underlay)),
        "colormap": underlay["colormap"],
        "cal_min": underlay["cal_min"], "cal_max": underlay["cal_max"],
        "opacity": 1.0, "isUnderlay": True, "angle_legend": False,
    }]
    for i, spec in enumerate(overlays):
        spec = _norm_spec(spec, {"colormap": "viridis", "cal_min": 0.0,
                                 "cal_max": None, "opacity": 0.8,
                                 "visible": i == 0, "angle_legend": False})
        vols.append({
            "name": nii_name(spec, i + 1),
            "label": spec.get("label") or Path(nii_name(spec, i + 1)).name,
            "base64": _b64(payload(spec)),
            "colormap": spec["colormap"],
            "cal_min": spec["cal_min"], "cal_max": spec["cal_max"],
            "opacity": spec["opacity"], "visible": bool(spec["visible"]),
            "isUnderlay": False, "angle_legend": bool(spec["angle_legend"]),
        })

    return _render({"mode": "volume", "title": title, "notes": notes,
                    "volumes": vols, "meshes": []}, out)


def build_surface_viewer(mesh, layers, out, title="brain viewer", notes=""):
    """mesh: path to a FreeSurfer surface (lh.inflated, rh.pial, ...);
    layers: list of dict specs, drawn in order (put the curv/sulc shading
    layer first).

    Layer spec keys: ``path`` OR ``values`` (per-vertex array, e.g. from
    :func:`masked_shape_values`, serialized as .shape.gii), ``name``,
    ``colormap``, ``cal_min``, ``cal_max``, ``opacity``, ``visible``
    (False loads at opacity 0, toggleable), ``shade`` (True = always-on
    shading, exempt from the exclusive stat-layer toggle), ``angle_legend``.
    """
    mesh = Path(mesh)
    structure = "CortexRight" if mesh.name.startswith("rh") else "CortexLeft"

    jlayers = []
    for i, spec in enumerate(layers):
        spec = _norm_spec(spec, {"colormap": "viridis", "cal_min": 0.0,
                                 "cal_max": None, "opacity": 0.7,
                                 "visible": None, "shade": False,
                                 "angle_legend": False})
        if spec.get("values") is not None:
            data = shape_gii_bytes(spec["values"], structure)
            name = spec.get("name") or f"layer{i}.shape.gii"
            if not name.endswith(".gii"):
                name += ".shape.gii"
        else:
            data = Path(spec["path"]).read_bytes()
            name = spec.get("name") or Path(spec["path"]).name
        if spec["visible"] is None:
            # default: shading always on, first stat layer on
            spec["visible"] = spec["shade"] or not any(
                not l.get("shade") for l in layers[:i])
        jlayers.append({
            "name": name, "label": spec.get("label") or name,
            "base64": _b64(data),
            "colormap": spec["colormap"],
            "cal_min": spec["cal_min"], "cal_max": spec["cal_max"],
            "opacity": spec["opacity"], "visible": bool(spec["visible"]),
            "shade": bool(spec["shade"]),
            "angle_legend": bool(spec["angle_legend"]),
        })

    meshes = [{"name": mesh.name, "base64": _b64(mesh.read_bytes()),
               "layers": jlayers}]
    return _render({"mode": "surface", "title": title, "notes": notes,
                    "volumes": [], "meshes": meshes}, out)


def _render(config, out):
    html = TEMPLATE.read_text()
    html = html.replace("__TITLE__", config["title"])
    # "</" only ever appears inside JSON string values, where "\/" is a
    # no-op escape — keeps a stray "</script>" in a label from closing the tag
    html = html.replace("__CONFIG_JSON__",
                        json.dumps(config).replace("</", "<\\/"))
    html = html.replace("__NIIVUE_JS__", _script_safe(NIIVUE_JS.read_text()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out

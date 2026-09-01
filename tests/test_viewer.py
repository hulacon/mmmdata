"""Viewer bundle builder (src/python/neuroimaging/viewer.py) on synthetic data.

The masking/sentinel conventions under test are the measured NiiVue behaviors
from the brain-viewer workbench log: strictly-below-cal_min hides a sample,
NaN survives every comparison and renders garbage, so invalid samples must
leave the builder as MASK_SENTINEL (-1) with cal_min >= 0.
"""

import base64
import gzip
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

# src/python is on sys.path via the repo-root conftest; a second insert here
# would trip test_portability's bootstrap-idempotence guard
from neuroimaging import viewer  # noqa: E402

NIIVUE_SHA256 = "43d966b756982173fdd0cd37736c348b4db02238cdcbdd21fb16714963aad3fa"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vol_pair(tmp_path):
    """Tiny stat map + mask volume. Values chosen to catch the two traps:
    a real 0.0 inside the mask (must survive) and a NaN (must sentinel)."""
    rng = np.random.default_rng(7)
    data = rng.uniform(0.0, 360.0, (4, 4, 3)).astype(np.float32)
    data[0, 0, 0] = 0.0          # real value at the cal_min boundary
    data[1, 1, 1] = np.nan       # invalid sample inside the mask
    mask = np.zeros((4, 4, 3), np.float32)
    mask[:2] = 50.0              # kept half (mask value > floor 10)
    stat = tmp_path / "stat.nii.gz"
    mvol = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), stat)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mvol)
    return stat, mvol, data


@pytest.fixture
def surf_files(tmp_path):
    """Minimal FreeSurfer mesh + curv + shape.gii with a NaN vertex."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    mesh = tmp_path / "lh.inflated"
    nib.freesurfer.write_geometry(str(mesh), verts, faces)
    curv = tmp_path / "lh.curv"
    nib.freesurfer.write_morph_data(str(curv), np.array([0.5, -0.5, 0.2, -0.2]))
    vals = np.array([0.0, 180.0, np.nan, 90.0], np.float32)
    r2 = np.array([50.0, 5.0, 50.0, 50.0], np.float32)
    gii = tmp_path / "lh.stat.shape.gii"
    r2gii = tmp_path / "lh.R2.shape.gii"
    for path, v in ((gii, vals), (r2gii, r2)):
        da = nib.gifti.GiftiDataArray(v, intent="NIFTI_INTENT_SHAPE")
        nib.save(nib.gifti.GiftiImage(darrays=[da]), str(path))
    return mesh, curv, gii, r2gii


def embedded_payloads(html):
    """Decode every base64 payload out of a bundle's config JSON."""
    cfg = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html).group(1)
                     .replace("<\\/", "</"))
    out = {}
    for v in cfg.get("volumes", []):
        out[v["label"]] = base64.b64decode(v["base64"])
    for m in cfg.get("meshes", []):
        out[m["name"]] = base64.b64decode(m["base64"])
        for l in m["layers"]:
            out[l["label"]] = base64.b64decode(l["base64"])
    return cfg, out


def nifti_from_bytes(raw):
    return nib.Nifti1Image.from_bytes(gzip.decompress(raw))


# ---------------------------------------------------------------------------
# vendored asset pins
# ---------------------------------------------------------------------------

def test_vendored_niivue_pinned():
    digest = hashlib.sha256(viewer.NIIVUE_JS.read_bytes()).hexdigest()
    assert digest == NIIVUE_SHA256, (
        "vendor/niivue.umd.js does not match the pinned build — update the "
        "hash here and in vendor/VENDOR.md together, deliberately")


def test_template_has_required_tokens():
    text = viewer.TEMPLATE.read_text()
    for token in ("__TITLE__", "__CONFIG_JSON__", "__NIIVUE_JS__"):
        assert token in text


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------

def test_masked_volume_sentinel_and_zero_survival(vol_pair):
    stat, mask, data = vol_pair
    out = np.asarray(viewer.masked_volume(stat, mask, 10.0).dataobj)
    assert out[0, 0, 0] == 0.0                      # boundary value kept
    assert out[1, 1, 1] == viewer.MASK_SENTINEL     # NaN -> sentinel
    assert (out[2:] == viewer.MASK_SENTINEL).all()  # below-floor half masked
    finite = np.isfinite(data[:2])
    assert np.array_equal(out[:2][finite], data[:2][finite])
    assert not np.isnan(out).any()                  # NaN never leaves


def test_masked_shape_values(surf_files):
    _, _, gii, r2gii = surf_files
    out = viewer.masked_shape_values(gii, r2gii, 10.0)
    assert out[0] == 0.0                            # real 0 survives
    assert out[1] == viewer.MASK_SENTINEL           # R2 below floor
    assert out[2] == viewer.MASK_SENTINEL           # NaN vertex
    assert out[3] == 90.0
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------

def test_volume_bundle_self_contained(vol_pair, tmp_path):
    stat, mask, _ = vol_pair
    out = viewer.build_volume_viewer(
        stat, [{"path": stat, "label": "angle", "colormap": "hsv",
                "cal_min": 0, "cal_max": 360, "angle_legend": True,
                "image": viewer.masked_volume(stat, mask, 10.0)}],
        tmp_path / "v.html", title="t", notes="n")
    html = out.read_text()
    assert 'src="http' not in html and 'href="http' not in html
    cfg, payloads = embedded_payloads(html)
    assert cfg["mode"] == "volume" and len(cfg["volumes"]) == 2
    assert cfg["volumes"][0]["isUnderlay"]
    assert cfg["volumes"][1]["cal_min"] == 0.0      # numeric, not string
    emb = np.asarray(nifti_from_bytes(payloads["angle"]).dataobj)
    assert emb[1, 1, 1] == viewer.MASK_SENTINEL     # masked copy embedded,
    assert not np.isnan(emb).any()                  # not the raw file


def test_surface_bundle_layers_and_sentinel(surf_files, tmp_path):
    mesh, curv, gii, r2gii = surf_files
    vals = viewer.masked_shape_values(gii, r2gii, 10.0)
    out = viewer.build_surface_viewer(
        mesh,
        [{"path": curv, "label": "curvature", "shade": True,
          "colormap": "gray", "cal_min": 0.3, "cal_max": 0.8},
         {"values": vals, "name": "stat.shape.gii", "label": "stat",
          "cal_min": 0, "cal_max": 360}],
        tmp_path / "s.html", title="t")
    cfg, payloads = embedded_payloads(out.read_text())
    assert cfg["mode"] == "surface"
    layers = cfg["meshes"][0]["layers"]
    assert layers[0]["shade"] and not layers[1]["shade"]
    assert layers[1]["visible"]                     # first stat layer on
    assert layers[1]["name"].endswith(".gii")       # NiiVue needs the ext
    emb = nib.gifti.GiftiImage.from_bytes(payloads["stat"]).darrays[0].data
    assert emb[1] == viewer.MASK_SENTINEL and not np.isnan(emb).any()
    assert payloads["lh.inflated"] == mesh.read_bytes()


def test_script_safe_neutralizes_inline_breakers():
    js = 'a.indexOf("<!--"); b("<script>"); c("</script>")'
    safe = viewer._script_safe(js)
    for seq in ("<!--", "<script", "</script"):
        assert seq not in safe
    # the escapes are no-ops for JS string values
    assert safe == ('a.indexOf("<\\!--"); b("<\\script>"); c("<\\/script>")')


def test_render_deterministic(vol_pair, tmp_path):
    stat, mask, _ = vol_pair
    args = dict(underlay=stat, overlays=[{"path": stat}], title="t")
    a = viewer.build_volume_viewer(out=tmp_path / "a.html", **args).read_bytes()
    b = viewer.build_volume_viewer(out=tmp_path / "b.html", **args).read_bytes()
    assert a == b


# ---------------------------------------------------------------------------
# CLI spec parsing
# ---------------------------------------------------------------------------

def test_cli_spec_parsing(tmp_path):
    scripts = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_brain_viewer as cli
    p = tmp_path / "x.nii.gz"
    p.write_bytes(b"")
    spec = cli.parse_spec(f"{p}:colormap=hsv,cal_min=0,shade=1,label=a b")
    assert spec == {"path": str(p), "colormap": "hsv", "cal_min": "0",
                    "shade": True, "label": "a b"}
    assert cli.parse_spec(str(p)) == {"path": str(p)}
    with pytest.raises(SystemExit):
        cli.parse_spec(f"{p}:bogus=1")
    with pytest.raises(SystemExit):
        cli.parse_spec(str(tmp_path / "missing.nii.gz"))

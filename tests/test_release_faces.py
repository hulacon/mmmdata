"""release.faces: no anatomical face anywhere in a tree.

Everything is synthetic: a 40^3 "head" whose face is a bright slab in the
anterior-inferior corner, a recorded face mask over that slab, and copies of
the head on other grids. Subject labels are placeholders.
"""

import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from release import faces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

SHAPE = (40, 40, 40)


@pytest.fixture(autouse=True)
def _small_core(monkeypatch):
    """The synthetic face mask is a few thousand voxels; the production
    floor (sized for 1 mm heads) would call every eroded core out of view."""
    monkeypatch.setattr(faces, "MIN_CORE", 50)
AFF = np.diag([1.0, 1.0, 1.0, 1.0])


def _face_mask(shape=SHAPE):
    m = np.zeros(shape, bool)
    m[6:34, 20:40, 0:14] = True          # anterior (high j), inferior (low k)
    return m


def _head(faced: bool, shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    d = np.zeros(shape, np.float32)
    d[4:36, 4:38, 2:38] = rng.uniform(200, 400, size=(32, 34, 36))   # head
    d[12:28, 10:30, 16:34] += 300                                     # brain
    if not faced:
        d[_face_mask(shape)] = 0
    return d


def _save(path: Path, data, affine=AFF):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(np.asarray(data), affine).to_filename(str(path))
    return path


def _sha(p):
    return faces._sha256(p)


class Env:
    def __init__(self, tmp: Path):
        self.bids = tmp / "work"
        self.source = tmp / "source"
        self.bids.mkdir()
        self.source.mkdir()

    def record(self, sub: str, files: list[Path], mni=True):
        """Write face masks (T1w leg on the 40^3 grid, MNI leg on a 2 mm
        grid) and a provenance entry per raw file."""
        d = self.source / "anat_defacing" / f"sub-{sub}"
        (d / "masks").mkdir(parents=True, exist_ok=True)
        legs = {"T1w": {"facemask": str(_save(d / "masks" / "T1w_facemask.nii.gz", _face_mask().astype(np.uint8)))}}
        if mni:
            mni_aff = np.diag([2.0, 2.0, 2.0, 1.0])
            mm = np.zeros((20, 20, 20), np.uint8)
            mm[2:18, 8:20, 0:7] = 1
            legs["MNI"] = {"facemask": str(_save(d / "masks" / "MNI_facemask.nii.gz", mm, mni_aff))}
        prov = {"subject": f"sub-{sub}", "legs": legs,
                "files": {str(f): {"sha_after": _sha(f), "leg": "T1w"} for f in files}}
        (d / "provenance.json").write_text(json.dumps(prov))


def _env(tmp_path, faced_raw=False):
    env = Env(tmp_path)
    raw = _save(env.bids / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz", _head(faced_raw))
    env.record("01", [raw])
    return env


def _audit(env, root=None, **kw):
    return faces.audit(root or env.bids, env.bids, env.source, **kw)


def _kinds(res):
    return sorted((f.kind, f.path) for f in res.findings)


# --------------------------------------------------------------------------- #
# content check                                                               #
# --------------------------------------------------------------------------- #
def test_defaced_tree_passes(tmp_path):
    env = _env(tmp_path)
    res = _audit(env)
    assert res.findings == []
    (sc,) = res.scores
    assert (sc.method, sc.verdict) == ("grid", "ok") and sc.signal == 0.0


def test_faced_derivative_on_the_same_grid_is_flagged(tmp_path):
    env = _env(tmp_path)
    _save(env.bids / "derivatives" / "pipe" / "sub-01" / "anat" / "sub-01_desc-preproc_T1w.nii.gz", _head(True))
    res = _audit(env)
    assert _kinds(res) == [("face", "derivatives/pipe/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz")]


def test_unlisted_whole_head_volume_on_another_grid_is_scored_by_world_carry(tmp_path):
    """A new pipeline's output on its own grid (shifted origin, 1 mm) is
    found without being listed anywhere: faced -> flagged, defaced -> ok."""
    env = _env(tmp_path)
    shift = AFF.copy()
    shift[:3, 3] = [-5.0, -3.0, 0.0]               # grid origin moved; same world
    big = np.zeros((50, 46, 40), np.float32)
    big[5:45, 3:43, :] = _head(True)
    _save(env.bids / "derivatives" / "newpipe" / "sub-01" / "mri" / "orig.mgz.nii.gz", big, shift)
    clean = big.copy()
    clean[5:45, 3:43, :] = _head(False)
    _save(env.bids / "derivatives" / "newpipe" / "sub-01" / "mri" / "T1.nii.gz", clean, shift)
    res = _audit(env)
    assert _kinds(res) == [("face", "derivatives/newpipe/sub-01/mri/orig.mgz.nii.gz")]
    assert {s.method for s in res.scores if "newpipe" in s.path} == {"world"}


def test_mgz_volumes_are_scored(tmp_path):
    env = _env(tmp_path)
    p = env.bids / "derivatives" / "fs" / "sub-01" / "mri" / "nu.mgz"
    p.parent.mkdir(parents=True)
    nib.MGHImage(_head(True), AFF).to_filename(str(p))
    assert _kinds(_audit(env)) == [("face", "derivatives/fs/sub-01/mri/nu.mgz")]


def test_mni_space_uses_the_mni_mask(tmp_path):
    env = _env(tmp_path)
    mni_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    d = np.full((20, 20, 20), 300.0, np.float32)
    _save(env.bids / "derivatives" / "p" / "sub-01" / "anat" / "sub-01_space-MNI152NLin2009cAsym_res-3_T1w.nii.gz",
          d, np.diag([2.0, 2.0, 2.0, 1.0]))
    # same grid as the MNI leg -> grid match; make a second one on a 1 mm grid -> world carry
    big = np.full((40, 40, 40), 300.0, np.float32)
    off = AFF.copy()
    off[0, 3] = 0.5                                  # not the T1w leg's grid: carried, not matched
    _save(env.bids / "derivatives" / "p" / "sub-01" / "anat" / "sub-01_space-MNI152NLin2009cAsym_res-1_T1w.nii.gz",
          big, off)
    res = _audit(env)
    got = {s.path.rsplit("_", 2)[-2]: (s.mask, s.method, s.verdict) for s in res.scores if "MNI" in s.path}
    assert got == {"res-3": ("MNI", "grid", "face"), "res-1": ("MNI", "world", "face")}


def test_unknown_space_is_unplaced_and_unfold_is_skipped(tmp_path):
    env = _env(tmp_path)
    _save(env.bids / "derivatives" / "q" / "sub-01" / "anat" / "sub-01_space-ACPC_desc-preproc_T1w.nii.gz",
          _head(False), np.diag([0.9, 0.9, 0.9, 1.0]))
    _save(env.bids / "derivatives" / "h" / "sub-01" / "anat" / "sub-01_space-unfold_label-hipp_T2w.nii.gz",
          np.ones((10, 10, 10), np.uint8), np.diag([0.9, 0.9, 0.9, 1.0]))
    res = _audit(env)
    assert _kinds(res) == [("unplaced", "derivatives/q/sub-01/anat/sub-01_space-ACPC_desc-preproc_T1w.nii.gz")]


def test_face_outside_the_field_of_view_passes(tmp_path):
    """A hippocampal crop holds no face: its grid never reaches the face mask."""
    env = _env(tmp_path)
    aff = AFF.copy()
    aff[:3, 3] = [12, 10, 18]
    _save(env.bids / "derivatives" / "h" / "sub-01" / "anat" / "sub-01_space-cropT1w_desc-preproc_T2w.nii.gz",
          np.full((10, 10, 10), 500.0, np.float32), aff)
    res = _audit(env)
    assert res.findings == []
    assert [s.method for s in res.scores if "crop" in s.path] == ["outside-fov"]


def test_epi_datatypes_and_4d_volumes_are_not_scored(tmp_path):
    env = _env(tmp_path)
    _save(env.bids / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-x_bold.nii.gz", _head(True)[..., None].repeat(2, 3))
    _save(env.bids / "derivatives" / "p" / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-x_boldref.nii.gz", _head(True))
    _save(env.bids / "derivatives" / "p" / "sub-01" / "anat" / "sub-01_desc-stack_T1w.nii.gz",
          np.stack([_head(True)] * 2, axis=3))
    res = _audit(env)
    assert res.findings == []
    assert [s.method for s in res.scores if "stack" in s.path] == ["skipped"]
    assert not any("func" in s.path for s in res.scores)


def test_cifti_is_not_scored_and_a_broken_volume_fails_closed(tmp_path):
    env = _env(tmp_path)
    d = env.bids / "derivatives" / "g" / "sub-01"
    d.mkdir(parents=True)
    (d / "sub-01_space-fsLR_den-91k_bold.dtseries.nii").write_bytes(b"not read")
    (d / "sub-01_desc-x_stat.dscalar.nii").write_bytes(b"not read")
    (d / "sub-01_desc-broken_T1w.nii.gz").write_bytes(b"truncated")
    res = _audit(env)
    assert _kinds(res) == [("unreadable", "derivatives/g/sub-01/sub-01_desc-broken_T1w.nii.gz")]
    assert not any("dscalar" in s.path or "dtseries" in s.path for s in res.scores)


def test_signed_volumes_score_by_magnitude(tmp_path):
    env = _env(tmp_path)
    _save(env.bids / "derivatives" / "x" / "sub-01" / "anat" / "sub-01_desc-signed_T1w.nii.gz", -_head(True))
    assert [f.kind for f in _audit(env).findings] == ["face"]


def test_statistical_label_and_mask_maps_are_not_scored(tmp_path):
    """Unmasked fits fill the face region with values; they are not images of a face."""
    env = _env(tmp_path)
    full = np.full(SHAPE, 3.0, np.float32)
    for name in ("sub-01_contrast-a_stat-effect_statmap.nii.gz", "sub-01_desc-hrfr2_stat.nii.gz",
                 "sub-01_desc-hrfindex_dseg.nii.gz", "sub-01_label-GM_probseg.nii.gz", "sub-01_desc-head_mask.nii.gz"):
        _save(env.bids / "derivatives" / "glm" / "sub-01" / name, full)
    res = _audit(env)
    assert res.findings == [] and not any("glm" in s.path for s in res.scores)


def test_unnamed_volume_on_the_mni_box_takes_the_mni_mask(tmp_path):
    """FreeSurfer 8's synthmorph test image has no space- entity but sits on
    the MNI grid; scoring it with the native mask would read the wrong place."""
    env = _env(tmp_path)
    one_mm = np.full((40, 40, 40), 300.0, np.float32)           # 1 mm grid spanning the 2 mm MNI leg's box
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = [0.0, 0.0, 0.0]
    p = env.bids / "derivatives" / "fs" / "sub-01" / "mri" / "transforms" / "sm" / "test.nii.gz"
    _save(p, one_mm[:39, :39, :39], aff)                         # box 0..38 vs MNI leg 0..38
    (sc,) = [s for s in _audit(env).scores if "transforms" in s.path]
    assert (sc.mask, sc.method, sc.verdict) == ("MNI", "world", "face")


def test_raw_only_subject_uses_its_first_raw_run_as_native(tmp_path):
    env = Env(tmp_path)
    raw = _save(env.bids / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz", _head(False))
    d = env.source / "anat_defacing" / "sub-01" / "masks"
    m = _save(d / "raw_facemask.nii.gz", _face_mask().astype(np.uint8))
    (d.parent / "provenance.json").write_text(json.dumps({
        "legs": {"raw:sub-01_ses-01": {"facemask": str(m)}},
        "files": {str(raw): {"sha_after": _sha(raw), "leg": "raw:sub-01_ses-01"}}}))
    shift = AFF.copy()
    shift[:3, 3] = [-1.0, 0.0, 0.0]
    big = np.zeros((41, 40, 40), np.float32)
    big[1:, :, :] = _head(True)
    _save(env.bids / "derivatives" / "fmriprep" / "sub-01" / "anat" / "sub-01_desc-preproc_T1w.nii.gz", big, shift)
    res = _audit(env)
    assert _kinds(res) == [("face", "derivatives/fmriprep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz")]


def test_subject_without_provenance_fails(tmp_path):
    env = _env(tmp_path)
    _save(env.bids / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_T1w.nii.gz", _head(False))
    kinds = _kinds(_audit(env))
    assert ("undefaced", "sub-02/ses-01/anat/sub-02_ses-01_T1w.nii.gz") in kinds
    assert ("no-provenance", "sub-02/ses-01/anat/sub-02_ses-01_T1w.nii.gz") in kinds


def test_subjects_filter(tmp_path):
    env = _env(tmp_path)
    _save(env.bids / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_T1w.nii.gz", _head(True))
    assert _audit(env, subjects={"01"}).findings == []


# --------------------------------------------------------------------------- #
# provenance check of raw anatomy                                             #
# --------------------------------------------------------------------------- #
def test_changed_raw_file_is_flagged(tmp_path):
    env = _env(tmp_path)
    p = env.bids / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    d = _head(False)
    d[20, 20, 20] += 1
    _save(p, d)
    assert _kinds(_audit(env)) == [("changed", "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz")]


def test_a_copy_is_matched_to_the_working_tree_record(tmp_path):
    env = _env(tmp_path)
    share = tmp_path / "share"
    shutil.copytree(env.bids, share)
    assert _audit(env, root=share).findings == []
    _save(share / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz", _head(True))
    kinds = _kinds(_audit(env, root=share))
    assert ("changed", "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz") in kinds
    assert ("face", "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz") in kinds


def test_top_level_sourcedata_and_stimuli_are_not_walked(tmp_path):
    env = _env(tmp_path)
    _save(env.bids / "sourcedata" / "sub-01" / "anat" / "x_T1w.nii.gz", _head(True))
    _save(env.bids / "stimuli" / "sub-01" / "x.nii.gz", _head(True))
    # ...but a derivative's own sourcedata/ (FreeSurfer under fMRIPrep) is
    _save(env.bids / "derivatives" / "fmriprep" / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "T1.nii.gz",
          _head(True))
    assert _kinds(_audit(env)) == [("face", "derivatives/fmriprep/sourcedata/freesurfer/sub-01/mri/T1.nii.gz")]


# --------------------------------------------------------------------------- #
# command lines                                                               #
# --------------------------------------------------------------------------- #
def _config(tmp_path, env):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "base.toml").write_text(
        f'[paths]\nbids_project_dir = "{env.bids}"\nsource_dir = "{env.source}"\n'
        f'output_dir = "{env.bids / "derivatives"}"\nsingularity_dir = "{tmp_path / "c"}"\n')
    return cfg


def test_release_export_faces_exit_codes(tmp_path, capsys):
    import release_export

    env = _env(tmp_path)
    cfg = _config(tmp_path, env)
    report = tmp_path / "scores.tsv"
    assert release_export.main(["--config-dir", str(cfg), "faces", str(env.bids), "--report", str(report)]) == 0
    assert report.read_text().splitlines()[0] == faces.TSV_HEADER
    _save(env.bids / "derivatives" / "p" / "sub-01" / "anat" / "sub-01_T1w.nii.gz", _head(True))
    assert release_export.main(["--config-dir", str(cfg), "faces", str(env.bids)]) == 1
    assert "face" in capsys.readouterr().out


def test_deface_anat_audit_takes_a_root(tmp_path, monkeypatch):
    import deface_anat as da

    env = _env(tmp_path)
    roots = da.Roots(bids=env.bids, source=env.source, deriv=env.bids / "derivatives", containers=tmp_path / "c")
    monkeypatch.setattr(da, "load_roots", lambda: roots)
    assert da.main(["audit"]) == 0
    share = tmp_path / "share"
    shutil.copytree(env.bids, share)
    _save(share / "derivatives" / "p" / "sub-01" / "anat" / "sub-01_T1w.nii.gz", _head(True))
    assert da.main(["audit", "--root", str(share)]) == 1
    assert da.main(["audit"]) == 0


def test_deface_anat_quarantine_moves_faced_volumes_and_records_them(tmp_path, monkeypatch, capsys):
    import deface_anat as da

    env = _env(tmp_path)
    roots = da.Roots(bids=env.bids, source=env.source, deriv=env.bids / "derivatives", containers=tmp_path / "c")
    monkeypatch.setattr(da, "load_roots", lambda: roots)
    bad = _save(env.bids / "derivatives" / "old" / "sub-01" / "anat" / "sub-01_desc-preproc_T1w.nii.gz", _head(True))
    sha = _sha(bad)
    scores = tmp_path / "scores.tsv"
    res = _audit(env)
    scores.write_text(faces.TSV_HEADER + "\n" + "".join(s.row() + "\n" for s in res.scores))
    assert da.main(["quarantine", "--subject", "01", "--scores", str(scores), "--dry-run"]) == 0
    assert bad.exists()
    assert da.main(["quarantine", "--subject", "01", "--scores", str(scores)]) == 0
    mirror = env.source / "derivatives_faced" / "old" / "sub-01" / "anat" / bad.name
    assert not bad.exists() and _sha(mirror) == sha
    prov = json.loads(roots.provenance_path("01").read_text())
    assert prov["quarantined"][str(bad)]["sha256"] == sha
    assert da.main(["quarantine", "--subject", "01", "--scores", str(scores)]) == 0     # idempotent
    assert _audit(env).findings == []
    assert da.main(["quarantine", "--subject", "01", "--scores", str(scores), "--root", str(tmp_path)]) == 2

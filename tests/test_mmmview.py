"""mmmview dispatcher (scripts/mmmview.py) on entity-named fake paths.

The charter's Settles-when (workbench brain-viewer, sub-effort mmmview) is
checked here without real data: (1) a map under derivatives/prf or
derivatives/glm_localizer resolves the right underlay/mesh with zero flags,
(2) a feature file dispatches to the dashboard of its sidecar's extractor,
(3) an unplaceable file exits 2 with a message naming the flag, (4) build
and open are separable and a second call reuses the bundle.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import mmmview  # noqa: E402
from mmmview import (Opts, Roots, Unplaceable, classify, family_of,  # noqa: E402
                     find_sidecar, parse_entities, render, resolve)


def touch(path, text=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def roots(tmp_path):
    """Fake derivatives tree: fMRIPrep anat in three spaces, a coreg
    boldref, FreeSurfer meshes, a stimulus image dir, no catalog."""
    d = tmp_path / "derivatives"
    anat = d / "fmriprep" / "sub-07" / "anat"
    touch(anat / "sub-07_acq-MPR_desc-preproc_T1w.nii.gz")
    touch(anat / "sub-07_acq-MPR_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz")
    touch(anat / "sub-07_acq-MPR_space-MNI152NLin2009cAsym_res-1_desc-preproc_T1w.nii.gz")
    func = d / "fmriprep" / "sub-07" / "ses-04" / "func"
    touch(func / "sub-07_ses-04_task-floc_desc-coreg_boldref.nii.gz")
    touch(func / "sub-07_ses-04_task-floc_desc-hmc_boldref.nii.gz")
    touch(func / "sub-07_ses-04_task-motor_run-02_desc-coreg_boldref.nii.gz")
    surf = d / "fmriprep" / "sourcedata" / "freesurfer" / "sub-07" / "surf"
    for h in ("lh", "rh"):
        touch(surf / f"{h}.inflated")
        touch(surf / f"{h}.pial")
        touch(surf / f"{h}.curv")
    bids = tmp_path
    touch(bids / "stimuli" / "shared1000" / "images" / "shared0001_nsd00001.png")
    env = tmp_path / "stimfeat"
    (env / "bin").mkdir(parents=True)
    return Roots(deriv=d, bids=bids, stimfeat_env=env, catalog_db=None)


# ---------------------------------------------------------------------------
# filename grammar + families
# ---------------------------------------------------------------------------

class TestGrammar:
    def test_entities_in_filename_order(self):
        ents, suffix, ext = parse_entities(
            "sub-07_ses-04_task-floc_space-MNI152NLin2009cAsym_res-2_"
            "contrast-faceVsObject_stat-z_statmap.nii.gz")
        assert list(ents) == ["sub", "ses", "task", "space", "res",
                              "contrast", "stat"]
        assert ents["res"] == "2" and suffix == "statmap" and ext == ".nii.gz"

    def test_prf_suffix_and_gii_ext(self):
        ents, suffix, ext = parse_entities(
            "sub-07_task-prf_space-fsnative_hemi-L_desc-angle_negprf.shape.gii")
        assert suffix == "negprf" and ext == ".shape.gii"
        assert ents["hemi"] == "L" and ents["desc"] == "angle"

    @pytest.mark.parametrize("ents,suffix,family", [
        ({"desc": "R2"}, "prf", "prf"),
        ({"desc": "angle"}, "negprf", "prf"),
        ({"desc": "R2"}, "acompcorprf", "prf"),
        ({"stat": "z"}, "statmap", "glm-z"),
        ({"stat": "t"}, "statmap", "glm-t"),
        ({"stat": "effect"}, "statmap", "glm-effect"),
        ({"desc": "preproc"}, "bold", "unknown"),
        ({"desc": "R2"}, "bold", "unknown"),      # pRF desc, non-pRF suffix
    ])
    def test_family_of(self, ents, suffix, family):
        assert family_of(ents, suffix) == family


# ---------------------------------------------------------------------------
# stage 1 — classify (one case per table row, plus the exit-2 rows)
# ---------------------------------------------------------------------------

class TestClassify:
    def test_volume_map(self, tmp_path):
        p = touch(tmp_path / "sub-07_ses-04_task-floc_space-T1w_"
                  "contrast-faceVsObject_stat-z_statmap.nii.gz")
        (t,) = classify(p)
        assert t.kind == "volume" and t.maps == [p]
        assert t.entities["space"] == "T1w" and t.suffix == "statmap"

    def test_surface_map(self, tmp_path):
        p = touch(tmp_path / "sub-07_task-prf_space-fsnative_hemi-R_"
                  "desc-R2_prf.shape.gii")
        (t,) = classify(p)
        assert t.kind == "surface" and t.entities["hemi"] == "R"

    def test_surface_without_hemi_names_mesh_flag(self, tmp_path):
        p = touch(tmp_path / "sub-07_desc-R2_prf.shape.gii")
        with pytest.raises(Unplaceable, match="--mesh"):
            classify(p)

    def test_fsaverage_surface_is_unplaceable(self, tmp_path):
        p = touch(tmp_path / "sub-07_space-fsaverage_hemi-L_desc-R2_prf.func.gii")
        with pytest.raises(Unplaceable, match="fsnative"):
            classify(p)

    def test_directory_groups_by_space_hemi_and_merges_prf_variants(self, tmp_path):
        d = tmp_path / "sub-07"
        for pol in ("prf", "negprf"):
            for param in ("R2", "angle", "eccentricity"):
                touch(d / f"sub-07_task-prf_space-T1w_desc-{param}_{pol}.nii.gz")
                for hemi in ("L", "R"):
                    touch(d / f"sub-07_task-prf_space-fsnative_hemi-{hemi}_"
                              f"desc-{param}_{pol}.shape.gii")
        touch(d / "sub-07_task-prf_space-T1w_prf.json")          # ignored
        touch(d / "viz" / "sub-07_task-prf_space-T1w_desc-viewer_prf.html")
        targets = classify(d)
        # pRF suffix variants merge into ONE target per (space, hemi)
        keys = sorted((t.kind, t.entities.get("space"), t.entities.get("hemi"),
                       t.suffix, t.variants) for t in targets)
        assert keys == [
            ("surface", "fsnative", "L", None, ("prf", "negprf")),
            ("surface", "fsnative", "R", None, ("prf", "negprf")),
            ("volume", "T1w", None, None, ("prf", "negprf"))]
        vol = next(t for t in targets if t.kind == "volume")
        assert len(vol.maps) == 6
        # desc differs across members, so it drops out of the shared entities
        assert "desc" not in vol.entities and vol.entities["task"] == "prf"

    def test_directory_single_prf_variant_is_unmerged(self, tmp_path):
        d = tmp_path / "sub-07"
        for param in ("R2", "angle"):
            touch(d / f"sub-07_task-prf_space-T1w_desc-{param}_negprf.nii.gz")
        (t,) = classify(d)
        assert t.suffix == "negprf" and t.variants is None

    def test_variant_order_plain_negative_then_confounds(self, tmp_path):
        d = tmp_path / "sub-07"
        for pol in ("acompcorprf", "negprf", "prf", "motion6prf"):
            touch(d / f"sub-07_task-prf_space-T1w_desc-R2_{pol}.nii.gz")
        (t,) = classify(d)
        assert t.variants == ("prf", "negprf", "acompcorprf", "motion6prf")

    def test_directory_spanning_subjects_is_unplaceable(self, tmp_path):
        touch(tmp_path / "sub-07_space-T1w_stat-z_statmap.nii.gz")
        touch(tmp_path / "sub-08_space-T1w_stat-z_statmap.nii.gz")
        with pytest.raises(Unplaceable, match="several subjects"):
            classify(tmp_path)

    def test_empty_directory_is_unplaceable(self, tmp_path):
        touch(tmp_path / "notes.txt")
        touch(tmp_path / "mean.nii.gz")          # no entities: skipped
        with pytest.raises(Unplaceable, match="no brain maps"):
            classify(tmp_path)

    def test_features_by_sidecar(self, tmp_path):
        csv = touch(tmp_path / "aesthetics.csv", "filename,a\n")
        touch(tmp_path / "aesthetics.meta.json",
              json.dumps({"extractor": "viz2psy"}))
        (t,) = classify(csv)
        assert t.kind == "features" and t.extractor == "viz2psy"

    def test_prefix_family_sidecar(self, tmp_path):
        csv = touch(tmp_path / "clap_text_chunks.csv", "x\n")
        meta = touch(tmp_path / "clap_text.meta.json",
                     json.dumps({"extractor": "word2psy"}))
        assert find_sidecar(csv) == meta
        (t,) = classify(csv)
        assert t.extractor == "word2psy"

    def test_features_without_sidecar(self, tmp_path):
        csv = touch(tmp_path / "orphan.csv", "x\n")
        with pytest.raises(Unplaceable, match="meta.json"):
            classify(csv)

    def test_movies_table_with_psytwill_sidecar(self, tmp_path):
        pq = touch(tmp_path / "movies_frames_features.parquet")
        touch(tmp_path / "movies_frames_features.meta.json",
              json.dumps({"extractor": "psytwill"}))
        other = touch(tmp_path / "movies_audio_frames_features.parquet")
        (t,) = classify(pq)
        assert t.kind == "movies" and t.maps == [pq.parent / other.name, pq]

    def test_movies_table_without_sidecar(self, tmp_path):
        # composed TB runs (tb-timelines) write no Contract B sidecar; the
        # movie-schema naming alone dispatches
        pq = touch(tmp_path / "movies_frames_features.parquet")
        (t,) = classify(pq)
        assert t.kind == "movies" and t.maps == [pq]

    def test_movies_directory(self, tmp_path):
        touch(tmp_path / "movies_frames_features.parquet")
        (t,) = classify(tmp_path)
        assert t.kind == "movies" and t.source == tmp_path

    def test_composed_run_root_with_features_child(self, tmp_path):
        pq = touch(tmp_path / "run" / "features" / "movies_frames_features.parquet")
        (tmp_path / "run" / "movies").mkdir()
        (t,) = classify(tmp_path / "run")
        assert t.kind == "movies" and t.maps == [pq]

    def test_non_movies_psytwill_aggregate_is_unplaceable(self, tmp_path):
        pq = touch(tmp_path / "shared1000_image_features.parquet")
        touch(tmp_path / "shared1000_image_features.meta.json",
              json.dumps({"extractor": "psytwill"}))
        with pytest.raises(Unplaceable, match="viz movies"):
            classify(pq)

    def test_events_table_names_plot_tools(self, tmp_path):
        p = touch(tmp_path / "sub-07_ses-04_task-floc_events.tsv")
        with pytest.raises(Unplaceable, match="plot_"):
            classify(p)

    def test_anything_else_names_the_flags(self, tmp_path):
        p = touch(tmp_path / "README.md")
        with pytest.raises(Unplaceable, match="--underlay/--mesh"):
            classify(p)

    def test_missing_path(self, tmp_path):
        with pytest.raises(Unplaceable, match="no such file"):
            classify(tmp_path / "nope.nii.gz")


# ---------------------------------------------------------------------------
# stage 2 — resolve
# ---------------------------------------------------------------------------

class TestResolveUnderlay:
    def _plan(self, roots, path, **kw):
        (t,) = classify(path)
        return resolve(t, roots, Opts(**kw))

    def test_t1w_space_gets_native_preproc_t1w(self, roots, tmp_path):
        p = touch(tmp_path / "derivatives" / "prf" / "sub-07" /
                  "sub-07_task-prf_space-T1w_desc-R2_prf.nii.gz")
        plan = self._plan(roots, p)
        assert plan.inputs["underlay"].name == "sub-07_acq-MPR_desc-preproc_T1w.nii.gz"
        assert plan.renderer == "volume"

    def test_mni_res2_gets_matching_res(self, roots, tmp_path):
        p = touch(tmp_path / "derivatives" / "glm_localizer" / "sub-07" /
                  "ses-04" / "func" / "sub-07_ses-04_task-floc_"
                  "space-MNI152NLin2009cAsym_res-2_contrast-faceVsObject_"
                  "stat-z_statmap.nii.gz")
        plan = self._plan(roots, p)
        assert "_res-2_" in plan.inputs["underlay"].name
        assert "_space-MNI152NLin2009cAsym_" in plan.inputs["underlay"].name

    def test_spaceless_map_gets_coreg_boldref_of_same_run_unit(self, roots, tmp_path):
        p = touch(tmp_path / "derivatives" / "glm_localizer" / "sub-07" /
                  "ses-04" / "func" /
                  "sub-07_ses-04_task-floc_stat-z_statmap.nii.gz")
        plan = self._plan(roots, p)
        assert plan.inputs["underlay"].name == (
            "sub-07_ses-04_task-floc_desc-coreg_boldref.nii.gz")

    def test_spaceless_map_with_run_prefers_run_boldref(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-07_ses-04_task-motor_run-02_stat-t_statmap.nii.gz")
        plan = self._plan(roots, p)
        assert "_run-02_" in plan.inputs["underlay"].name

    def test_missing_underlay_names_flag_and_never_another_subject(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-99_task-prf_space-T1w_desc-R2_prf.nii.gz")
        with pytest.raises(Unplaceable, match="--underlay") as ei:
            self._plan(roots, p)
        assert "sub-07" not in str(ei.value)

    def test_unknown_space_names_flag(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-07_space-MNI152NLin6Asym_stat-z_statmap.nii.gz")
        with pytest.raises(Unplaceable, match="--underlay"):
            self._plan(roots, p)

    def test_no_sub_entity_names_flag(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "zmap.nii.gz")
        with pytest.raises(Unplaceable, match="--underlay"):
            self._plan(roots, p)

    def test_underlay_override_wins(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "zmap.nii.gz")
        u = touch(tmp_path / "x" / "mean.nii.gz")
        plan = self._plan(roots, p, underlay=str(u))
        assert plan.inputs["underlay"] == u

    def test_ambiguous_underlay_lists_candidates(self, roots, tmp_path):
        anat = roots.deriv / "fmriprep" / "sub-07" / "anat"
        touch(anat / "sub-07_acq-MP2RAGE_desc-preproc_T1w.nii.gz")
        p = touch(tmp_path / "x" / "sub-07_space-T1w_stat-z_statmap.nii.gz")
        with pytest.raises(Unplaceable, match="ambiguous.*MP2RAGE.*--underlay"):
            self._plan(roots, p)


class TestResolveMesh:
    def test_hemi_r_gets_rh_inflated_and_curv(self, roots, tmp_path):
        p = touch(tmp_path / "derivatives" / "prf" / "sub-07" /
                  "sub-07_task-prf_space-fsnative_hemi-R_desc-R2_prf.shape.gii")
        (t,) = classify(p)
        plan = resolve(t, roots)
        assert plan.renderer == "surface"
        assert plan.inputs["mesh"].name == "rh.inflated"
        assert plan.inputs["curv"].name == "rh.curv"

    def test_surf_flavor(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-07_space-fsnative_hemi-L_desc-R2_prf.shape.gii")
        (t,) = classify(p)
        assert resolve(t, roots, Opts(surf="pial")).inputs["mesh"].name == "lh.pial"

    def test_missing_mesh_names_flag(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-42_space-fsnative_hemi-L_desc-R2_prf.shape.gii")
        with pytest.raises(Unplaceable, match="--mesh"):
            resolve(classify(p)[0], roots)

    def test_mesh_override_wins(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-42_space-fsnative_hemi-L_desc-R2_prf.shape.gii")
        m = touch(tmp_path / "surf" / "lh.white")
        plan = resolve(classify(p)[0], roots, Opts(mesh=str(m)))
        assert plan.inputs["mesh"] == m and plan.inputs["curv"] is None


class TestOutputPath:
    def test_lands_in_subject_viz_dir(self, roots, tmp_path):
        p = touch(tmp_path / "derivatives" / "glm_localizer" / "sub-07" /
                  "ses-04" / "func" / "sub-07_ses-04_task-floc_space-T1w_"
                  "contrast-faceVsObject_stat-z_statmap.nii.gz")
        plan = resolve(classify(p)[0], roots)
        assert plan.out.parent == tmp_path / "derivatives" / "glm_localizer" / "sub-07" / "viz"
        assert plan.out.name == ("sub-07_ses-04_task-floc_space-T1w_"
                                 "contrast-faceVsObject_stat-z_desc-viewer_statmap.html")

    def test_directory_bundle_matches_todays_prf_names(self, roots, tmp_path):
        d = tmp_path / "derivatives" / "prf" / "sub-07"
        for param in ("R2", "angle"):
            touch(d / f"sub-07_task-prf_space-T1w_desc-{param}_prf.nii.gz")
        (t,) = classify(d)
        plan = resolve(t, roots)
        assert plan.out == d / "viz" / "sub-07_task-prf_space-T1w_desc-viewer_prf.html"

    def test_merged_variant_bundle_name_and_display(self, roots, tmp_path):
        d = tmp_path / "derivatives" / "prf" / "sub-07"
        for pol in ("negprf", "prf"):
            for param in ("R2", "angle"):
                touch(d / f"sub-07_task-prf_space-T1w_desc-{param}_{pol}.nii.gz")
        (t,) = classify(d)
        plan = resolve(t, roots)
        assert plan.out == (d / "viz" /
                            "sub-07_task-prf_space-T1w_desc-viewer_prfvariants.html")
        # sorted plain-fit first, each entry carrying its variant, each
        # variant masked on its OWN R2
        assert [(e["label"], e["variant"]) for e in plan.display] == [
            ("R2", "prf"), ("angle", "prf"),
            ("R2", "negprf"), ("angle", "negprf")]
        ang_neg = next(e for e in plan.display
                       if e["label"] == "angle" and e["variant"] == "negprf")
        assert ang_neg["mask"].name.endswith("_desc-R2_negprf.nii.gz")
        assert "prf+negprf" in plan.title

    def test_out_dir_override(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-07_space-T1w_stat-z_statmap.nii.gz")
        plan = resolve(classify(p)[0], roots, Opts(out_dir=str(tmp_path / "o")))
        assert plan.out.parent == tmp_path / "o"


# ---------------------------------------------------------------------------
# display profiles
# ---------------------------------------------------------------------------

class TestDisplay:
    def test_prf_profile_masks_on_sibling_r2(self, roots, tmp_path):
        d = tmp_path / "x"
        r2 = touch(d / "sub-07_task-prf_space-T1w_desc-R2_prf.nii.gz")
        ang = touch(d / "sub-07_task-prf_space-T1w_desc-angle_prf.nii.gz")
        plan = resolve(classify(ang)[0], roots, Opts(r2_floor=15.0))
        (e,) = plan.display
        assert e["family"] == "prf" and e["mask"] == r2
        assert e["profile"]["colormap"] == "hsv" and e["profile"]["angle_legend"]
        assert e["profile"]["floor"] == 15.0 and e["label"] == "angle"
        assert not plan.messages

    def test_prf_r2_embeds_unthresholded(self, roots, tmp_path):
        # R2 thresholds itself, so it is not baked: cal_min carries the
        # floor and the viewer's threshold slider can walk it down to 0
        r2 = touch(tmp_path / "x" /
                   "sub-07_task-prf_space-T1w_desc-R2_negprf.nii.gz")
        plan = resolve(classify(r2)[0], roots, Opts(r2_floor=15.0))
        (e,) = plan.display
        assert e["family"] == "prf" and e["mask"] is None
        assert e["profile"]["cal_min"] == 15.0
        assert not plan.messages

    def test_prf_without_r2_warns(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-07_task-prf_space-T1w_desc-size_prf.nii.gz")
        plan = resolve(classify(p)[0], roots)
        assert plan.display[0]["mask"] is None
        assert any("no R2" in m for m in plan.messages)

    @pytest.mark.parametrize("stat,cal_min,thr", [
        ("z", 2.3, 3.1), ("t", 2.0, None), ("effect", 0.0, None)])
    def test_glm_profiles(self, roots, tmp_path, stat, cal_min, thr):
        p = touch(tmp_path / "x" / f"sub-07_space-T1w_contrast-a_stat-{stat}_statmap.nii.gz")
        plan = resolve(classify(p)[0], roots)
        (e,) = plan.display
        assert e["family"] == f"glm-{stat}"
        assert e["profile"]["tails"] and e["profile"]["cal_min"] == cal_min
        assert e["profile"]["threshold"] == thr
        assert e["label"] == f"contrast-a stat-{stat}"
        assert not plan.messages

    def test_unknown_family_warns_and_defaults(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-07_space-T1w_desc-brain_mask.nii.gz")
        plan = resolve(classify(p)[0], roots)
        (e,) = plan.display
        assert e["family"] == "unknown" and e["profile"]["colormap"] == "viridis"
        assert any("no display profile" in m for m in plan.messages)

    def test_directory_layers_grouped_by_family(self, roots, tmp_path):
        d = tmp_path / "x"
        touch(d / "sub-07_space-T1w_desc-brain_mask.nii.gz")
        touch(d / "sub-07_space-T1w_contrast-a_stat-effect_statmap.nii.gz")
        touch(d / "sub-07_space-T1w_contrast-a_stat-z_statmap.nii.gz")
        # three suffixes -> three targets; each carries its own family order
        fams = {t.suffix: [e["family"] for e in resolve(t, roots).display]
                for t in classify(d)}
        assert fams == {"mask": ["unknown"], "statmap": ["glm-z", "glm-effect"]}


# ---------------------------------------------------------------------------
# dispatch — feature files
# ---------------------------------------------------------------------------

class TestDispatch:
    def _feat(self, tmp_path, subdir, name, meta):
        d = tmp_path / "derivatives" / "stimuli_features" / subdir
        csv = touch(d / f"{name}.csv", "stimulus_id,x\n")
        touch(d / f"{name}.meta.json", json.dumps(meta))
        return csv

    def test_viz2psy_dashboard_with_image_root_from_sidecar(self, roots, tmp_path):
        img = roots.bids / "stimuli" / "shared1000" / "images" / "shared0001_nsd00001.png"
        csv = self._feat(tmp_path, "shared1000", "aesthetics",
                         {"extractor": "viz2psy", "input": {"paths": [str(img)]}})
        plan = resolve(classify(csv)[0], roots)
        assert plan.renderer == "features"
        assert plan.command[0] == str(roots.stimfeat_env / "bin" / "viz2psy-viz")
        assert plan.command[1:3] == ["dashboard", str(csv)]
        assert plan.command[plan.command.index("--image-root") + 1] == str(img.parent)
        assert plan.out == csv.parent / "viz" / "aesthetics_desc-viewer.html"
        assert not plan.messages

    def test_viz2psy_falls_back_to_registry_then_no_images(self, roots, tmp_path):
        csv = self._feat(tmp_path, "shared1000", "gist",
                         {"extractor": "viz2psy",
                          "input": {"paths": ["/moved/away/x.png"]}})
        plan = resolve(classify(csv)[0], roots)
        root = plan.command[plan.command.index("--image-root") + 1]
        assert root == str(roots.bids / "stimuli" / "shared1000" / "images")
        csv2 = self._feat(tmp_path, "somewhere_else", "gist",
                          {"extractor": "viz2psy", "input": {}})
        plan2 = resolve(classify(csv2)[0], roots)
        assert "--no-images" in plan2.command
        assert any("thumbnails" in m for m in plan2.messages)

    def test_aud2psy_browse_with_audio_when_input_resolves(self, roots, tmp_path):
        wav = touch(tmp_path / "stimuli" / "twp1000" / "nova" / "apple.wav")
        csv = self._feat(tmp_path, "twp1000", "psychoacoustic",
                         {"extractor": "aud2psy", "input": {"paths": [str(wav)]}})
        plan = resolve(classify(csv)[0], roots)
        assert plan.command[:3] == [str(roots.stimfeat_env / "bin" / "aud2psy"),
                                    "viz", "browse"]
        assert plan.command[plan.command.index("--audio") + 1] == str(wav)

    def test_aud2psy_without_audio_notes_it(self, roots, tmp_path):
        csv = self._feat(tmp_path, "twp1000", "psychoacoustic",
                         {"extractor": "aud2psy", "input": {"paths": ["/gone.wav"]}})
        plan = resolve(classify(csv)[0], roots)
        assert "--audio" not in plan.command
        assert any("playback" in m for m in plan.messages)

    def test_word2psy_browse(self, roots, tmp_path):
        csv = self._feat(tmp_path, "twp1000", "lexical_norms",
                         {"extractor": "word2psy", "input": {"type": "text"}})
        plan = resolve(classify(csv)[0], roots)
        assert plan.command[:3] == [str(roots.stimfeat_env / "bin" / "word2psy"),
                                    "viz", "browse"]
        assert "-o" in plan.command and not plan.messages

    def test_no_stimfeat_env_runs_from_path(self, roots, tmp_path):
        csv = self._feat(tmp_path, "twp1000", "lexical_norms",
                         {"extractor": "word2psy"})
        bare = Roots(roots.deriv, roots.bids, None, None)
        plan = resolve(classify(csv)[0], bare)
        assert plan.command[0] == "word2psy"
        assert any("stimfeat_env" in m for m in plan.messages)

    def test_features_render_runs_command_and_passes_stderr(self, roots, tmp_path):
        csv = self._feat(tmp_path, "twp1000", "lexical_norms",
                         {"extractor": "word2psy"})
        plan = resolve(classify(csv)[0], roots)
        plan.command = [sys.executable, "-c",
                        "import sys; sys.stderr.write('boom'); sys.exit(4)"]
        with pytest.raises(mmmview.RenderError, match="boom"):
            render(plan)
        plan.command = [sys.executable, "-c",
                        f"open({str(plan.out)!r}, 'w').write('<html>')"]
        out, built = render(plan)
        assert built and out.read_text() == "<html>"
        # second call: output newer than csv + sidecar -> reused
        assert render(plan) == (out, False)

    def _movies(self, tmp_path):
        d = tmp_path / "derivatives" / "stimuli_features"
        pq = touch(d / "psytwill" / "movies_frames_features.parquet")
        (d / "movies").mkdir()
        return pq

    def test_movies_command_and_default_out(self, roots, tmp_path):
        pq = self._movies(tmp_path)
        (roots.bids / "stimuli" / "stimulus_registry").mkdir()
        plan = resolve(classify(pq)[0], roots)
        assert plan.renderer == "movies"
        films = str(pq.parent.parent / "movies")
        assert plan.command[:3] == [str(roots.stimfeat_env / "bin" / "psytwill"),
                                    "viz", "movies"]
        assert plan.command[plan.command.index("--features-dir") + 1] == str(pq.parent)
        assert plan.command[plan.command.index("--films-dir") + 1] == films
        assert plan.command[plan.command.index("--registry") + 1] == \
            str(roots.bids / "stimuli" / "stimulus_registry")
        assert plan.out == Path(films) / "viz" / "timeline" / "index.html"
        assert not plan.messages

    def test_movies_without_registry_notes_it(self, roots, tmp_path):
        plan = resolve(classify(self._movies(tmp_path))[0], roots)
        assert "--registry" not in plan.command
        assert any("registry" in m for m in plan.messages)

    def test_movies_missing_films_dir_names_the_flag(self, roots, tmp_path):
        pq = touch(tmp_path / "alone" / "movies_frames_features.parquet")
        with pytest.raises(Unplaceable, match="--films-dir"):
            resolve(classify(pq)[0], roots)
        films = tmp_path / "elsewhere"
        films.mkdir()
        plan = resolve(classify(pq)[0], roots, Opts(films_dir=str(films)))
        assert plan.command[plan.command.index("--films-dir") + 1] == str(films)

    def test_movies_render_reuses_current_bundle(self, roots, tmp_path):
        pq = self._movies(tmp_path)
        plan = resolve(classify(pq)[0], roots)
        plan.command = [sys.executable, "-c", "import pathlib,sys; "
                        f"p = pathlib.Path({str(plan.out)!r}); "
                        "p.parent.mkdir(parents=True, exist_ok=True); "
                        "p.write_text('<html>')"]
        out, built = render(plan)
        assert built and out.read_text() == "<html>"
        assert render(plan) == (out, False)


# ---------------------------------------------------------------------------
# render + idempotence on real (tiny) volumes
# ---------------------------------------------------------------------------

@pytest.fixture
def zmap(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.normal(0, 3, (4, 4, 3)).astype(np.float32)
    data[0, 0, 0] = np.nan
    p = tmp_path / "x" / "sub-07_ses-04_task-floc_space-T1w_contrast-faceVsObject_stat-z_statmap.nii.gz"
    p.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), p)
    u = tmp_path / "x" / "underlay.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 3), np.float32), np.eye(4)), u)
    return p, u, data


def _config(html):
    import re
    m = re.search(r"const CONFIG = (\{.*?\});\n", html)
    return json.loads(m.group(1).replace("<\\/", "</"))


class TestRenderVolume:
    def test_zmap_builds_two_tails_and_is_idempotent(self, roots, zmap):
        p, u, data = zmap
        plan = resolve(classify(p)[0], roots, Opts(underlay=str(u)))
        out, built = render(plan)
        assert built and out.exists()
        html = out.read_text()
        assert f"mmmview-key: {plan.key}" in html
        cfg = _config(html)
        labels = [v["label"] for v in cfg["volumes"]]
        assert labels[0] == "underlay"
        assert labels[1].startswith("contrast-faceVsObject stat-z +")
        assert "threshold 3.1" in labels[1]
        assert labels[2].startswith("contrast-faceVsObject stat-z −")
        assert [v["visible"] for v in cfg["volumes"][1:]] == [True, False]
        assert cfg["volumes"][1]["colormap"] == "warm"
        assert cfg["volumes"][2]["colormap"] == "winter"
        assert cfg["volumes"][1]["cal_min"] == 2.3
        # second call reuses; a changed floor changes the key and rebuilds
        mtime = out.stat().st_mtime
        assert render(plan) == (out, False)
        assert out.stat().st_mtime == mtime
        plan2 = resolve(classify(p)[0], roots, Opts(underlay=str(u), r2_floor=20.0))
        assert plan2.key != plan.key
        assert render(plan2)[1] is True

    def test_changed_input_rebuilds(self, roots, zmap):
        p, u, data = zmap
        plan = resolve(classify(p)[0], roots, Opts(underlay=str(u)))
        render(plan)
        nib.save(nib.Nifti1Image(data * 2, np.eye(4)), p)
        plan2 = resolve(classify(p)[0], roots, Opts(underlay=str(u)))
        assert plan2.key != plan.key and render(plan2)[1] is True

    def test_force_rebuilds(self, roots, zmap):
        p, u, _ = zmap
        plan = resolve(classify(p)[0], roots, Opts(underlay=str(u)))
        render(plan)
        assert render(plan, force=True)[1] is True

    def test_prf_volume_masks_to_r2(self, roots, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        r2 = np.full((3, 3, 2), 5.0, np.float32)
        r2[0] = 50.0
        ang = np.full((3, 3, 2), 90.0, np.float32)
        base = "sub-07_task-prf_space-T1w_desc-{}_prf.nii.gz"
        nib.save(nib.Nifti1Image(r2, np.eye(4)), d / base.format("R2"))
        nib.save(nib.Nifti1Image(ang, np.eye(4)), d / base.format("angle"))
        u = d / "u.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((3, 3, 2), np.float32), np.eye(4)), u)
        (t,) = classify(d)
        plan = resolve(t, roots, Opts(underlay=str(u)))
        out, _ = render(plan)
        cfg = _config(out.read_text())
        labels = [v["label"] for v in cfg["volumes"]]
        assert labels == ["underlay", "R2", "angle"]
        assert cfg["volumes"][1]["cal_min"] == 10.0
        assert cfg["volumes"][2]["angle_legend"] and cfg["volumes"][2]["cal_max"] == 360.0

    def test_prf_variant_bundle_config(self, roots, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        u = d / "u.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((3, 3, 2), np.float32), np.eye(4)), u)
        for pol in ("prf", "negprf"):
            base = f"sub-07_task-prf_space-T1w_desc-{{}}_{pol}.nii.gz"
            nib.save(nib.Nifti1Image(np.full((3, 3, 2), 50.0, np.float32),
                                     np.eye(4)), d / base.format("R2"))
            nib.save(nib.Nifti1Image(np.full((3, 3, 2), 90.0, np.float32),
                                     np.eye(4)), d / base.format("angle"))
        (t,) = classify(d)
        plan = resolve(t, roots, Opts(underlay=str(u)))
        out, _ = render(plan)
        assert out.name.endswith("_desc-viewer_prfvariants.html")
        html = out.read_text()
        cfg = _config(html)
        got = [(v["label"], v.get("variant")) for v in cfg["volumes"]]
        assert got == [("underlay", None), ("R2", "prf"), ("angle", "prf"),
                       ("R2", "negprf"), ("angle", "negprf")]
        assert [v["visible"] for v in cfg["volumes"][1:]] == [
            True, False, False, False]
        assert 'id="variantbox"' in html
        assert "variants: prf, negprf" in cfg["notes"]

    def test_renderer_failure_is_render_error(self, roots, tmp_path):
        p = touch(tmp_path / "x" / "sub-07_space-T1w_stat-z_statmap.nii.gz", "not nifti")
        u = touch(tmp_path / "x" / "u.nii.gz", "not nifti")
        plan = resolve(classify(p)[0], roots, Opts(underlay=str(u)))
        with pytest.raises(mmmview.RenderError):
            render(plan)


class TestSurfaceRender:
    def test_prf_surface_layers(self, roots, tmp_path):
        surf = roots.deriv / "fmriprep" / "sourcedata" / "freesurfer" / "sub-07" / "surf"
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
        nib.freesurfer.write_geometry(str(surf / "lh.inflated"), verts, faces)
        nib.freesurfer.write_morph_data(str(surf / "lh.curv"),
                                        np.array([0.5, -0.5, 0.2, -0.2]))
        d = tmp_path / "derivatives" / "prf" / "sub-07"
        d.mkdir(parents=True)

        def gii(name, vals):
            da = nib.gifti.GiftiDataArray(np.asarray(vals, np.float32),
                                          intent="NIFTI_INTENT_SHAPE")
            nib.save(nib.gifti.GiftiImage(darrays=[da]), d / name)
        base = "sub-07_task-prf_space-fsnative_hemi-L_desc-{}_prf.shape.gii"
        gii(base.format("R2"), [50, 5, 50, np.nan])
        gii(base.format("eccentricity"), [1.0, 2.0, 3.0, 4.0])
        (t,) = classify(d)
        plan = resolve(t, roots)
        out, built = render(plan)
        assert built
        cfg = _config(out.read_text())
        (mesh,) = cfg["meshes"]
        assert mesh["name"] == "lh.inflated"
        assert [l["label"] for l in mesh["layers"]] == ["curvature", "R2", "eccentricity"]
        assert mesh["layers"][0]["shade"] and mesh["layers"][1]["visible"]
        assert not mesh["layers"][2]["visible"]


# ---------------------------------------------------------------------------
# viz-dir index
# ---------------------------------------------------------------------------

class TestIndex:
    def test_lists_bundles_with_entity_labels(self, tmp_path):
        viz = tmp_path / "sub-07" / "viz"
        vol = "sub-07_task-prf_space-T1w_desc-viewer_prfvariants.html"
        surf = "sub-07_task-prf_space-fsnative_hemi-L_desc-viewer_prfvariants.html"
        feat = "aesthetics_desc-viewer.html"          # a dashboard counts too
        for n in (vol, surf, feat):
            touch(viz / n)
        touch(viz / "notes.txt")                      # never listed
        idx = mmmview.write_index(viz)
        html = idx.read_text()
        assert idx.name == "index.html"
        assert f'value="{vol}"' in html
        assert "task-prf space-fsnative hemi-L prfvariants" in html
        assert "aesthetics" in html
        assert "notes.txt" not in html
        assert "<h1>sub-07</h1>" in html
        assert "<iframe" in html and 'src="http' not in html   # data-free, local

    def test_lone_bundle_needs_no_index(self, tmp_path):
        viz = tmp_path / "viz"
        touch(viz / "sub-07_task-x_desc-viewer_statmap.html")
        assert mmmview.write_index(viz) is None
        assert not (viz / "index.html").exists()

    def test_unchanged_index_is_not_rewritten(self, tmp_path):
        viz = tmp_path / "viz"
        touch(viz / "sub-07_stat-z_desc-viewer_statmap.html")
        touch(viz / "sub-07_desc-brain_desc-viewer_mask.html")
        idx = mmmview.write_index(viz)
        before = idx.stat().st_mtime_ns
        assert mmmview.write_index(viz) == idx
        assert idx.stat().st_mtime_ns == before

    def test_cli_multi_bundle_dir_writes_index(self, roots, tmp_path, capsys,
                                               monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "x"
        d.mkdir()
        u = d / "u.nii.gz"
        ones = np.ones((3, 3, 2), np.float32)
        nib.save(nib.Nifti1Image(ones, np.eye(4)), u)
        nib.save(nib.Nifti1Image(ones, np.eye(4)),
                 d / "sub-07_space-T1w_stat-z_statmap.nii.gz")
        nib.save(nib.Nifti1Image(ones, np.eye(4)),
                 d / "sub-07_space-T1w_desc-brain_mask.nii.gz")
        rc = mmmview.main([str(d), "--no-open", "--underlay", str(u)])
        out = capsys.readouterr().out
        assert rc == 0
        idx = d / "viz" / "index.html"
        assert idx.exists() and f"index {idx}" in out
        html = idx.read_text()
        assert "_desc-viewer_statmap.html" in html
        assert "_desc-viewer_mask.html" in html

    def test_open_fragment_preselects(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BROWSER", "true")
        calls = []
        monkeypatch.setattr(mmmview.subprocess, "Popen",
                            lambda cmd, **kw: calls.append(cmd))
        idx = touch(tmp_path / "index.html")
        assert mmmview.open_view(idx, fragment="a_desc-viewer_b.html")
        assert calls[0][-1].endswith("index.html#a_desc-viewer_b.html")


# ---------------------------------------------------------------------------
# CLI: exit codes, --no-open, the shim
# ---------------------------------------------------------------------------

class TestCli:
    def test_unplaceable_exits_2(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots",
                            lambda *_: Roots(tmp_path, tmp_path))
        p = touch(tmp_path / "README.md")
        assert mmmview.main([str(p), "--no-open"]) == 2
        assert "--underlay/--mesh" in capsys.readouterr().err

    def test_no_open_prints_path(self, roots, zmap, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        p, u, _ = zmap
        rc = mmmview.main([str(p), "--no-open", "--underlay", str(u)])
        out = capsys.readouterr().out
        assert rc == 0 and "wrote" in out and "_desc-viewer_statmap.html" in out
        rc = mmmview.main([str(p), "--no-open", "--underlay", str(u)])
        assert rc == 0 and "current" in capsys.readouterr().out

    def test_open_prints_hint_when_nothing_available(self, roots, zmap, capsys,
                                                     monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        monkeypatch.delenv("BROWSER", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        # a headless Linux host: no macOS `open`, no xdg-open
        monkeypatch.setattr(mmmview.sys, "platform", "linux")
        p, u, _ = zmap
        assert mmmview.main([str(p), "--underlay", str(u)]) == 0
        assert "Remote-SSH" in capsys.readouterr().out

    def test_open_uses_open_on_macos(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BROWSER", raising=False)
        monkeypatch.setattr(mmmview.sys, "platform", "darwin")
        monkeypatch.setattr(mmmview.shutil, "which", lambda n: f"/usr/bin/{n}")
        calls = []
        monkeypatch.setattr(mmmview.subprocess, "Popen",
                            lambda cmd, **kw: calls.append(cmd))
        bundle = touch(tmp_path / "b_desc-viewer_statmap.html")
        assert mmmview.open_view(bundle) == "opened via open"
        assert calls and calls[0][0] == "open"

    def test_render_failure_exits_3(self, roots, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        p = touch(tmp_path / "x" / "sub-07_space-T1w_stat-z_statmap.nii.gz", "bad")
        u = touch(tmp_path / "x" / "u.nii.gz", "bad")
        assert mmmview.main([str(p), "--no-open", "--underlay", str(u)]) == 3
        assert "renderer failed" in capsys.readouterr().err

    def test_shim_pins_the_repo_venv(self):
        shim = SCRIPTS.parent / "bin" / "mmmview"
        assert shim.exists() and shim.stat().st_mode & 0o111
        text = shim.read_text()
        assert ".venv/bin/python" in text and "scripts/mmmview.py" in text


# ---------------------------------------------------------------------------
# browse index (mmmview-browse increment 1): a directory mmmview cannot place
# gets a navigable page instead of exit 2
# ---------------------------------------------------------------------------

@pytest.fixture
def browse_tree(tmp_path):
    """A directory mmmview cannot place: no maps of its own, but a feature
    file it CAN place, a subdirectory of maps, and a subdirectory carrying
    its own index."""
    d = tmp_path / "study"
    touch(d / "README.md")
    touch(d / "aesthetics.csv", "x")
    touch(d / "aesthetics.meta.json",
          json.dumps({"extractor": "viz2psy", "input": {"paths": []}}))
    touch(d / "sub-07" / "sub-07_space-T1w_stat-z_statmap.nii.gz")
    touch(d / "sub-07" / "viz" / "index.html")
    (d / "plain").mkdir()
    return d


class TestBrowsePage:
    def test_unplaceable_dir_exits_zero_with_signed_browse_page(
            self, roots, browse_tree, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        assert mmmview.main([str(browse_tree), "--no-open"]) == 0
        page = browse_tree / "viz" / "browse.html"
        assert page.exists()
        html = page.read_text()
        assert mmmview.BROWSE_SIGNATURE in html
        assert f"browse {page}" in capsys.readouterr().out

    def test_header_states_staleness_and_regeneration_command(
            self, roots, browse_tree):
        page = mmmview.write_browse(browse_tree, roots)
        html = page.read_text()
        assert "static snapshot" in html
        assert f"mmmview {browse_tree}" in html
        assert "study" in html

    def test_subdir_with_a_page_links_to_it_others_get_a_recipe(
            self, roots, browse_tree):
        html = mmmview.write_browse(browse_tree, roots).read_text()
        assert 'href="../sub-07/viz/index.html"' in html
        assert "<code>mmmview " in html
        assert str(browse_tree / "plain") in html

    def test_viewable_child_shows_recipe_then_links_once_built(
            self, roots, browse_tree):
        html = mmmview.write_browse(browse_tree, roots).read_text()
        assert "aesthetics.csv" in html
        assert 'href="aesthetics_desc-viewer.html"' not in html
        touch(browse_tree / "viz" / "aesthetics_desc-viewer.html")
        html = mmmview.write_browse(browse_tree, roots).read_text()
        assert 'href="aesthetics_desc-viewer.html"' in html

    def test_existing_bundles_are_listed_with_entity_labels(
            self, roots, browse_tree):
        touch(browse_tree / "viz"
              / "sub-07_task-prf_space-T1w_desc-viewer_prfvariants.html")
        html = mmmview.write_browse(browse_tree, roots).read_text()
        assert "task-prf space-T1w prfvariants" in html

    def test_empty_sections_are_omitted(self, roots, tmp_path):
        d = tmp_path / "bare"
        touch(d / "README.md")
        html = mmmview.write_browse(d, roots).read_text()
        assert "Subdirectories" not in html
        assert "Viewable here" not in html
        assert "Existing bundles" not in html

    def test_page_never_matches_the_deface_viewer_glob(self, roots,
                                                       browse_tree):
        page = mmmview.write_browse(browse_tree, roots)
        assert "desc-viewer" not in page.name
        assert not list(page.parent.glob("*_desc-viewer_*.html"))

    def test_browse_does_not_recurse_into_children(self, roots, browse_tree):
        mmmview.write_browse(browse_tree, roots)
        assert not (browse_tree / "plain" / "viz").exists()
        assert not (browse_tree / "sub-07" / "viz" / "browse.html").exists()

    def test_unchanged_page_is_not_rewritten(self, roots, browse_tree):
        page = mmmview.write_browse(browse_tree, roots)
        before = page.stat().st_mtime_ns
        assert mmmview.write_browse(browse_tree, roots) == page
        assert page.stat().st_mtime_ns == before

    def test_placeable_directory_keeps_todays_behavior(self, roots, tmp_path,
                                                       capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "maps"
        ones = np.ones((3, 3, 2), np.float32)
        u = d / "u.nii.gz"
        d.mkdir()
        nib.save(nib.Nifti1Image(ones, np.eye(4)), u)
        nib.save(nib.Nifti1Image(ones, np.eye(4)),
                 d / "sub-07_space-T1w_stat-z_statmap.nii.gz")
        rc = mmmview.main([str(d), "--no-open", "--underlay", str(u)])
        assert rc == 0 and "wrote" in capsys.readouterr().out
        assert not (d / "viz" / "browse.html").exists()

    def test_file_input_still_exits_2(self, roots, tmp_path, capsys,
                                      monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        p = touch(tmp_path / "README.md")
        assert mmmview.main([str(p), "--no-open"]) == 2
        assert "--underlay/--mesh" in capsys.readouterr().err


class TestSourcedataGuard:
    @pytest.mark.parametrize("leaf", ["sub-07/ses-01",
                                      "sub-07/ses-01/dicom/x.nii.gz"])
    def test_mmmsourcedata_paths_are_refused(self, roots, tmp_path, capsys,
                                             monkeypatch, leaf):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        p = tmp_path / "mmmsourcedata" / leaf
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix:
            touch(p)
        else:
            p.mkdir(parents=True, exist_ok=True)
        assert mmmview.main([str(p), "--no-open"]) != 0
        err = capsys.readouterr().err
        assert "mmmsourcedata" in err and "refusing" in err

    def test_guard_sees_through_a_symlink(self, roots, tmp_path, capsys,
                                          monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        real = tmp_path / "mmmsourcedata" / "sub-07"
        real.mkdir(parents=True)
        link = tmp_path / "shortcut"
        link.symlink_to(real)
        assert mmmview.main([str(link), "--no-open"]) != 0
        assert "mmmsourcedata" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# serve mode (mmmview-browse increment 2): the same index generated per
# request, with build-on-demand links and Range support
# ---------------------------------------------------------------------------

class TestServe:
    @pytest.fixture
    def server(self, roots, tmp_path):
        """A live server on an ephemeral port over a fake tree, torn down
        with the fixture."""
        import threading
        root = tmp_path / "served"
        touch(root / "README.md")
        touch(root / "media.m4a", "0123456789" * 40)
        touch(root / "sub-07" / "sub-07_space-T1w_stat-z_statmap.nii.gz")
        (root / "sub-07" / "viz").mkdir(parents=True)
        (root / "empty").mkdir()
        srv = mmmview.make_server(root, roots, Opts(), "127.0.0.1", 0)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        yield srv, root, srv.server_address[1]
        srv.shutdown()
        srv.server_close()

    def get(self, port, path, headers=None):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", path, headers=headers or {})
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, dict(r.getheaders()), body

    def test_directory_get_returns_the_signed_browse_page(self, server):
        _, _, port = server
        status, _, body = self.get(port, "/")
        assert status == 200
        assert mmmview.BROWSE_SIGNATURE.encode() in body
        assert b"sub-07" in body

    def test_served_page_is_not_written_to_disk(self, server):
        _, root, port = server
        self.get(port, "/")
        assert not (root / "viz").exists()

    def test_unbuilt_viewable_links_to_build_endpoint(self, server):
        _, _, port = server
        _, _, body = self.get(port, "/")
        assert b"/__build?path=sub-07" in body

    def test_file_get_honors_range(self, server):
        _, root, port = server
        size = (root / "media.m4a").stat().st_size
        status, hdrs, body = self.get(port, "/media.m4a",
                                      {"Range": "bytes=0-99"})
        assert status == 206
        assert hdrs["Content-Range"] == f"bytes 0-99/{size}"
        assert len(body) == 100
        assert hdrs["Content-Type"] == "audio/mp4"

    def test_file_get_without_range_is_whole(self, server):
        _, root, port = server
        size = (root / "media.m4a").stat().st_size
        status, _, body = self.get(port, "/media.m4a")
        assert status == 200 and len(body) == size

    @pytest.mark.parametrize("path", ["/../../etc/passwd", "/sub-07/../../x"])
    def test_traversal_is_refused(self, server, path):
        _, _, port = server
        status, _, _ = self.get(port, path)
        assert status == 403

    def test_mmmsourcedata_is_refused(self, server):
        _, root, port = server
        (root / "mmmsourcedata" / "sub-07").mkdir(parents=True)
        status, _, body = self.get(port, "/mmmsourcedata/sub-07")
        assert status == 403 and b"mmmsourcedata" in body

    def test_build_endpoint_redirects_to_the_bundle(self, server,
                                                    monkeypatch):
        _, root, port = server

        def fake_render(plan, force=False):
            plan.out.parent.mkdir(parents=True, exist_ok=True)
            plan.out.write_text("bundle")
            return plan.out, True

        monkeypatch.setattr(mmmview, "render", fake_render)
        status, hdrs, _ = self.get(port, "/__build?path=sub-07")
        assert status in (302, 303)
        assert hdrs["Location"].startswith("/sub-07/viz/")
        assert hdrs["Location"].endswith("_desc-viewer_statmap.html")

    def test_build_of_an_unplaceable_path_is_422_not_a_traceback(self,
                                                                 server):
        _, _, port = server
        status, _, body = self.get(port, "/__build?path=empty")
        assert status == 422
        assert b"Traceback" not in body

    def test_build_outside_the_root_is_refused(self, server):
        _, _, port = server
        status, _, _ = self.get(port, "/__build?path=../outside")
        assert status == 403

    def test_serve_refuses_a_sourcedata_root(self, tmp_path, capsys,
                                             monkeypatch, roots):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "mmmsourcedata" / "sub-07"
        d.mkdir(parents=True)
        assert mmmview.main(["serve", str(d), "--no-open"]) != 0
        assert "mmmsourcedata" in capsys.readouterr().err

    def test_serve_verb_does_not_disturb_path_parsing(self, roots, zmap,
                                                      capsys, monkeypatch):
        # `mmmview PATH` must keep working exactly as before the verb split
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        p, u, _ = zmap
        assert mmmview.main([str(p), "--no-open", "--underlay", str(u)]) == 0
        assert "_desc-viewer_statmap.html" in capsys.readouterr().out

    def test_served_subdir_links_are_live_not_a_static_snapshot(self, server):
        # a child that already has a browse.html on disk must still be
        # addressed live — serving yesterday's snapshot is the staleness
        # this mode exists to avoid
        _, root, port = server
        touch(root / "sub-07" / "viz" / "browse.html", "stale")
        _, _, body = self.get(port, "/")
        assert b'href="/sub-07"' in body
        assert b"/sub-07/viz/browse.html" not in body


# ---------------------------------------------------------------------------
# reap (mmmview-browse increment 3): provenance-keyed removal of bundles
# whose inputs moved on. Dry-run by default; signature-gated deletion only.
# ---------------------------------------------------------------------------

class TestReap:
    @pytest.fixture
    def reaped(self, roots, tmp_path):
        """A subject dir with two claimed bundles (keys current), one
        superseded orphan, and one look-alike carrying no signature."""
        d = tmp_path / "sub-07"
        touch(d / "sub-07_space-T1w_stat-z_statmap.nii.gz", "z")
        touch(d / "sub-07_space-T1w_desc-brain_mask.nii.gz", "m")
        viz = d / "viz"
        claimed = []
        for t in classify(d):
            plan = resolve(t, roots, Opts())
            touch(viz / plan.out.name,
                  f"<html>notes mmmview-key: {plan.key} </html>")
            claimed.append(plan.out.name)
        # superseded: signed by mmmview, no longer claimed by any plan
        touch(viz / "sub-07_space-T1w_desc-viewer_oldvariant.html",
              "<html>mmmview-key: " + "0" * 64 + "</html>")
        # a look-alike with no signature: NOT ours (the qc/ montage lesson)
        touch(viz / "sub-07_space-T1w_desc-viewer_montage.html",
              "<html>a deface montage</html>")
        mmmview.write_index(viz)
        return d, viz, claimed

    def run(self, args):
        return mmmview.main(["reap"] + args)

    def test_dry_run_reports_and_deletes_nothing(self, roots, reaped, capsys,
                                                 monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d, viz, claimed = reaped
        before = sorted(p.name for p in viz.iterdir())
        assert self.run([str(d)]) == 0
        out = capsys.readouterr().out
        assert "stale-orphan" in out
        assert "sub-07_space-T1w_desc-viewer_oldvariant.html" in out
        assert sorted(p.name for p in viz.iterdir()) == before

    def test_current_bundles_are_kept(self, roots, reaped, capsys,
                                      monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d, _, claimed = reaped
        self.run([str(d)])
        out = capsys.readouterr().out
        for name in claimed:
            assert any(line.startswith("keep") and name in line
                       for line in out.splitlines())

    def test_unsigned_lookalike_is_invisible(self, roots, reaped, capsys,
                                             monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d, _, _ = reaped
        self.run([str(d)])
        assert "montage" not in capsys.readouterr().out

    def test_changed_input_is_stale_key(self, roots, reaped, capsys,
                                        monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d, _, claimed = reaped
        (d / "sub-07_space-T1w_stat-z_statmap.nii.gz").write_text("changed")
        self.run([str(d)])
        out = capsys.readouterr().out
        stale = [l for l in out.splitlines() if l.startswith("stale-key")]
        assert len(stale) == 1 and "statmap" in stale[0]

    def test_yes_deletes_stale_and_rewrites_the_index(self, roots, reaped,
                                                      capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d, viz, claimed = reaped
        assert self.run([str(d), "--yes"]) == 0
        names = sorted(p.name for p in viz.iterdir())
        assert "sub-07_space-T1w_desc-viewer_oldvariant.html" not in names
        assert "sub-07_space-T1w_desc-viewer_montage.html" in names
        for name in claimed:
            assert name in names
        html = (viz / "index.html").read_text()
        assert "oldvariant" not in html
        for name in claimed:
            assert name in html

    def test_features_bundle_is_unverifiable_and_never_deleted(
            self, roots, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "feat"
        touch(d / "aesthetics.csv", "x")
        touch(d / "aesthetics.meta.json",
              json.dumps({"extractor": "viz2psy", "input": {"paths": []}}))
        bundle = touch(d / "viz" / "aesthetics_desc-viewer.html", "<html>")
        assert self.run([str(d), "--yes"]) == 0
        out = capsys.readouterr().out
        assert "unverifiable" in out and "no key sidecar" in out
        assert bundle.exists()

    def test_legacy_qc_dirs_are_swept_too(self, roots, tmp_path, capsys,
                                          monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "sub-07"
        touch(d / "sub-07_space-T1w_stat-z_statmap.nii.gz", "z")
        touch(d / "qc" / "sub-07_space-T1w_desc-viewer_oldvariant.html",
              "<html>mmmview-key: " + "0" * 64 + "</html>")
        assert self.run([str(d)]) == 0
        assert "stale-orphan" in capsys.readouterr().out

    def test_paths_from_restricts_the_walk(self, roots, tmp_path, capsys,
                                           monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        root = tmp_path / "deriv"
        for sub in ("sub-07", "sub-06"):
            touch(root / sub / f"{sub}_space-T1w_stat-z_statmap.nii.gz", "z")
            touch(root / sub / "viz" / f"{sub}_desc-viewer_oldvariant.html",
                  "<html>mmmview-key: " + "0" * 64 + "</html>")
        listing = touch(tmp_path / "changed.txt", str(root / "sub-07") + "\n")
        assert self.run([str(root), "--paths-from", str(listing)]) == 0
        out = capsys.readouterr().out
        assert "sub-07_desc-viewer_oldvariant.html" in out
        assert "sub-06_desc-viewer_oldvariant.html" not in out

    def test_totals_are_printed(self, roots, reaped, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d, _, _ = reaped
        self.run([str(d)])
        out = capsys.readouterr().out
        assert "1 stale" in out and "dry run" in out

    def test_mmmsourcedata_is_refused(self, roots, tmp_path, capsys,
                                      monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "mmmsourcedata" / "sub-07"
        d.mkdir(parents=True)
        assert self.run([str(d)]) != 0
        assert "mmmsourcedata" in capsys.readouterr().err

    def test_bundles_whose_maps_live_deeper_are_not_orphans(
            self, roots, tmp_path, capsys, monkeypatch):
        # viz_dir_for walks up to sub-##, so a subject's bundles land in
        # <sub>/viz while its maps sit in <sub>/func or <sub>/ses-##/func.
        # Recomputing claims from viz.parent alone would call these live
        # bundles orphans — and --yes would delete them.
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "sub-07"
        m = touch(d / "ses-04" / "func"
                  / "sub-07_space-T1w_stat-z_statmap.nii.gz", "z")
        (t,) = classify(m)
        plan = resolve(t, roots, Opts())
        assert plan.out.parent == d / "viz"
        touch(plan.out, f"<html>mmmview-key: {plan.key}</html>")
        assert self.run([str(d)]) == 0
        out = capsys.readouterr().out
        assert not [l for l in out.splitlines() if l.startswith("stale")]
        assert out.splitlines()[0].startswith("keep")
        assert "0 stale" in out

    def test_single_map_bundle_is_claimed_not_orphaned(self, roots, tmp_path,
                                                       capsys, monkeypatch):
        # `mmmview <one map>` writes a bundle whose name carries contrast-
        # and stat-; the directory-level merge claims a broader name, so
        # recomputing claims from the directory alone would orphan it
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "sub-07"
        one = touch(d / "func" / "sub-07_space-T1w_contrast-faceVsObject_"
                    "stat-z_statmap.nii.gz", "z")
        touch(d / "func" / "sub-07_space-T1w_contrast-placeVsObject_"
              "stat-z_statmap.nii.gz", "z2")
        (t,) = classify(one)
        plan = resolve(t, roots, Opts())
        assert "contrast-faceVsObject" in plan.out.name
        touch(plan.out, f"<html>mmmview-key: {plan.key}</html>")
        assert self.run([str(d)]) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0].startswith("keep")
        assert "0 stale" in out


# ---------------------------------------------------------------------------
# agent artifacts (mmmview-browse increment 4): agent-written HTML appears
# in the surface, labelled by its provenance sidecar, and is never reaped
# ---------------------------------------------------------------------------

class TestAgentArtifacts:
    def artifact(self, viz, slug="prfnotes", sidecar=True):
        name = f"sub-07_desc-agent_{slug}.html"
        touch(viz / name, "<html>an agent wrote this</html>")
        if sidecar:
            touch(viz / f"sub-07_desc-agent_{slug}.prov.json",
                  json.dumps({"author": "mmmdata-qc agent",
                              "date": "2026-09-15",
                              "inputs": ["sub-##/func/sub-##_bold.nii.gz"],
                              "command": "mmmview ..."}))
        return viz / name

    def test_index_lists_it_in_its_own_group(self, tmp_path):
        viz = tmp_path / "sub-07" / "viz"
        touch(viz / "sub-07_desc-viewer_statmap.html")
        self.artifact(viz)
        html = mmmview.write_index(viz).read_text()
        assert "Agent-generated" in html
        assert "sub-07_desc-agent_prfnotes.html" in html
        assert "mmmdata-qc agent" in html

    def test_browse_page_has_an_agent_section(self, roots, tmp_path):
        d = tmp_path / "study"
        touch(d / "README.md")
        self.artifact(d / "viz")
        html = mmmview.write_browse(d, roots).read_text()
        assert "Agent-generated" in html
        assert "mmmdata-qc agent" in html and "2026-09-15" in html

    def test_missing_sidecar_is_flagged_unattributed(self, roots, tmp_path):
        d = tmp_path / "study"
        touch(d / "README.md")
        self.artifact(d / "viz", slug="hunch", sidecar=False)
        html = mmmview.write_browse(d, roots).read_text()
        assert "unattributed" in html
        assert "sub-07_desc-agent_hunch.html" in html

    def test_unreadable_sidecar_is_unattributed_not_an_error(self, roots,
                                                             tmp_path):
        d = tmp_path / "study"
        touch(d / "README.md")
        self.artifact(d / "viz", slug="broken")
        touch(d / "viz" / "sub-07_desc-agent_broken.prov.json", "{not json")
        html = mmmview.write_browse(d, roots).read_text()
        assert "unattributed" in html

    def test_reap_never_lists_an_agent_artifact(self, roots, tmp_path,
                                                capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        d = tmp_path / "sub-07"
        touch(d / "sub-07_space-T1w_stat-z_statmap.nii.gz", "z")
        art = self.artifact(d / "viz")
        assert mmmview.main(["reap", str(d), "--yes"]) == 0
        assert "desc-agent" not in capsys.readouterr().out
        assert art.exists()

    def test_deface_report_glob_does_not_match_it(self, tmp_path):
        viz = tmp_path / "viz"
        art = self.artifact(viz)
        assert art not in list(viz.glob("*_desc-viewer_*.html"))
        assert list(viz.glob("*_desc-agent_*.html")) == [art]

    def test_artifacts_are_not_classified_as_data(self, tmp_path):
        viz = tmp_path / "viz"
        self.artifact(viz)
        with pytest.raises(Unplaceable):
            classify(viz)


# ---------------------------------------------------------------------------
# catalog annotation (mmmview-browse increment 5): enrichment only — every
# page must render the same without it
# ---------------------------------------------------------------------------

def make_catalog(db, rows, started_at="2026-01-02T03:04:05+00:00"):
    """A minimal Contract A catalog: the `files` columns the annotation
    reads, plus a sweep_meta report. rows = (dataset_relpath, path, suffix)."""
    duckdb = pytest.importorskip("duckdb")
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE files (dataset_relpath VARCHAR, path VARCHAR, "
                "suffix VARCHAR)")
    con.executemany("INSERT INTO files VALUES (?, ?, ?)", rows)
    con.execute("CREATE TABLE sweep_meta (key VARCHAR, value VARCHAR)")
    con.execute("INSERT INTO sweep_meta VALUES ('report', ?)",
                [json.dumps({"started_at": started_at})])
    con.close()
    return db


@pytest.fixture
def cataloged(tmp_path):
    """BIDS root with raw sub-## dirs and an fMRIPrep derivative, and a
    catalog describing both (plus rows for a sibling derivative that must
    not leak into fMRIPrep's counts)."""
    bids = tmp_path / "ds"
    deriv = bids / "derivatives"
    for sub in ("sub-01", "sub-02"):
        (bids / sub / "ses-01").mkdir(parents=True)
    (deriv / "fmriprep" / "sub-01" / "ses-01").mkdir(parents=True)
    (deriv / "fmriprep" / "logs").mkdir(parents=True)
    rows = [(".", "sub-01/ses-01/func/a_bold.nii.gz", "bold"),
            (".", "sub-01/ses-01/func/b_bold.nii.gz", "bold"),
            (".", "sub-01/ses-01/func/a_events.tsv", "events"),
            (".", "sub-01/ses-01/anat/a_T1w.nii.gz", "T1w"),
            (".", "sub-010/ses-01/func/a_bold.nii.gz", "bold"),
            ("derivatives/fmriprep", "sub-01/ses-01/func/a_boldref.nii.gz",
             "boldref"),
            ("derivatives/mriqc", "sub-01/ses-01/func/a_bold.json", "bold")]
    db = make_catalog(bids / "inventory" / "catalog.duckdb", rows)
    return Roots(deriv=deriv, bids=bids, catalog_db=db)


def _subdir(model, name):
    return next(s for s in model["subdirs"] if s["name"] == name)


class TestCatalogAnnotation:
    def test_subject_rows_get_file_counts_by_suffix(self, cataloged):
        model = mmmview.browse_model(cataloged.bids, cataloged)
        assert _subdir(model, "sub-01")["counts"] == {"bold": 2, "events": 1,
                                                     "T1w": 1}
        assert _subdir(model, "derivatives")["counts"] is None
        assert model["catalog_date"] == "2026-01-02"

    def test_a_label_prefix_does_not_capture_a_longer_label(self, cataloged):
        model = mmmview.browse_model(cataloged.bids, cataloged)
        assert _subdir(model, "sub-02")["counts"] is None

    def test_derivative_rows_count_only_their_own_dataset(self, cataloged):
        model = mmmview.browse_model(cataloged.deriv / "fmriprep", cataloged)
        assert _subdir(model, "sub-01")["counts"] == {"boldref": 1}
        assert _subdir(model, "logs")["counts"] is None

    def test_session_rows_are_annotated_below_a_subject(self, cataloged):
        model = mmmview.browse_model(cataloged.bids / "sub-01", cataloged)
        assert _subdir(model, "ses-01")["counts"]["bold"] == 2

    def test_counts_resolve_through_a_derivatives_root_outside_bids(
            self, cataloged, tmp_path):
        # the Mac staging layout: output_dir is not <bids>/derivatives
        staged = tmp_path / "staged"
        (staged / "fmriprep" / "sub-01").mkdir(parents=True)
        roots = Roots(deriv=staged, bids=cataloged.bids,
                      catalog_db=cataloged.catalog_db)
        model = mmmview.browse_model(staged / "fmriprep", roots)
        assert _subdir(model, "sub-01")["counts"] == {"boldref": 1}

    def test_page_shows_counts_and_the_sweep_date(self, cataloged):
        html = mmmview.write_browse(cataloged.bids, cataloged).read_text()
        assert "4 files" in html and "2 bold" in html
        assert "catalog swept 2026-01-02" in html

    def test_missing_catalog_renders_the_same_page_silently(
            self, cataloged, capsys):
        with_db = mmmview.write_browse(cataloged.bids, cataloged).read_text()
        bare = Roots(deriv=cataloged.deriv, bids=cataloged.bids,
                     catalog_db=cataloged.bids / "nope.duckdb")
        page = mmmview.write_browse(cataloged.bids, bare)
        html = page.read_text()
        assert "files" not in html.split("<main>")[1].split("Subdirectories")[0]
        assert "catalog swept" not in html and "2 bold" not in html
        assert "sub-01/" in html and "sub-01/" in with_db
        assert capsys.readouterr().err == ""

    def test_unreadable_catalog_renders_without_counts_and_notes_it(
            self, cataloged, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "_catalog_note_sent", False)
        cataloged.catalog_db.write_text("not a database")
        model = mmmview.browse_model(cataloged.bids, cataloged)
        assert _subdir(model, "sub-01")["counts"] is None
        assert model["catalog_date"] is None
        mmmview.browse_model(cataloged.bids, cataloged)
        err = capsys.readouterr().err
        assert err.count("catalog annotation skipped") == 1

    def test_duckdb_import_failure_renders_without_counts(
            self, cataloged, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "_catalog_note_sent", False)
        # hide an already-imported module too: `from core import catalog`
        # reads the package attribute before sys.modules
        import core
        monkeypatch.delattr(core, "catalog", raising=False)
        monkeypatch.setitem(sys.modules, "core.catalog", None)
        model = mmmview.browse_model(cataloged.bids, cataloged)
        assert _subdir(model, "sub-01")["counts"] is None
        assert "catalog annotation skipped" in capsys.readouterr().err

    def test_serve_mode_pages_carry_the_same_annotation(self, cataloged):
        model = mmmview.browse_model(cataloged.bids, cataloged)
        html = mmmview.render_browse(model, str, dir_link=str)
        assert "2 bold" in html


# ---------------------------------------------------------------------------
# label maps (dseg): template atlases over their TemplateFlow T1w, regions
# named from the sibling lookup table
# ---------------------------------------------------------------------------

TPL = "tpl-MNI152NLin2009cAsym"


def _dseg(path, n_labels):
    data = np.zeros((4, 4, 3), np.int16)
    data.flat[1:n_labels + 1] = np.arange(1, n_labels + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def _lut(path, names, color="#781283"):
    rows = ["index\tname\tcolor"] + [f"{i}\t{n}\t{color}"
                                     for i, n in enumerate(names, 1)]
    return touch(path, "\n".join(rows) + "\n")


@pytest.fixture
def atlas_dir(tmp_path):
    """An atlases-style tpl-*/anat directory: Schaefer-like dsegs in two
    network solutions at three scales (named so a lexical sort would put
    1000 before 200), each with its LUT, plus the TemplateFlow T1w —
    which spells resolution res-02 where the atlases say res-2."""
    anat = tmp_path / "atlases" / TPL / "anat"
    for seg in ("7n", "17n"):
        for scale in ("100", "200", "1000"):
            stem = f"{TPL}_atlas-Schaefer2018_seg-{seg}_scale-{scale}_res-2_dseg"
            _dseg(anat / f"{stem}.nii.gz", 3)
            _lut(anat / f"{stem}.tsv",
                 [f"{seg}_LH_Vis_1", f"{seg}_LH_Vis_2", f"{seg}_LH_Vis_3"])
    t1 = anat / f"{TPL}_res-02_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 3), np.float32), np.eye(4)), t1)
    return anat


class TestLabelMaps:
    @pytest.mark.parametrize("ents", [{"tpl": "MNI152NLin2009cAsym"},
                                      {"sub": "07", "desc": "aparcaseg"}])
    def test_dseg_family(self, ents):
        assert family_of(ents, "dseg") == "dseg"

    def test_atlas_directory_is_one_target_without_the_template(
            self, atlas_dir):
        [t] = classify(atlas_dir)
        assert t.kind == "volume" and t.suffix == "dseg"
        assert len(t.maps) == 6
        assert all(m.name.endswith("_dseg.nii.gz") for m in t.maps)
        assert t.entities == {"tpl": "MNI152NLin2009cAsym",
                              "atlas": "Schaefer2018", "res": "2"}

    def test_two_atlases_in_one_directory_are_two_targets(self, atlas_dir):
        _dseg(atlas_dir / f"{TPL}_atlas-HarvardOxford_res-2_dseg.nii.gz", 2)
        atlases = sorted(t.entities["atlas"] for t in classify(atlas_dir))
        assert atlases == ["HarvardOxford", "Schaefer2018"]

    def test_template_underlay_matches_resolution_numerically(
            self, roots, atlas_dir):
        plan = resolve(classify(atlas_dir)[0], roots)
        assert plan.inputs["underlay"] == atlas_dir / f"{TPL}_res-02_T1w.nii.gz"
        assert plan.out == (atlas_dir / "viz" / f"{TPL}_atlas-Schaefer2018_"
                            "res-2_desc-viewer_dseg.html")
        assert TPL in plan.title and "atlas-Schaefer2018" in plan.title

    def test_template_underlay_is_found_in_the_tpl_root(self, roots,
                                                        atlas_dir):
        t1 = atlas_dir / f"{TPL}_res-02_T1w.nii.gz"
        t1.rename(atlas_dir.parent / t1.name)
        plan = resolve(classify(atlas_dir)[0], roots)
        assert plan.inputs["underlay"] == atlas_dir.parent / t1.name

    def test_missing_template_names_the_flag_and_the_file(self, roots,
                                                          atlas_dir):
        (atlas_dir / f"{TPL}_res-02_T1w.nii.gz").unlink()
        with pytest.raises(Unplaceable) as e:
            resolve(classify(atlas_dir)[0], roots)
        assert "--underlay" in str(e.value)
        assert f"{TPL}_res-02_T1w.nii.gz" in str(e.value)

    def test_selector_axes_are_the_varying_entities_in_natural_order(
            self, roots, atlas_dir):
        plan = resolve(classify(atlas_dir)[0], roots)
        labels = list(dict.fromkeys(e["label"] for e in plan.display))
        variants = list(dict.fromkeys(e["variant"] for e in plan.display))
        assert labels == ["seg-7n", "seg-17n"]
        assert variants == ["scale-100", "scale-200", "scale-1000"]

    def test_bundle_config_carries_named_regions(self, roots, atlas_dir):
        plan = resolve(classify(atlas_dir)[0], roots)
        out, built = render(plan)
        cfg = _config(out.read_text())
        assert cfg["overlay_title"] == "seg" and cfg["variant_title"] == "scale"
        overlays = [v for v in cfg["volumes"] if not v["isUnderlay"]]
        assert len(overlays) == 6
        lut = overlays[0]["lut"]
        assert lut["I"][0] == 0 and lut["A"][0] == 0      # background clear
        assert lut["labels"][1:] == ["7n_LH_Vis_1", "7n_LH_Vis_2",
                                     "7n_LH_Vis_3"]
        assert overlays[0]["visible"] and overlays[0]["cal_max"] is None
        # one network colour in the table; parcels still tell apart
        rgb = set(zip(lut["R"][1:], lut["G"][1:], lut["B"][1:]))
        assert len(rgb) == 3

    def test_parcel_colours_keep_the_network_hue(self, roots, atlas_dir):
        import colorsys
        plan = resolve(classify(atlas_dir)[0], roots)
        lut = mmmview.label_lut(plan.display[0])
        base_h = colorsys.rgb_to_hls(0x78 / 255, 0x12 / 255, 0x83 / 255)[0]
        for r, g, b in zip(lut["R"][1:], lut["G"][1:], lut["B"][1:]):
            h = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)[0]
            assert abs(h - base_h) < 0.02

    def test_lookup_table_edit_rebuilds(self, roots, atlas_dir):
        plan = resolve(classify(atlas_dir)[0], roots)
        render(plan)
        tsv = next(atlas_dir.glob("*seg-7n_scale-100_res-2_dseg.tsv"))
        _lut(tsv, ["renamed_1", "renamed_2", "renamed_3"])
        plan2 = resolve(classify(atlas_dir)[0], roots)
        assert plan2.key != plan.key
        assert tsv.name in plan2.notes

    def test_spaceless_anatomical_dseg_gets_the_native_t1w(self, roots,
                                                           tmp_path):
        p = _dseg(tmp_path / "anat" / "sub-07_acq-MPR_desc-aparcaseg_dseg.nii.gz",
                  2)
        plan = resolve(classify(p)[0], roots)
        assert plan.inputs["underlay"].name == "sub-07_acq-MPR_desc-preproc_T1w.nii.gz"

    def test_dseg_without_a_table_gets_numbered_regions(self, roots,
                                                        tmp_path):
        p = _dseg(tmp_path / "anat" / "sub-07_desc-aparcaseg_dseg.nii.gz", 4)
        u = tmp_path / "u.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((4, 4, 3), np.float32), np.eye(4)), u)
        plan = resolve(classify(p)[0], roots, Opts(underlay=str(u)))
        assert any("lookup table" in m for m in plan.messages)
        lut = mmmview.label_lut(plan.display[0])
        assert lut["labels"][1:] == ["1", "2", "3", "4"]
        assert len(set(zip(lut["R"], lut["G"], lut["B"]))) == 5

    def test_cli_on_an_atlas_directory_builds_a_bundle_not_a_browse_page(
            self, roots, atlas_dir, capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        assert mmmview.main([str(atlas_dir), "--no-open"]) == 0
        assert "wrote" in capsys.readouterr().out
        assert list((atlas_dir / "viz").glob("*_desc-viewer_dseg.html"))
        assert not (atlas_dir / "viz" / "browse.html").exists()


# ---------------------------------------------------------------------------
# crosshair readout: each overlay tells the page how to describe its value
# at the crosshair, and the bundle says what its mm coordinates are
# ---------------------------------------------------------------------------

def _overlays(out):
    return [v for v in _config(out.read_text())["volumes"]
            if not v["isUnderlay"]]


class TestCrosshairReadout:
    def test_prf_maps_carry_quantity_units_and_mask_meaning(self, roots,
                                                            tmp_path):
        d = tmp_path / "x"
        u = d / "u.nii.gz"
        d.mkdir()
        nib.save(nib.Nifti1Image(np.ones((3, 3, 2), np.float32), np.eye(4)), u)
        for desc, val in (("R2", 50.0), ("angle", 90.0),
                          ("eccentricity", 3.0)):
            nib.save(nib.Nifti1Image(np.full((3, 3, 2), val, np.float32),
                                     np.eye(4)),
                     d / f"sub-07_task-prf_space-T1w_desc-{desc}_prf.nii.gz")
        (t,) = classify(d)
        out, _ = render(resolve(t, roots, Opts(underlay=str(u))))
        got = {v["label"]: v["readout"] for v in _overlays(out)}
        assert got["angle"] == {"kind": "value", "quantity": "angle",
                                "unit": "°", "masked": "below R² floor"}
        assert got["eccentricity"]["unit"] == "°"
        assert got["R2"]["unit"] == "%"
        assert got["R2"]["masked"] == "no fit"     # R2 itself is unmasked

    def test_glm_tails_share_a_group_and_carry_their_sign(self, roots, zmap):
        p, u, _ = zmap
        out, _ = render(resolve(classify(p)[0], roots, Opts(underlay=str(u))))
        pos, neg = (v["readout"] for v in _overlays(out))
        assert pos["kind"] == neg["kind"] == "tail"
        assert pos["quantity"] == neg["quantity"] == "z"
        assert (pos["sign"], neg["sign"]) == (1, -1)
        assert pos["group"] == neg["group"] == p.name

    def test_unknown_maps_read_out_a_plain_value(self, roots, tmp_path):
        p = tmp_path / "x" / "sub-07_space-T1w_desc-foo_bold.nii.gz"
        p.parent.mkdir()
        nib.save(nib.Nifti1Image(np.ones((3, 3, 2), np.float32), np.eye(4)), p)
        out, _ = render(resolve(classify(p)[0], roots,
                                Opts(underlay=str(p))))
        [ov] = _overlays(out)
        assert ov["readout"] == {"kind": "value", "quantity": None,
                                 "unit": "", "masked": "no data"}

    def test_label_maps_read_out_regions(self, roots, atlas_dir):
        out, _ = render(resolve(classify(atlas_dir)[0], roots))
        assert all(v["readout"] == {"kind": "label"}
                   for v in _overlays(out))

    @pytest.mark.parametrize("ents,label", [
        ({"sub": "07", "space": "MNI152NLin2009cAsym"},
         "MNI152NLin2009cAsym mm"),
        ({"tpl": "MNI152NLin2009cAsym"}, "MNI152NLin2009cAsym mm"),
        ({"sub": "07", "space": "T1w"}, "scanner mm (native T1w)"),
        ({"sub": "07"}, "scanner mm (native T1w)"),
        ({"sub": "07", "ses": "04", "task": "floc"},
         "scanner mm (native func)"),
    ])
    def test_coordinate_label_names_the_space(self, ents, label):
        assert mmmview.coord_label(ents) == label

    def test_bundle_config_carries_the_coordinate_label(self, roots,
                                                         atlas_dir):
        out, _ = render(resolve(classify(atlas_dir)[0], roots))
        cfg = _config(out.read_text())
        assert cfg["coord_label"] == "MNI152NLin2009cAsym mm"
        assert 'id="readout"' in out.read_text()

    def test_viewer_version_is_part_of_the_key(self, roots, zmap,
                                               monkeypatch):
        p, u, _ = zmap
        plan = resolve(classify(p)[0], roots, Opts(underlay=str(u)))
        monkeypatch.setattr(mmmview, "PROFILE_VERSION",
                            mmmview.PROFILE_VERSION + 1)
        assert resolve(classify(p)[0], roots,
                       Opts(underlay=str(u))).key != plan.key

    def test_volumes_carry_their_own_voxel_grid(self, roots, tmp_path):
        # an LPS-stored map: the page must report the file's i,j,k, not
        # NiiVue's RAS-reoriented ones
        d = tmp_path / "x"
        d.mkdir()
        aff = np.diag([-2.0, -2.0, 2.0, 1.0])
        aff[:3, 3] = [10.0, 20.0, -30.0]
        p = d / "sub-07_space-T1w_desc-foo_bold.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((5, 6, 7), np.float32), aff), p)
        u = d / "u.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((3, 3, 2), np.float32), np.eye(4)), u)
        out, _ = render(resolve(classify(p)[0], roots, Opts(underlay=str(u))))
        cfg = _config(out.read_text())
        [ov] = [v for v in cfg["volumes"] if not v["isUnderlay"]]
        assert ov["grid"]["shape"] == [5, 6, 7]
        m = np.array(ov["grid"]["mm2vox"])
        ijk = m @ np.array([6.0, 14.0, -24.0, 1.0])     # voxel (2, 3, 3)
        assert np.allclose(ijk, [2, 3, 3])
        und = next(v for v in cfg["volumes"] if v["isUnderlay"])
        assert und["grid"]["shape"] == [3, 3, 2]


# ---------------------------------------------------------------------------
# results kind (mmmview-browse increment 6): long-format TSV + Vega-Lite spec
# ---------------------------------------------------------------------------

RESULT_TABLE = "subject\tcondition\taccuracy\n01\tsingle\t0.61\n02\tsingle\t0.72\n"


def result_spec(**meta):
    m = {"schema_version": 1, "err_over": "none", **meta}
    return json.dumps({
        "usermeta": {"mmmview": m}, "mark": "bar",
        "encoding": {"x": {"field": "subject", "type": "nominal"},
                     "y": {"field": "accuracy", "type": "quantitative"}}})


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "behavioral_analysis" / "group"
    for stem in ("acc", "dprime"):
        touch(d / f"{stem}.tsv", RESULT_TABLE)
        touch(d / f"{stem}.vl.json", result_spec())
    return d


class TestResults:
    def test_directory_is_one_results_target_over_every_spec(self,
                                                             results_dir):
        (t,) = classify(results_dir)
        assert t.kind == "results"
        assert [p.name for p in t.maps] == ["acc.vl.json", "dprime.vl.json"]

    def test_table_and_spec_each_classify(self, results_dir):
        for name in ("acc.tsv", "acc.vl.json"):
            (t,) = classify(results_dir / name)
            assert t.kind == "results"
            assert [p.name for p in t.maps] == ["acc.vl.json"]

    def test_table_without_spec_names_the_sidecar_to_add(self, tmp_path):
        p = touch(tmp_path / "rt_by_condition.tsv", RESULT_TABLE)
        with pytest.raises(Unplaceable, match=r"rt_by_condition\.vl\.json"):
            classify(p)

    def test_events_tables_still_go_to_the_plot_tools(self, tmp_path):
        p = touch(tmp_path / "sub-07_ses-04_task-floc_events.tsv")
        touch(tmp_path / "sub-07_ses-04_task-floc_events.vl.json",
              result_spec())
        with pytest.raises(Unplaceable, match="plot_"):
            classify(p)

    def test_broken_spec_fails_at_classify_with_the_contract_message(
            self, results_dir):
        touch(results_dir / "acc.vl.json", result_spec(err_over=""))
        with pytest.raises(Unplaceable, match="err_over is required"):
            classify(results_dir)

    def test_output_names_and_place(self, roots, results_dir):
        (t,) = classify(results_dir)
        plan = resolve(t, roots, Opts())
        assert plan.out == results_dir / "viz" / "group_desc-viewer_results.html"
        (t,) = classify(results_dir / "acc.tsv")
        assert resolve(t, roots, Opts()).out.name == "acc_desc-viewer_results.html"

    def test_key_follows_table_content(self, roots, results_dir):
        (t,) = classify(results_dir)
        k1 = resolve(t, roots, Opts()).key
        touch(results_dir / "acc.tsv", RESULT_TABLE + "03\tsingle\t0.55\n")
        mmmview._SHA_CACHE.clear()
        assert resolve(t, roots, Opts()).key != k1

    def test_render_then_reuse(self, roots, results_dir):
        (t,) = classify(results_dir)
        plan = resolve(t, roots, Opts())
        out, built = render(plan)
        assert built and out.exists()
        html = out.read_text()
        assert f"mmmview-key: {plan.key}" in html
        assert "mmmview " + str(results_dir) in html       # regenerate recipe
        assert render(plan) == (out, False)

    def test_reap_keeps_a_current_results_page(self, roots, results_dir,
                                               capsys, monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        for target in (results_dir, results_dir / "acc.tsv"):
            (t,) = classify(target)
            render(resolve(t, roots, Opts()))
        assert mmmview.main(["reap", str(results_dir), "--yes"]) == 0
        out = capsys.readouterr().out
        for name in ("group_desc-viewer_results.html",
                     "acc_desc-viewer_results.html"):
            assert any(l.startswith("keep") and name in l
                       for l in out.splitlines())
            assert (results_dir / "viz" / name).exists()

    def test_index_label(self):
        assert (mmmview._index_label("group_desc-viewer_results.html")
                == "group (results)")

    def test_browse_lists_the_results_dir_as_viewable(self, roots,
                                                      results_dir):
        model = mmmview.browse_model(results_dir.parent, roots)
        (v,) = [e for e in model["viewable"] if e["name"] == "group"]
        assert v["kind"] == "results" and v["label"] == "2 result charts"

    def test_cli_builds_and_indexes(self, roots, results_dir, capsys,
                                    monkeypatch):
        monkeypatch.setattr(mmmview, "load_roots", lambda *_: roots)
        touch(results_dir / "viz" / "acc_desc-viewer_results.html")
        assert mmmview.main([str(results_dir), "--no-open"]) == 0
        out = capsys.readouterr().out
        assert "results)" in out
        assert (results_dir / "viz" / "index.html").exists()


# ---------------------------------------------------------------------------
# "Refresh from current data" (mmmview-browse increment 7): served pages ask
# /__status and POST /__refresh; the truth is reap's (claims + key)
# ---------------------------------------------------------------------------

def _built_results(roots, d, target=None):
    (t,) = classify(target or d)
    plan = resolve(t, roots, Opts())
    render(plan)
    return plan


class TestFreshness:
    def test_current_page(self, roots, results_dir):
        plan = _built_results(roots, results_dir)
        s = mmmview.page_status(plan.out, roots)
        assert s["overall"] == "current" and s["refreshable"]
        assert s["items"] == [{"name": plan.out.name, "state": "current",
                               "basis": "key", "built": s["items"][0]["built"]}]
        assert s["items"][0]["built"]           # parsed from the notes

    def test_changed_table_is_stale(self, roots, results_dir):
        plan = _built_results(roots, results_dir)
        touch(results_dir / "acc.tsv", RESULT_TABLE + "03\tsingle\t0.55\n")
        mmmview._SHA_CACHE.clear()
        s = mmmview.page_status(plan.out, roots)
        assert s["overall"] == "stale"
        assert "data changed since this was built" in s["text"]

    def test_index_aggregates_its_bundles(self, roots, results_dir):
        _built_results(roots, results_dir)
        _built_results(roots, results_dir, results_dir / "acc.tsv")
        idx = mmmview.write_index(results_dir / "viz")
        touch(results_dir / "dprime.tsv", RESULT_TABLE + "03\tsingle\t0.55\n")
        mmmview._SHA_CACHE.clear()
        s = mmmview.page_status(idx, roots)
        states = {i["name"]: i["state"] for i in s["items"]}
        assert states == {"acc_desc-viewer_results.html": "current",
                          "group_desc-viewer_results.html": "stale"}
        assert s["text"] == "1 of 2 outputs are stale"

    def test_unclaimed_page_cannot_be_verified(self, roots, results_dir,
                                               tmp_path):
        (t,) = classify(results_dir)
        plan = resolve(t, roots, Opts(out_dir=str(tmp_path / "else" / "viz")))
        render(plan)
        s = mmmview.page_status(plan.out, roots)
        assert s["overall"] == "unknown" and not s["refreshable"]
        assert s["items"][0]["state"] == "orphan"

    def test_refresh_rebuilds_only_stale_unless_forced(self, roots,
                                                      results_dir):
        plan = _built_results(roots, results_dir)
        assert mmmview.refresh_page(plan.out, roots)["current"] == \
            [plan.out.name]
        touch(results_dir / "acc.tsv", RESULT_TABLE + "03\tsingle\t0.55\n")
        mmmview._SHA_CACHE.clear()
        res = mmmview.refresh_page(plan.out, roots)
        assert res["rebuilt"] == [plan.out.name] and not res["errors"]
        assert mmmview.page_status(plan.out, roots)["overall"] == "current"
        assert mmmview.refresh_page(plan.out, roots, force=True)["rebuilt"] \
            == [plan.out.name]

    def test_refresh_never_deletes_orphans(self, roots, results_dir):
        viz = results_dir / "viz"
        orphan = touch(viz / "gone_desc-viewer_results.html",
                       "<html>mmmview-key: " + "0" * 64 + "</html>")
        _built_results(roots, results_dir)
        idx = mmmview.write_index(viz)
        res = mmmview.refresh_page(idx, roots)
        assert res["orphans"] == ["gone_desc-viewer_results.html"]
        assert orphan.exists()

    def test_render_errors_are_collected(self, roots, results_dir,
                                         monkeypatch):
        plan = _built_results(roots, results_dir)

        def boom(plan, force=False):
            raise mmmview.RenderError("renderer exploded\nstack")
        monkeypatch.setattr(mmmview, "render", boom)
        res = mmmview.refresh_page(plan.out, roots, force=True)
        assert res["errors"] == [{"name": plan.out.name,
                                  "error": "renderer exploded"}]

    def test_widget_is_in_index_and_results_pages(self, roots, results_dir):
        plan = _built_results(roots, results_dir)
        _built_results(roots, results_dir, results_dir / "acc.tsv")
        idx = mmmview.write_index(results_dir / "viz")
        for page in (idx, plan.out):
            html = page.read_text()
            assert "Refresh from current data" in html
            assert '"X-mmmview": "1"' in html


class TestRefreshEndpoints:
    @pytest.fixture
    def served(self, roots, tmp_path):
        import threading
        d = tmp_path / "served" / "group"
        for stem in ("acc", "dprime"):
            touch(d / f"{stem}.tsv", RESULT_TABLE)
            touch(d / f"{stem}.vl.json", result_spec())
        plan = _built_results(roots, d)
        srv = mmmview.make_server(tmp_path / "served", roots, Opts(),
                                  "127.0.0.1", 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        yield d, plan, srv.server_address[1]
        srv.shutdown()
        srv.server_close()

    def call(self, port, method, path, headers=None):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request(method, path, headers=headers or {})
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, json.loads(body) if body else None

    PAGE = "/group/viz/group_desc-viewer_results.html"

    def test_status_json(self, served):
        _, _, port = served
        status, body = self.call(port, "GET", f"/__status?page={self.PAGE}")
        assert status == 200 and body["overall"] == "current"

    def test_refresh_needs_the_header(self, served):
        _, _, port = served
        status, _ = self.call(port, "POST", f"/__refresh?page={self.PAGE}")
        assert status == 403

    def test_refresh_is_post_only(self, served):
        _, _, port = served
        status, _ = self.call(port, "GET", f"/__refresh?page={self.PAGE}")
        assert status == 405

    def test_refresh_rebuilds_a_stale_page(self, served):
        d, plan, port = served
        touch(d / "acc.tsv", RESULT_TABLE + "03\tsingle\t0.55\n")
        mmmview._SHA_CACHE.clear()
        status, body = self.call(port, "POST",
                                 f"/__refresh?page={self.PAGE}",
                                 {"X-mmmview": "1"})
        assert status == 200 and body["rebuilt"] == [plan.out.name]
        _, st = self.call(port, "GET", f"/__status?page={self.PAGE}")
        assert st["overall"] == "current"

    @pytest.mark.parametrize("page,code", [("/../../etc/passwd", 403),
                                           ("/group/acc.tsv", 404),
                                           ("/group/viz/nope.html", 404)])
    def test_bad_pages_are_refused(self, served, page, code):
        _, _, port = served
        status, _ = self.call(port, "GET", f"/__status?page={page}")
        assert status == code

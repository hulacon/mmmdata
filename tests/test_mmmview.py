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

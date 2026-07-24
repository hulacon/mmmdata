"""Tests for neuroimaging.qc_guidance — the QC review guidance registry.

These tests guard the contract the dashboard depends on: every column it
renders carries guidance, every entry is complete, and every citation is a
real link rather than a placeholder.
"""

import re

import pytest


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

class TestRegistryCoverage:

    def test_every_dashboard_column_is_documented(self):
        """No column may reach a reviewer without an explanation."""
        from neuroimaging.qc_guidance import undocumented_keys
        from neuroimaging.qc_dashboard import _DASHBOARD_COLS

        missing = {}
        for modality, cols in _DASHBOARD_COLS.items():
            gaps = undocumented_keys(cols)
            if gaps:
                missing[modality] = gaps
        assert missing == {}, f"Dashboard columns lacking guidance: {missing}"

    def test_derived_motion_columns_are_documented(self):
        """mean_fd/pct_high_motion are computed here, not by MRIQC."""
        from neuroimaging.qc_guidance import undocumented_keys
        assert undocumented_keys(["mean_fd", "pct_high_motion"]) == []

    def test_every_key_iqm_is_documented(self):
        """The full per-modality IQM lists, not just the dashboard subset."""
        from neuroimaging.qc_guidance import undocumented_keys
        from neuroimaging import qc

        missing = {}
        for modality, metrics in qc._KEY_IQMS_BY_MODALITY.items():
            gaps = undocumented_keys(metrics)
            if gaps:
                missing[modality] = gaps
        assert missing == {}, f"Key IQMs lacking guidance: {missing}"

    def test_registry_key_matches_entry_key(self):
        from neuroimaging.qc_guidance import MEASURE_GUIDANCE
        for key, g in MEASURE_GUIDANCE.items():
            assert key == g.key


# ---------------------------------------------------------------------------
# Entry quality
# ---------------------------------------------------------------------------

class TestEntryQuality:

    def test_all_entries_answer_all_three_questions(self):
        """why / look_for / auto_flag are the point of the layer."""
        from neuroimaging.qc_guidance import MEASURE_GUIDANCE
        for key, g in MEASURE_GUIDANCE.items():
            assert g.why.strip(), f"{key} missing 'why'"
            assert g.look_for.strip(), f"{key} missing 'look_for'"
            assert g.auto_flag.strip(), f"{key} missing 'auto_flag'"
            assert g.label.strip(), f"{key} missing label"

    def test_all_entries_carry_at_least_one_reference(self):
        from neuroimaging.qc_guidance import MEASURE_GUIDANCE
        bare = [k for k, g in MEASURE_GUIDANCE.items() if not g.references]
        assert bare == [], f"Measures with no cited source: {bare}"

    def test_all_entries_declare_modalities(self):
        from neuroimaging.qc_guidance import MEASURE_GUIDANCE
        valid = {"bold", "T1w", "T2w", "dwi"}
        for key, g in MEASURE_GUIDANCE.items():
            assert g.modalities, f"{key} declares no modality"
            assert set(g.modalities) <= valid, f"{key} has unknown modality"

    def test_invalid_direction_rejected(self):
        from neuroimaging.qc_guidance import MeasureGuidance
        with pytest.raises(ValueError, match="Invalid direction"):
            MeasureGuidance(
                key="x", label="X", modalities=("bold",), direction="sideways",
                why="w", look_for="l", auto_flag="a",
            )

    def test_invalid_reference_kind_rejected(self):
        from neuroimaging.qc_guidance import Reference
        with pytest.raises(ValueError, match="Invalid reference kind"):
            Reference(label="L", detail="D", url="https://x", kind="rumour")


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

class TestReferences:

    def test_all_urls_are_resolvable_links(self):
        from neuroimaging.qc_guidance import all_references
        for r in all_references():
            assert r.url.startswith("https://"), f"{r.label}: {r.url!r}"

    def test_no_placeholder_citations(self):
        from neuroimaging.qc_guidance import all_references
        for r in all_references():
            assert "example.com" not in r.url
            assert "TODO" not in r.label + r.detail + r.url

    def test_references_are_deduplicated_by_url(self):
        from neuroimaging.qc_guidance import all_references
        urls = [r.url for r in all_references()]
        assert len(urls) == len(set(urls))

    def test_dois_are_well_formed(self):
        from neuroimaging.qc_guidance import all_references
        for r in all_references():
            if "doi.org" in r.url:
                assert re.search(r"doi\.org/10\.\d{4,}/", r.url), r.url


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

class TestLookup:

    def test_get_guidance_hit_and_miss(self):
        from neuroimaging.qc_guidance import get_guidance
        assert get_guidance("fd_mean") is not None
        assert get_guidance("not_a_metric") is None

    def test_guidance_for_modality_filters(self):
        from neuroimaging.qc_guidance import guidance_for_modality
        bold_keys = {g.key for g in guidance_for_modality("bold")}
        anat_keys = {g.key for g in guidance_for_modality("T1w")}
        assert "tsnr" in bold_keys
        assert "tsnr" not in anat_keys
        assert "cjv" in anat_keys
        assert "cjv" not in bold_keys

    def test_guidance_for_keys_preserves_order_and_skips_unknown(self):
        from neuroimaging.qc_guidance import guidance_for_keys
        result = guidance_for_keys(["tsnr", "nope", "fd_mean"])
        assert [g.key for g in result] == ["tsnr", "fd_mean"]

    def test_tooltip_mentions_direction_and_why(self):
        from neuroimaging.qc_guidance import get_guidance
        tip = get_guidance("fd_mean").tooltip()
        assert "lower is better" in tip
        assert "Why:" in tip and "Look for:" in tip


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:

    def test_measure_card_has_anchor_and_sections(self):
        from neuroimaging.qc_guidance import render_measure_card, get_guidance
        html = render_measure_card(get_guidance("fd_mean"))
        assert 'id="guidance-fd_mean"' in html
        assert "Why it is here" in html
        assert "What to look for" in html
        assert "Flagged automatically" in html

    def test_measure_card_escapes_html(self):
        from neuroimaging.qc_guidance import render_measure_card, MeasureGuidance, Reference
        g = MeasureGuidance(
            key="x", label="X", modalities=("bold",), direction="context",
            why="<script>alert(1)</script>", look_for="a & b", auto_flag="c",
            references=(Reference(label="L", detail="D", url="https://x.org"),),
        )
        html = render_measure_card(g)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "a &amp; b" in html

    def test_guidance_section_renders_requested_keys_only(self):
        from neuroimaging.qc_guidance import render_guidance_section
        html = render_guidance_section(["tsnr"])
        assert 'id="guidance-tsnr"' in html
        assert 'id="guidance-cjv"' not in html

    def test_process_section_renders_all_notes(self):
        from neuroimaging.qc_guidance import render_process_section, PROCESS_GUIDANCE
        html = render_process_section()
        for note in PROCESS_GUIDANCE:
            assert f'id="guidance-{note.key}"' in html
            assert note.title in html

    def test_markdown_export_covers_registry(self):
        from neuroimaging.qc_guidance import guidance_markdown, MEASURE_GUIDANCE
        md = guidance_markdown()
        assert md.startswith("# QC review guidance")
        for key in MEASURE_GUIDANCE:
            assert f"`{key}`" in md

    def test_markdown_export_can_be_scoped_to_one_modality(self):
        from neuroimaging.qc_guidance import guidance_markdown
        md = guidance_markdown(modalities=["bold"])
        assert "`tsnr`" in md
        assert "`cjv`" not in md


# ---------------------------------------------------------------------------
# Dashboard integration
# ---------------------------------------------------------------------------

class TestDashboardIntegration:

    def test_dashboard_embeds_guidance_sections(self, mriqc_dir, fmriprep_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        out = tmp_path / "dash.html"
        generate_dashboard(
            mriqc_dir=mriqc_dir, fmriprep_dir=fmriprep_dir,
            modality="bold", save_path=out,
        )
        html = out.read_text()
        assert 'id="how-to-review"' in html
        assert 'id="measure-guidance"' in html
        assert 'id="guidance-fd_mean"' in html
        assert 'id="guidance-tsnr"' in html

    def test_documented_columns_get_tooltips(self, mriqc_dir, fmriprep_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        out = tmp_path / "dash.html"
        generate_dashboard(
            mriqc_dir=mriqc_dir, fmriprep_dir=fmriprep_dir,
            modality="bold", save_path=out,
        )
        html = out.read_text()
        assert 'class="th-documented"' in html
        assert 'href="#guidance-tsnr"' in html

    def test_anat_dashboard_documents_anat_measures(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        out = tmp_path / "dash_t1w.html"
        generate_dashboard(mriqc_dir=mriqc_dir, modality="T1w", save_path=out)
        html = out.read_text()
        assert 'id="guidance-cjv"' in html
        assert 'id="guidance-tsnr"' not in html

"""Tests for QC decision provenance and config-backed thresholds.

Covers the rule that a decision only counts when an identifiable human
recorded it, and that QC thresholds come from ``config/base.toml``
rather than being hardcoded.
"""

import json

import pytest


@pytest.fixture
def decisions_dir(tmp_path):
    d = tmp_path / "qc_decisions"
    d.mkdir()
    return d


def _latest(decisions_dir, run_key, subject="01"):
    path = decisions_dir / f"sub-{subject}" / f"{run_key}_decision.json"
    return json.loads(path.read_text())["decisions"][-1]


# ---------------------------------------------------------------------------
# Who may record what
# ---------------------------------------------------------------------------

class TestSignOffRules:

    def test_human_decision_records_attribution(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision, is_signed_off
        rec = save_decision(
            decisions_dir, "01", "01", "encoding", "01",
            "keep", "Looks clean", "bhutch",
        )
        assert rec["automated"] is False
        assert rec["reviewer"] == "bhutch"
        assert rec["timestamp"]
        assert is_signed_off(rec)

    def test_human_signoff_requires_named_reviewer(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        for bad in ("", "   ", "auto-stub", "AUTOMATED"):
            with pytest.raises(ValueError, match="identifiable reviewer"):
                save_decision(
                    decisions_dir, "01", "01", "encoding", "01",
                    "keep", "reason", bad,
                )

    def test_automation_may_not_record_a_judgement(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        for verdict in ("keep", "exclude", "investigate"):
            with pytest.raises(ValueError, match="may only record 'pending'"):
                save_decision(
                    decisions_dir, "01", "01", "encoding", "01",
                    verdict, "reason", "auto-stub", automated=True,
                )

    def test_automation_records_pending_with_recommendation(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision, is_signed_off
        rec = save_decision(
            decisions_dir, "01", "01", "encoding", "01",
            "pending", "mean_fd=0.6mm", "auto-stub",
            automated=True, recommendation="investigate",
        )
        assert rec["decision"] == "pending"
        assert rec["recommendation"] == "investigate"
        assert rec["automated"] is True
        assert not is_signed_off(rec)

    def test_invalid_recommendation_rejected(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        with pytest.raises(ValueError, match="Invalid recommendation"):
            save_decision(
                decisions_dir, "01", "01", "encoding", "01",
                "pending", "r", "auto-stub", automated=True,
                recommendation="probably-fine",
            )

    def test_history_preserved_across_signoff(self, decisions_dir):
        """An auto stub followed by a human call keeps both entries."""
        from neuroimaging.qc_dashboard import save_decision, is_signed_off
        save_decision(
            decisions_dir, "01", "01", "encoding", "01",
            "pending", "auto", "auto-stub", automated=True,
            recommendation="keep",
        )
        save_decision(
            decisions_dir, "01", "01", "encoding", "01",
            "exclude", "Ringing on mosaic", "bhutch",
        )
        path = decisions_dir / "sub-01" / "sub-01_ses-01_task-encoding_run-01_bold_decision.json"
        history = json.loads(path.read_text())["decisions"]
        assert len(history) == 2
        assert not is_signed_off(history[0])
        assert is_signed_off(history[1])


class TestIsSignedOff:

    def test_none_and_empty(self):
        from neuroimaging.qc_dashboard import is_signed_off
        assert not is_signed_off(None)
        assert not is_signed_off({})

    def test_legacy_auto_stub_without_flag_is_not_signed_off(self):
        """Records written before the automated flag existed."""
        from neuroimaging.qc_dashboard import is_signed_off
        legacy = {"decision": "keep", "reviewer": "auto-stub", "reason": "..."}
        assert not is_signed_off(legacy)

    def test_legacy_human_record_without_flag_is_signed_off(self):
        from neuroimaging.qc_dashboard import is_signed_off
        legacy = {"decision": "keep", "reviewer": "bhutch", "reason": "..."}
        assert is_signed_off(legacy)

    def test_pending_never_counts(self):
        from neuroimaging.qc_dashboard import is_signed_off
        assert not is_signed_off({"decision": "pending", "reviewer": "bhutch"})

    def test_automated_flag_overrides_human_name(self):
        from neuroimaging.qc_dashboard import is_signed_off
        assert not is_signed_off(
            {"decision": "keep", "reviewer": "bhutch", "automated": True}
        )


# ---------------------------------------------------------------------------
# Dashboard reporting
# ---------------------------------------------------------------------------

class TestDashboardCounts:

    def _dashboard(self, mriqc_dir, fmriprep_dir, decisions_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        out = tmp_path / "d.html"
        generate_dashboard(
            mriqc_dir=mriqc_dir, fmriprep_dir=fmriprep_dir,
            decisions_dir=decisions_dir, modality="bold", save_path=out,
        )
        return out.read_text()

    def test_auto_stub_shows_as_pending(
        self, mriqc_dir, fmriprep_dir, decisions_dir, tmp_path
    ):
        from neuroimaging.qc_dashboard import save_decision
        save_decision(
            decisions_dir, "01", "01", "encoding", "01",
            "pending", "auto", "auto-stub", automated=True,
            recommendation="keep",
        )
        html = self._dashboard(mriqc_dir, fmriprep_dir, decisions_dir, tmp_path)
        assert "PENDING" in html
        assert "auto: keep" in html

    def test_legacy_auto_keep_still_shows_as_pending(
        self, mriqc_dir, fmriprep_dir, decisions_dir, tmp_path
    ):
        """A pre-existing auto-stub 'keep' must not read as reviewed."""
        sub = decisions_dir / "sub-01"
        sub.mkdir(parents=True)
        run_key = "sub-01_ses-01_task-encoding_run-01_bold"
        (sub / f"{run_key}_decision.json").write_text(json.dumps({
            "run_key": run_key,
            "decisions": [
                {"decision": "keep", "reason": "auto-stub from confounds",
                 "reviewer": "auto-stub", "timestamp": "2026-01-01T00:00:00Z"}
            ],
        }))
        html = self._dashboard(mriqc_dir, fmriprep_dir, decisions_dir, tmp_path)
        assert "<span>Signed Off</span>" in html
        # The run must not be badged KEEP on the strength of an auto record.
        assert 'class="badge badge-keep"' not in html
        assert 'class="badge badge-pending"' in html

    def test_human_signoff_counts_as_reviewed(
        self, mriqc_dir, fmriprep_dir, decisions_dir, tmp_path
    ):
        from neuroimaging.qc_dashboard import save_decision
        save_decision(
            decisions_dir, "01", "01", "encoding", "01",
            "keep", "Checked the carpet plot", "bhutch",
        )
        html = self._dashboard(mriqc_dir, fmriprep_dir, decisions_dir, tmp_path)
        assert "badge-keep" in html
        assert "bhutch" in html

    def test_summary_table_separates_auto_from_signed_off(self):
        from neuroimaging.qc_dashboard import _build_subject_summary
        decisions = {
            "sub-01_ses-01_task-a_run-01_bold": {
                "latest": {"decision": "keep", "reviewer": "bhutch"}},
            "sub-01_ses-01_task-a_run-02_bold": {
                "latest": {"decision": "pending", "reviewer": "auto-stub",
                           "automated": True}},
            "sub-01_ses-01_task-a_run-03_bold": {
                "latest": {"decision": "keep", "reviewer": "auto-stub"}},
        }
        outliers = {"summary_by_subject": {"01": {
            "n_runs": 3, "n_outlier_runs": 0, "pct_outlier": 0}}}
        rows = _build_subject_summary(outliers, None, decisions, "bold")
        row = rows[0]
        assert row["n_reviewed"] == 1
        assert row["n_auto"] == 2
        assert row["n_pending"] == 2


# ---------------------------------------------------------------------------
# Pipeline gate
# ---------------------------------------------------------------------------

class TestPipelineGate:

    def test_pending_excluded_by_default(self):
        from pipeline.qc_decisions import is_signed_off
        assert not is_signed_off({"decision": "pending", "reviewer": "auto-stub"})

    def test_gate_and_dashboard_agree_on_signoff(self):
        """Two implementations, one rule — they must not drift."""
        from neuroimaging.qc_dashboard import is_signed_off as dash
        from pipeline.qc_decisions import is_signed_off as gate
        cases = [
            None, {},
            {"decision": "keep", "reviewer": "bhutch"},
            {"decision": "keep", "reviewer": "auto-stub"},
            {"decision": "pending", "reviewer": "bhutch"},
            {"decision": "keep", "reviewer": "bhutch", "automated": True},
            {"decision": "exclude", "reviewer": ""},
        ]
        for case in cases:
            assert dash(case) == gate(case), case

    def test_invalid_treat_pending_as_rejected(self, tmp_path):
        from pipeline.qc_decisions import get_included_runs
        with pytest.raises(ValueError, match="treat_pending_as"):
            get_included_runs(
                "01", "01", bids_root=tmp_path, treat_pending_as="maybe",
            )

    def test_summarize_reports_signoff_split(self, tmp_path):
        from pipeline.qc_decisions import summarize
        from neuroimaging.constants import DERIVATIVES_DIRS
        root = tmp_path / DERIVATIVES_DIRS["preprocessing_qc"] / "sub-01"
        root.mkdir(parents=True)
        entries = [
            ("a", {"decision": "keep", "reviewer": "bhutch"}),
            ("b", {"decision": "pending", "reviewer": "auto-stub",
                   "automated": True}),
            ("c", {"decision": "keep", "reviewer": "auto-stub"}),
        ]
        for name, rec in entries:
            (root / f"{name}_decision.json").write_text(
                json.dumps({"run_key": name, "decisions": [rec]})
            )
        counts = summarize(bids_root=tmp_path)
        assert counts["total"] == 3
        assert counts["signed_off"] == 1
        assert counts["awaiting_signoff"] == 2
        assert counts["pending"] == 1


# ---------------------------------------------------------------------------
# Config-backed thresholds
# ---------------------------------------------------------------------------

class TestQcSettings:

    def test_settings_come_from_repo_config(self):
        from neuroimaging.qc import qc_settings
        s = qc_settings()
        assert set(s) == {"fd_threshold", "investigate_threshold", "iqr_multiplier"}
        assert all(isinstance(v, float) for v in s.values())

    def test_repo_config_declares_qc_section(self):
        """The thresholds must actually live in base.toml, not just default."""
        import tomllib
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        cfg = tomllib.loads((root / "config" / "base.toml").read_text())
        assert "qc" in cfg
        assert cfg["qc"]["fd_threshold"] == 0.5
        assert cfg["qc"]["iqr_multiplier"] == 1.5
        assert "investigate_threshold" in cfg["qc"]

    def test_config_values_actually_reach_the_code(self, tmp_path, monkeypatch):
        """Regression: core/__init__ imports pybids, so a naive
        ``from core.config import ...`` silently strands these on defaults
        wherever pybids is absent."""
        from pathlib import Path
        import importlib

        repo = Path(__file__).resolve().parent.parent
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        text = (repo / "config" / "base.toml").read_text()
        text = text.replace("fd_threshold = 0.5", "fd_threshold = 0.2")
        text = text.replace("iqr_multiplier = 1.5", "iqr_multiplier = 3.0")
        (cfg / "base.toml").write_text(text)

        monkeypatch.setenv("MMMDATA_CONFIG_DIR", str(cfg))
        from neuroimaging import qc
        settings = qc.qc_settings()
        assert settings["fd_threshold"] == 0.2
        assert settings["iqr_multiplier"] == 3.0

    def test_unreadable_config_warns_rather_than_silently_defaulting(
        self, tmp_path, monkeypatch
    ):
        from neuroimaging import qc
        from neuroimaging.constants import DEFAULT_QC_SETTINGS
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "base.toml").write_text("this is not = valid toml [[[")
        monkeypatch.setenv("MMMDATA_CONFIG_DIR", str(bad))
        with pytest.warns(RuntimeWarning, match="Could not read"):
            assert qc.qc_settings() == DEFAULT_QC_SETTINGS

    def test_explicit_argument_overrides_config(self):
        from neuroimaging.qc import _setting
        assert _setting("fd_threshold", 0.9) == 0.9
        assert _setting("iqr_multiplier", 3.0) == 3.0

    def test_none_falls_back_to_config(self):
        from neuroimaging.qc import _setting, qc_settings
        assert _setting("fd_threshold", None) == qc_settings()["fd_threshold"]

    def test_detect_outliers_uses_configured_multiplier(self, mriqc_dir):
        from neuroimaging.qc import detect_outliers, qc_settings
        result = detect_outliers(mriqc_dir, "bold")
        assert result["iqr_multiplier"] == qc_settings()["iqr_multiplier"]

    def test_summarize_motion_uses_configured_threshold(self, fmriprep_dir):
        from neuroimaging.qc import summarize_motion, qc_settings
        result = summarize_motion(fmriprep_dir)
        assert result["fd_threshold_mm"] == qc_settings()["fd_threshold"]

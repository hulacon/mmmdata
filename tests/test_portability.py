"""Off-cluster portability guards.

The dataset and the shared envs live on Talapas GPFS, but the repo is
checked out and worked on where neither exists -- a laptop, a CI runner,
the claude.ai web workspace. These guards pin the properties that make
that bearable, and they all run *without* the cluster, which is the
point.

What they cover:

1. The suite imports from a bare checkout. Tests import the packages
   under ``src/python`` by bare name; nothing put that directory on
   ``sys.path``, so a fresh clone collected zero tests. The repo-root
   ``conftest.py`` bootstraps it.
2. Configured paths use the resolved ``/gpfs`` prefix, which works in
   every harness, rather than the bare ``/projects`` form which does not.
3. An absent SLURM is reported, not tracebacked.
"""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestBareCheckoutImports:
    """A clone with no environment setup must be able to run the suite."""

    def test_src_python_is_on_sys_path(self):
        assert str(REPO_ROOT / "src" / "python") in sys.path

    def test_bare_name_packages_import(self):
        """This is how the suite imports them, and it is what broke: the
        shared Talapas env happened to provide the path, so nothing in
        the repo had to."""
        import behavioral.io  # noqa: F401
        import core.config  # noqa: F401

    def test_conftest_bootstrap_is_idempotent(self):
        """Repeated collection must not stack duplicate entries."""
        src = str(REPO_ROOT / "src" / "python")
        assert sys.path.count(src) == 1


class TestConfiguredPathsResolveEverywhere:
    def test_base_toml_uses_resolved_gpfs_prefix(self):
        """`/projects/hulacon` is not a synonym for `/gpfs/projects/
        hulacon`: it resolves on login and compute nodes but not inside
        every sandbox, which is what a committed local.toml was papering
        over."""
        text = (REPO_ROOT / "config" / "base.toml").read_text()
        offenders = [
            line.strip()
            for line in text.splitlines()
            if '"/projects/' in line and not line.lstrip().startswith("#")
        ]
        assert not offenders, (
            "use the /gpfs-prefixed form in base.toml: " + "; ".join(offenders)
        )

    def test_local_toml_does_not_shadow_base_paths(self):
        """local.toml is tracked here, so a [paths] block in it applies to
        every operator. Base should be right at source instead."""
        local = REPO_ROOT / "config" / "local.toml"
        if not local.exists():
            pytest.skip("no local.toml in this checkout")
        cfg = tomllib.loads(local.read_text())
        assert not cfg.get("paths"), (
            "local.toml overrides [paths] for everyone; fix config/base.toml "
            f"instead (found: {sorted(cfg['paths'])})"
        )

    def test_every_configured_path_is_absolute(self):
        cfg = tomllib.loads((REPO_ROOT / "config" / "base.toml").read_text())
        for key, value in cfg.get("paths", {}).items():
            assert Path(value).is_absolute(), f"{key} is not absolute: {value}"


class TestSlurmAbsence:
    def test_submit_reports_absent_slurm(self, monkeypatch, capsys):
        """Without the guard, subprocess raises a bare FileNotFoundError
        whose message is just "sbatch" -- which reads like the .sbatch
        file is missing rather than the scheduler."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import submit_dcm2bids

        monkeypatch.setattr(submit_dcm2bids.shutil, "which", lambda _: None)

        def fail(*args, **kwargs):
            raise AssertionError("submitted with no SLURM on PATH")

        monkeypatch.setattr(submit_dcm2bids.subprocess, "run", fail)

        rc = submit_dcm2bids.submit("sub-03", "ses-06", repo_root=REPO_ROOT)
        assert rc == 1
        assert "sbatch" in capsys.readouterr().err

    def test_dry_run_still_previews_without_slurm(self, monkeypatch, capsys):
        """Assembling the command is useful off-cluster even though
        submitting it is not, so the guard sits after the dry-run branch."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import submit_dcm2bids

        monkeypatch.setattr(submit_dcm2bids.shutil, "which", lambda _: None)

        rc = submit_dcm2bids.submit(
            "sub-03", "ses-06", dry_run=True, repo_root=REPO_ROOT
        )
        assert rc == 0
        assert "sbatch" in capsys.readouterr().out


class TestDatasetTierIsGated:
    def test_marker_is_registered(self, pytestconfig):
        """--strict-markers is on, so an unregistered marker would error;
        this states the contract rather than relying on that."""
        markers = pytestconfig.getini("markers")
        assert any(m.startswith("requires_dataset") for m in markers)

    def test_dataset_available_is_answered_from_config(self):
        """Not from a hardcoded path: a local.toml pointing at a copy of
        the tree must count as available."""
        sys.path.insert(0, str(REPO_ROOT))
        from conftest import dataset_available

        assert isinstance(dataset_available(), bool)

    def test_dataset_available_is_false_without_the_tree(self, monkeypatch):
        sys.path.insert(0, str(REPO_ROOT))
        import conftest as root_conftest
        from core import config as core_config

        monkeypatch.setattr(
            core_config, "load_config",
            lambda *a, **k: {"paths": {"bids_project_dir": "/nonexistent/tree"}},
        )
        assert root_conftest.dataset_available() is False

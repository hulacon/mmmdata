"""The campaign runs viz2psy `caption_vlm` on movie frames only, scores its text
as a word2psy source, uses the reference batch size, and refuses its cells off
the pinned GPU type.

Offline: no extractor runs, and the registries are stubbed so the test does
not depend on which extractors are installed. The driver is imported as a
module, as in test_stimfeat_verify.py.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stimfeat_campaign.py"
REGISTRY = {
    "viz2psy": ["clip", "caption", "caption_vlm", "motion"],
    "aud2psy": ["loudness"],
    "word2psy": ["minilm", "emotion"],
}


@pytest.fixture
def campaign(monkeypatch):
    spec = importlib.util.spec_from_file_location("stimfeat_campaign_caption_vlm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "registry", lambda package: list(REGISTRY[package]))
    return mod


def _sources(campaign):
    return {s.key: s for s in campaign.build_sources()}


def test_caption_vlm_runs_on_movie_frames_only(campaign):
    with_vlm = sorted(k for k, s in _sources(campaign).items() if "caption_vlm" in s.models)
    assert with_vlm == ["movies/frames"]


def test_vlmcap_scores_the_caption_vlm_tables(campaign):
    s = _sources(campaign)["movies/vlmcap"]
    assert (s.package, s.prefix, s.depends) == ("word2psy", "vlmcap_", "movies/frames/caption_vlm")
    assert s.models == REGISTRY["word2psy"]


def test_caption_vlm_batch_size_is_the_reference_runs(campaign, tmp_path):
    src = campaign.Source("movies", "image", "viz2psy", "", True, "synthetic", lambda: [])
    unit = campaign.Unit(id="u", out_dir=tmp_path, inputs=[str(tmp_path / "a.png")])

    def batch(model):
        cmd = campaign.command_for(src, unit, model)
        return cmd[cmd.index("--batch-size") + 1]
    assert (batch("caption_vlm"), batch("clip")) == ("32", "64")


def test_skip_model_filter(campaign):
    src = _sources(campaign)["movies/frames"]
    args = argparse.Namespace(model=None, skip_model=["caption_vlm"])
    assert "caption_vlm" not in campaign._selected_models(src, args)
    assert campaign._selected_models(src, argparse.Namespace(model="clip", skip_model=None)) == ["clip"]


def test_caption_vlm_gpu_guard(campaign):
    problem = campaign.caption_vlm_gpu_problem
    h100, a100 = "NVIDIA H100 80GB HBM3", "NVIDIA A100 80GB PCIe"
    assert problem({"clip"}, []) is None
    assert problem({"caption_vlm"}, [h100], "H100") is None
    assert problem({"caption_vlm"}, [a100], "") is None
    msg = problem({"caption_vlm", "clip"}, [a100], "H100")
    assert "--skip-model caption_vlm" in msg and "A100" in msg
    assert "no GPU" in problem({"caption_vlm"}, [], "H100")
    assert problem({"caption_vlm"}, [h100, a100], "H100")

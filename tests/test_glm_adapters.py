"""The TBencoding repetition adapter labels trials from the whole subject's events."""

import numpy as np
import pandas as pd
import pytest

from neuroimaging.glm.adapters import AdapterError, TbRepetitionAdapter, adapt_events, get_adapter
from neuroimaging.glm.models import ModelSpecError, load_model, parse_model


def _tb_events(n_sessions=3, runs_per_session=2):
    """Synthetic TBencoding: anchors in every session; once items; repeated items 3x in-session."""
    runs = []
    item = 100
    for ses in range(1, n_sessions + 1):
        # per-session item pool: 2 anchors (ids 1,2), 4 once items, 2 repeated items x3
        schedule = []
        for rep_item in (item, item + 1):
            schedule += [(rep_item, 3)] * 3
        schedule += [(item + 2 + i, 1) for i in range(4)]
        schedule += [(1, 2), (2, 2)]
        item += 10
        rng = np.random.default_rng(ses)
        rng.shuffle(schedule)
        per_run = len(schedule) // runs_per_session
        for run in range(1, runs_per_session + 1):
            rows = []
            for j, (mmm, encon) in enumerate(schedule[(run - 1) * per_run: run * per_run]):
                onset = 9.0 + 4.5 * j
                for modality, dur in (("visual", 3.0), ("auditory", 0.5)):
                    rows.append({"onset": onset, "duration": dur, "modality": modality, "ses_num": ses,
                                 "run_idx": run, "mmmId": mmm, "enCon": encon, "trial_type": modality[:5]})
            runs.append(pd.DataFrame(rows))
    return runs


def test_labels_anchor_once_first_later_from_all_sessions():
    runs = _tb_events()
    out = adapt_events("tbrepetition", runs)
    allt = pd.concat(out)
    assert set(allt["trial_type"]) == {"first", "later", "once", "anchor"}
    assert (allt["modality"] == "visual").all()  # one row per trial
    assert set(allt.loc[allt["trial_type"] == "anchor", "mmmId"]) == {1, 2}
    assert (allt.loc[allt["enCon"] == 1, "trial_type"] == "once").all()
    rep = allt[allt["trial_type"].isin(["first", "later"])]
    assert (rep.groupby(["ses_num", "mmmId"])["trial_type"].apply(lambda s: (s == "first").sum()) == 1).all()
    assert (rep["trial_type"] == "later").sum() == 2 * (rep["trial_type"] == "first").sum()
    # first is the earliest presentation in (run, onset) order
    for (_, _), g in rep.groupby(["ses_num", "mmmId"]):
        g = g.sort_values(["run_idx", "onset"])
        assert g["trial_type"].tolist() == ["first", "later", "later"]


def test_single_session_and_unprepared_apply_are_refused():
    runs = _tb_events(n_sessions=1)
    with pytest.raises(AdapterError, match="every session"):
        adapt_events("tbrepetition", runs)
    a = TbRepetitionAdapter()
    with pytest.raises(AdapterError, match="prepare"):
        a.apply(runs[0])
    with pytest.raises(AdapterError, match="lack columns"):
        a.prepare([pd.DataFrame({"onset": [1.0]})])
    with pytest.raises(KeyError, match="Unknown events adapter"):
        get_adapter("nope")
    assert adapt_events(None, runs) == runs


def test_tbrepetition_model_declares_the_adapter_and_designs_build():
    pytest.importorskip("nilearn")
    from neuroimaging.constants import MOTION_6
    from neuroimaging.glm.config import GlmConfig
    from neuroimaging.glm.design import build_design_matrix

    model = load_model("tbrepetition")
    assert model.adapter == "tbrepetition" and model.task == "TBencoding"
    assert model.conditions == ("first", "later", "once", "anchor")
    ev = adapt_events(model.adapter, _tb_events())[0]
    conf = pd.DataFrame(np.zeros((60, 6)), columns=MOTION_6)
    dm = build_design_matrix(ev, conf, 1.5, 60, model, GlmConfig(smoothing_fwhm=None))
    assert {"first", "later", "once", "anchor"} <= set(dm.columns)


def test_adapter_instruction_must_be_first_and_known():
    base = {
        "Name": "x", "BIDSModelVersion": "1.0.0", "Input": {"task": ["t"]},
        "Nodes": [{"Level": "Run", "Name": "run", "Transformations": {"Instructions": [
            {"Name": "Factor", "Input": ["trial_type"]},
            {"Name": "Adapter", "Input": ["tbrepetition"]},
            {"Name": "Convolve", "Model": "spm", "Input": ["trial_type.a"]}]},
            "Model": {"Type": "glm", "X": ["trial_type.a", 1]},
            "Contrasts": [{"Name": "a", "ConditionList": ["trial_type.a"], "Weights": [1]}]}],
    }
    with pytest.raises(ModelSpecError, match="first in the list"):
        parse_model(base)
    base["Nodes"][0]["Transformations"]["Instructions"] = [
        {"Name": "Adapter", "Input": ["nope"]},
        {"Name": "Factor", "Input": ["trial_type"]},
        {"Name": "Convolve", "Model": "spm", "Input": ["trial_type.a"]}]
    with pytest.raises(ModelSpecError, match="unknown events adapter"):
        parse_model(base)

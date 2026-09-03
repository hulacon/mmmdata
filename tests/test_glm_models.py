"""The BIDS Stats Models loader: shipped specs parse; wrong ones are refused."""

import copy
import json

import pytest

from neuroimaging.constants import TASK_STREAM_MAP
from neuroimaging.glm.models import (
    MODELS_DIR,
    ModelSpecError,
    list_models,
    load_model,
    parse_model,
)


def test_models_dir_exists_and_lists_the_shipped_specs():
    assert MODELS_DIR.is_dir(), MODELS_DIR
    names = list_models()
    assert {"floc", "motor", "auditory"} <= set(names)


@pytest.mark.parametrize("name", list_models())
def test_every_shipped_spec_parses(name):
    model = load_model(name)
    assert model.name == name
    assert model.task in TASK_STREAM_MAP, f"{name} names a task label the constants do not know"
    assert model.conditions
    assert model.contrasts
    assert model.fixed_effects
    for c in model.contrasts:
        assert set(c.conditions) <= set(model.conditions)
        assert c.name.replace("_", "").isalnum()


def test_floc_contrasts_follow_the_archived_plan():
    m = load_model("floc")
    face = m.contrast("faceVsObject")
    assert face.weights == {"adult": 0.5, "child": 0.5, "car": -0.5, "instrument": -0.5}
    assert "baseline" not in m.conditions  # implicit baseline, not a regressor


def test_motor_contrasts_are_against_explicit_rest():
    m = load_model("motor")
    for c in m.contrasts:
        assert c.weights["rest"] == -1.0
        assert sum(c.weights.values()) == 0.0


def test_load_by_path_and_unknown_name(tmp_path):
    path = MODELS_DIR / "model-motor_smdl.json"
    assert load_model(path).name == "motor"
    with pytest.raises(FileNotFoundError, match="available"):
        load_model("nosuchmodel")


def _spec():
    with open(MODELS_DIR / "model-motor_smdl.json") as f:
        return json.load(f)


def test_refuses_a_contrast_over_a_condition_not_in_x():
    spec = _spec()
    spec["Nodes"][0]["Contrasts"][0]["ConditionList"] = ["trial_type.hand", "trial_type.tongue"]
    with pytest.raises(ModelSpecError, match="not in Model.X"):
        parse_model(spec)


def test_refuses_weight_length_mismatch():
    spec = _spec()
    spec["Nodes"][0]["Contrasts"][0]["Weights"] = [1]
    with pytest.raises(ModelSpecError, match="same non-zero length"):
        parse_model(spec)


def test_refuses_an_x_without_intercept():
    spec = _spec()
    spec["Nodes"][0]["Model"]["X"] = [x for x in spec["Nodes"][0]["Model"]["X"] if x != 1]
    with pytest.raises(ModelSpecError, match="intercept"):
        parse_model(spec)


def test_refuses_unimplemented_transformation():
    spec = _spec()
    spec["Nodes"][0]["Transformations"]["Instructions"].append({"Name": "Scale", "Input": ["trial_type.hand"]})
    with pytest.raises(ModelSpecError, match="not implemented"):
        parse_model(spec)


def test_refuses_a_dataset_level_node():
    spec = _spec()
    spec["Nodes"].append({"Level": "Dataset", "Name": "group", "Model": {"Type": "glm", "X": [1]}})
    with pytest.raises(ModelSpecError, match="out of scope"):
        parse_model(spec)


def test_refuses_a_subject_node_that_is_not_fixed_effects():
    spec = _spec()
    spec["Nodes"][1]["Model"] = {"Type": "glm", "X": [1]}
    with pytest.raises(ModelSpecError, match="fixed effects"):
        parse_model(spec)


def test_refuses_a_missing_edge():
    spec = _spec()
    spec["Edges"] = []
    with pytest.raises(ModelSpecError, match="Edge"):
        parse_model(spec)


def test_no_subject_node_means_no_pooling():
    spec = copy.deepcopy(_spec())
    spec["Nodes"] = spec["Nodes"][:1]
    spec["Edges"] = []
    assert parse_model(spec).fixed_effects is False


def test_refuses_two_tasks_at_fit_time():
    spec = _spec()
    spec["Input"]["task"] = ["motor", "floc"]
    with pytest.raises(ModelSpecError, match="one task"):
        _ = parse_model(spec).task

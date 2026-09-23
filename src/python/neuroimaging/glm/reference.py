"""The frozen reference GLM specification — the stand-in every contrast consumer cites.

Until braintwill's estimator exists, every condition-level contrast or effect
map in MMMData is fitted to one resolved specification rather than to whatever
``GlmConfig()`` happens to default to (mmmdata-agents
``docs/workbench/glm-strategy/``: ratification R7/R8, WP5; stand-in DECIDED
2026-09-23). The specification is data, in ``reference_spec.json`` beside this
module, so it can be cited, hashed and read without importing any code.
:func:`reference_config` is how code turns it into a :class:`~.config.GlmConfig`.

Frozen means: ``sha256`` covers every key except ``confound_regimes``, and
:func:`load_reference_spec` refuses a file whose content no longer matches it.
Regimes may GROW — a new named regime is an extension, not a change — but a
regime already marked ``frozen`` may not change (the bake-off harness's rule,
glm-strategy log 2026-09-12). Changing anything else is a new version in a new
file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import GlmConfig

SPEC_PATH = Path(__file__).with_name("reference_spec.json")

#: Keys outside the frozen digest: the digest itself, and the regimes, which
#: may grow (a regime's own ``status`` says whether it may still change).
UNHASHED_KEYS: tuple[str, ...] = ("sha256", "confound_regimes")

#: GlmConfig fields a regime supplies; every other field comes from ``config``.
REGIME_FIELDS: tuple[str, ...] = ("confounds", "acompcor_n")

#: GlmConfig fields that are a consumer's choice, not a modelling decision.
CONSUMER_FIELDS: tuple[str, ...] = ("output_tree",)


def spec_digest(spec: dict[str, Any]) -> str:
    content = {k: v for k, v in spec.items() if k not in UNHASHED_KEYS}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


def load_reference_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    """The specification, after checking it was not edited since it was frozen."""
    spec = json.loads(Path(path).read_text())
    if spec_digest(spec) != spec.get("sha256"):
        raise RuntimeError(
            f"{path} no longer matches its sha256 {spec.get('sha256')!r}: the frozen reference "
            "was edited. Restore it from git; a changed reference is a new version in a new file."
        )
    return spec


def reference_config(regime: str = "reference", *, calibrated: bool = False, **overrides: Any) -> GlmConfig:
    """The reference :class:`GlmConfig`, under one named confound regime.

    ``calibrated=True`` swaps in the spec's calibrated-inference engine (AR(1)
    prewhitening) for consumers that need calibrated z/t; the default is the
    effect engine (OLS). ``overrides`` are for consumer fields such as
    ``output_tree``; anything else changes the model, and the fit's metadata
    will say so because it records the whole config.
    """
    spec = load_reference_spec()
    regimes = spec["confound_regimes"]
    if regime not in regimes:
        raise KeyError(f"Unknown confound regime {regime!r}; the reference spec has {sorted(regimes)}")
    fields = dict(spec["config"])
    fields["confounds"] = tuple(regimes[regime]["confounds"])
    fields["acompcor_n"] = int(regimes[regime]["acompcor_n"])
    if calibrated:
        fields["noise_model"] = spec["estimation"]["calibrated_noise_model"]
    return dataclasses.replace(GlmConfig(**fields), **overrides)

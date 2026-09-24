"""The one registry of named confound regimes, and the design each one builds.

Before this module, four places each defined their own confound sets under
overlapping names (``glm/config.py`` presets, ``fit_prf.py`` presets, an
inline ``clean()`` in the pRF gate, Nastase's numbered models in
``isc_confounds/``), and two "acompcor" definitions disagreed on drift
(mmmdata-agents ``docs/workbench/data-quality/log.md``, OBSERVED 2026-09-23).
Ben's rule (DECIDED 2026-09-23): one standardized set of definitions that
every consumer imports; no script defines its own columns.

The registry is the ``confound_regimes`` block of the frozen GLM reference
specification (``glm/reference_spec.json``). That block sits outside the
spec's digest so it may **grow**; a regime already marked ``frozen`` may not
change. Each entry::

    "name": {
      "status":      "frozen" | "defined" | "provisional",
      "confounds":   [fMRIPrep confound column names],
      "acompcor_n":  how many a_comp_cor_* columns (combined mask, variance order),
      "drift":       "cosine" | "polynomial" | "none"   (absent = "cosine"),
      "drift_order": polynomial order, "polynomial" only,
      "description": one sentence,
      "source":      where the definition came from (optional)
    }

``provisional`` regimes are recorded but not yet confirmed by whoever
defined them; a driver must ask for them explicitly. ``regime_design`` turns
an entry plus a run's confounds table into the regressor columns for that
run; non-steady-state volumes are reported separately so a consumer can
either drop those rows (a measure) or add one-hot spikes (a GLM) — the two
are the same model under OLS.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .constants import COSINE_PREFIX
from .glm.config import ACOMPCOR_PREFIX, NON_STEADY_STATE_PREFIX
from .glm.reference import SPEC_PATH, load_reference_spec

REGIME_STATUSES: tuple[str, ...] = ("frozen", "defined", "provisional")
DRIFT_MODELS: tuple[str, ...] = ("cosine", "polynomial", "none")


@dataclasses.dataclass(frozen=True)
class Regime:
    """One named confound regime, as the registry defines it."""

    name: str
    status: str
    confounds: tuple[str, ...]
    acompcor_n: int
    drift: str
    drift_order: int
    description: str = ""
    source: str = ""

    @property
    def version(self) -> str:
        """A digest of the definition, so an output can say which one it used."""
        content = {
            "name": self.name,
            "confounds": list(self.confounds),
            "acompcor_n": self.acompcor_n,
            "drift": self.drift,
            "drift_order": self.drift_order,
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def provisional(self) -> bool:
        return self.status == "provisional"


def _regime_from_entry(name: str, entry: dict[str, Any]) -> Regime:
    status = entry.get("status")
    if status not in REGIME_STATUSES:
        raise ValueError(f"Regime {name!r} has status {status!r}; expected one of {REGIME_STATUSES}")
    drift = entry.get("drift", "cosine")
    if drift not in DRIFT_MODELS:
        raise ValueError(f"Regime {name!r} has drift {drift!r}; expected one of {DRIFT_MODELS}")
    order = int(entry.get("drift_order", 0))
    if drift == "polynomial" and order < 1:
        raise ValueError(f"Regime {name!r} declares polynomial drift without a drift_order >= 1")
    return Regime(
        name=name,
        status=status,
        confounds=tuple(entry.get("confounds", ())),
        acompcor_n=int(entry.get("acompcor_n", 0)),
        drift=drift,
        drift_order=order if drift == "polynomial" else 0,
        description=entry.get("description", ""),
        source=entry.get("source", ""),
    )


def load_regimes(path: Path = SPEC_PATH) -> dict[str, Regime]:
    """Every regime in the registry, by name, after the spec's digest check."""
    spec = load_reference_spec(path)
    return {name: _regime_from_entry(name, entry) for name, entry in spec["confound_regimes"].items()}


def get_regime(name: str, path: Path = SPEC_PATH) -> Regime:
    regimes = load_regimes(path)
    if name not in regimes:
        raise KeyError(f"Unknown confound regime {name!r}; the registry has {sorted(regimes)}")
    return regimes[name]


def confirmed_regimes(path: Path = SPEC_PATH) -> list[str]:
    """Names of the regimes a driver may use without asking: frozen and defined."""
    return [n for n, r in load_regimes(path).items() if not r.provisional]


def polynomial_drift(n_vol: int, order: int) -> np.ndarray:
    """Legendre polynomials of degree 1..order on the run's time axis, no constant.

    Orthogonal on [-1, 1], so the columns are nearly orthogonal to each other
    and to the intercept; the intercept itself is the consumer's to add.
    """
    if order < 1:
        return np.empty((n_vol, 0))
    t = np.linspace(-1.0, 1.0, n_vol)
    cols = [np.polynomial.legendre.Legendre.basis(k)(t) for k in range(1, order + 1)]
    return np.column_stack(cols)


@dataclasses.dataclass(frozen=True)
class RegimeDesign:
    """The regressors one regime asks for on one run, without an intercept.

    ``columns`` has one row per volume; ``nss`` marks the volumes fMRIPrep
    flagged as not at steady state (``non_steady_state_outlier*``), which are
    NOT in ``columns``. A measure drops those rows; a GLM adds one one-hot
    column per flagged volume. ``dof_resid`` is what is left after both:
    ``n_vol - n_nss - n_regressors - 1``.
    """

    regime: Regime
    columns: pd.DataFrame
    nss: np.ndarray
    n_drift: int

    @property
    def n_vol(self) -> int:
        return len(self.columns)

    @property
    def n_nss(self) -> int:
        return int(self.nss.sum())

    @property
    def n_regressors(self) -> int:
        return self.columns.shape[1]

    @property
    def dof_resid(self) -> int:
        return self.n_vol - self.n_nss - self.n_regressors - 1

    @property
    def dof_loss(self) -> float:
        return 1.0 - self.dof_resid / self.n_vol


def non_steady_state_mask(confounds: pd.DataFrame) -> np.ndarray:
    """Boolean per volume: True where any ``non_steady_state_outlier*`` column is 1."""
    cols = [c for c in confounds.columns if c.startswith(NON_STEADY_STATE_PREFIX)]
    if not cols:
        return np.zeros(len(confounds), dtype=bool)
    return confounds[cols].fillna(0).to_numpy().sum(axis=1) > 0


def regime_design(regime: Regime, confounds: pd.DataFrame) -> RegimeDesign:
    """Build the regressor columns for ``regime`` from a run's confounds table.

    Raises KeyError naming every column the TSV lacks: a design short a
    regressor is a different model, not a degraded one. A ``cosine`` drift
    on a run for which fMRIPrep wrote no cosine columns is NOT an error —
    fMRIPrep writes none for runs shorter than half its high-pass period —
    and ``n_drift`` records the 0 (data-quality DECIDED 2026-09-24: no
    special case, no polynomial fallback).
    """
    available = list(confounds.columns)
    missing = [c for c in regime.confounds if c not in available]
    if missing:
        raise KeyError(
            f"Regime {regime.name!r} needs confound columns the TSV lacks: {missing}. "
            "Check the fMRIPrep version and that the confounds TSV is complete."
        )
    wanted = list(regime.confounds)
    if regime.acompcor_n:
        acc = [f"{ACOMPCOR_PREFIX}{i:02d}" for i in range(regime.acompcor_n)]
        absent = [c for c in acc if c not in available]
        if absent:
            raise KeyError(
                f"Regime {regime.name!r} needs aCompCor columns the TSV lacks: {absent}. "
                "fMRIPrep writes them only when its aCompCor step ran; check the confounds sidecar."
            )
        wanted.extend(acc)
    # fMRIPrep writes n/a for the first sample of derivative and FD columns;
    # zero is the right value for a regressor's undefined first sample.
    design = confounds[wanted].astype(float).fillna(0.0).reset_index(drop=True)

    n_vol = len(confounds)
    if regime.drift == "cosine":
        cos = [c for c in available if c.startswith(COSINE_PREFIX)]
        drift = confounds[cos].astype(float).fillna(0.0).reset_index(drop=True)
    elif regime.drift == "polynomial":
        arr = polynomial_drift(n_vol, regime.drift_order)
        drift = pd.DataFrame(arr, columns=[f"poly_{k}" for k in range(1, regime.drift_order + 1)])
    else:
        drift = pd.DataFrame(index=range(n_vol))
    columns = pd.concat([design, drift], axis=1)
    return RegimeDesign(
        regime=regime,
        columns=columns,
        nss=non_steady_state_mask(confounds),
        n_drift=drift.shape[1],
    )


def describe_regimes(path: Path = SPEC_PATH) -> pd.DataFrame:
    """One row per regime: the table a CLI prints or a log cites."""
    rows = []
    for r in load_regimes(path).values():
        rows.append(
            {
                "regime": r.name,
                "status": r.status,
                "n_confounds": len(r.confounds),
                "acompcor_n": r.acompcor_n,
                "drift": r.drift if r.drift != "polynomial" else f"polynomial:{r.drift_order}",
                "version": r.version,
                "description": r.description,
            }
        )
    return pd.DataFrame(rows)

"""confounds(): a run's nuisance regressors, through the regime registry."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from neuroimaging import confounds as registry
from neuroimaging.io import FmriprepRun, load_confounds

from .provenance import record
from .errors import OpenDecisionError

STANDARD = "standard"


def confounds(
    run: FmriprepRun,
    regime: Optional[str] = None,
    *,
    regimes: str | Sequence[str] | None = None,
    columns: Optional[Sequence[str]] = None,
) -> registry.RegimeDesign | dict[str, registry.RegimeDesign] | pd.DataFrame:
    """A run's confound regressors, by regime name.

    Give exactly one of:

    ``regime="<name>"``
        One regime from the registry (``neuroimaging.confounds``; the
        ``confound_regimes`` block of ``glm/reference_spec.json``). Returns
        its :class:`~neuroimaging.confounds.RegimeDesign`: the regressor
        columns, the non-steady-state volumes, and the drift count.
    ``regimes=[...]``
        Several regimes at once: ``{name: RegimeDesign}``.
        ``regimes="standard"`` will mean the lab's comparison set; it raises
        :class:`~mmmget.errors.OpenDecisionError` until that set is chosen.
    ``columns=[...]``
        The escape hatch: these raw fMRIPrep columns, as a DataFrame, with
        no drift or non-steady-state handling. Works, and is recorded as
        non-standard so any result built on it says so.

    A provisional regime is returned when asked for by name but recorded
    as non-standard, since its definition is not yet confirmed.
    """
    given = [x is not None for x in (regime, regimes, columns)]
    if sum(given) != 1:
        raise TypeError("confounds() takes exactly one of regime=, regimes= or columns=")

    if columns is not None:
        cols = list(columns)
        df = load_confounds(run, columns=cols)
        record("confounds", standard=False,
               reason="raw columns instead of a registry regime", columns=cols)
        return df

    if regimes is not None:
        if isinstance(regimes, str):
            if regimes != STANDARD:
                raise ValueError(
                    f"regimes={regimes!r}: pass a list of regime names, or "
                    f"{STANDARD!r} for the comparison set")
            raise OpenDecisionError(
                "D3",
                "the lab's standard comparison set of confound regimes has not "
                "been chosen, so confounds(run, regimes='standard') has nothing "
                "to return. Name the regimes you compare instead, e.g. "
                f"confounds(run, regimes=[...]); the registry has "
                f"{sorted(registry.load_regimes())}.")
        return {name: confounds(run, regime=name) for name in regimes}

    reg = registry.get_regime(regime)
    design = registry.regime_design(reg, load_confounds(run))
    record("confounds", standard=not reg.provisional,
           reason="provisional regime" if reg.provisional else None,
           regime=reg.name, regime_version=reg.version, regime_status=reg.status)
    return design

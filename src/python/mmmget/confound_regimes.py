"""confounds(): a run's nuisance regressors, through the regime registry."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from neuroimaging import confounds as registry
from neuroimaging.io import FmriprepRun, load_confounds

from .provenance import record

STANDARD = "standard"


class ComparisonDesigns(dict):
    """``{regime: RegimeDesign}`` for one comparison set, on one run.

    A member the run cannot carry (too few aCompCor components) is left out
    of the mapping and listed in :attr:`absent` with the reason: declared
    absent, never filled with a smaller design.
    """

    def __init__(self, comparison_set: registry.ComparisonSet,
                 designs: dict[str, registry.RegimeDesign], absent: dict[str, str]):
        super().__init__(designs)
        self.comparison_set = comparison_set.name
        self.version = comparison_set.version
        self.members = comparison_set.regimes
        self.absent = absent


def confounds(
    run: FmriprepRun,
    regime: Optional[str] = None,
    *,
    regimes: str | Sequence[str] | None = None,
    columns: Optional[Sequence[str]] = None,
) -> registry.RegimeDesign | dict[str, registry.RegimeDesign] | ComparisonDesigns | pd.DataFrame:
    """A run's confound regressors, by regime name.

    Give exactly one of:

    ``regime="<name>"``
        One regime from the registry (``neuroimaging.confounds``; the
        ``confound_regimes`` block of ``glm/reference_spec.json``). Returns
        its :class:`~neuroimaging.confounds.RegimeDesign`: the regressor
        columns, the non-steady-state volumes, and the drift count.
    ``regimes="standard"`` (or another comparison set's name)
        The lab's comparison set (decision D3): every regime a result is
        compared across, as a :class:`ComparisonDesigns` mapping. Members
        the run cannot carry are in its ``.absent``, not in the mapping.
        The set's name and version are recorded.
    ``regimes=[...]``
        Several regimes by name: ``{name: RegimeDesign}``. A regime the run
        cannot carry raises :class:`~neuroimaging.confounds.RegimeNotApplicable`.
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
            cset = registry.get_comparison_set(regimes)
            designs, absent = {}, {}
            for name in cset.regimes:
                try:
                    designs[name] = confounds(run, regime=name)
                except registry.RegimeNotApplicable as exc:
                    absent[name] = str(exc)
            record("confounds", standard=True, comparison_set=cset.name,
                   comparison_set_version=cset.version,
                   absent_regimes=sorted(absent) or None)
            return ComparisonDesigns(cset, designs, absent)
        return {name: confounds(run, regime=name) for name in regimes}

    reg = registry.get_regime(regime)
    design = registry.regime_design(reg, load_confounds(run))
    record("confounds", standard=not reg.provisional,
           reason="provisional regime" if reg.provisional else None,
           regime=reg.name, regime_version=reg.version, regime_status=reg.status)
    return design

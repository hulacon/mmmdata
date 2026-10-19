"""runs(): preprocessed BOLD runs from the catalog, QC exclusions applied."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Optional

from core import catalog
from neuroimaging.constants import (
    DEFAULT_SPACE,
    DEFAULT_VARIANT,
    DERIVATIVES_DIRS,
    FMRIPREP_VARIANTS,
    NATIVE_SPACE,
    MixedLocalizerDesignError,
    check_single_design,
)
from neuroimaging.io import FmriprepRun
from neuroimaging.qc_dashboard import (
    _build_run_key,
    decision_search_dirs,
    is_signed_off,
    load_decisions,
)

from . import _config
from .provenance import record


@dataclasses.dataclass(frozen=True)
class Run(FmriprepRun):
    """One preprocessed BOLD run, as the catalog and the QC record see it.

    An :class:`~neuroimaging.io.FmriprepRun`, so everything that takes one
    (``load_confounds``, ``load_bold``, ``load_mask``) takes this too. Paths
    are the catalog's, as of its last sweep; a field is ``None`` when the
    catalog has no such file for this run.

    Attributes (beyond FmriprepRun's)
    ----------
    qc_decision : dict | None
        The run's current run-level QC entry (the last one without a
        ``domain``), or None when the run has no record.
    excluded : bool
        True when that entry is a person's ``exclude``. Automated and
        ``pending`` entries never exclude a run.
    """

    qc_decision: Optional[dict[str, Any]] = None
    excluded: bool = False

    @property
    def run_key(self) -> str:
        """The QC decision key: ``sub-##_ses-##_task-<label>[_run-##]_bold``."""
        return _build_run_key(self.subject, self.session, self.task, self.run, "bold")


def split_space(space: str) -> tuple[Optional[str], Optional[str]]:
    """(catalog ``space``, catalog ``res``) for a space label.

    ``"MNI152NLin2009cAsym_res-2"`` -> ``("MNI152NLin2009cAsym", "2")``;
    ``"T1w"`` -> ``("T1w", None)``; the native label ``"func"`` -> ``(None,
    None)``, since fMRIPrep writes native-grid files with no space entity.
    """
    if space == NATIVE_SPACE:
        return None, None
    name, _, res = space.partition("_res-")
    return name, (res or None)


def _run_int(run: int | str | None) -> Optional[int]:
    if run is None:
        return None
    return int(str(run).removeprefix("run-"))


_RUN_LABEL = re.compile(r"_run-([^_]+)_")

#: How the raw dataset's root may be spelled in ``dataset_relpath``.
RAW_RELPATHS = (None, "", ".")


def _abs(bids_root: Path, row: dict) -> Path:
    rel = row["dataset_relpath"]
    base = bids_root if rel in RAW_RELPATHS else bids_root / rel
    return base / row["path"]


def _pick(rows: list[dict], what: str, key: tuple, **want: Any) -> Optional[dict]:
    hits = [r for r in rows if all(r.get(k) == v for k, v in want.items())]
    if len(hits) > 1:
        names = ", ".join(Path(r["path"]).name for r in hits)
        raise ValueError(
            f"the catalog has {len(hits)} {what} files for {key}: {names}. An "
            "entity runs() does not key on (acq, dir, echo, ...) distinguishes "
            "them; select the file by that entity in a catalog query instead")
    return hits[0] if hits else None


def _check_filters(db: Path, deriv: str, filters: dict[str, Any]) -> None:
    """An empty result from a label the dataset does not have is a typo,
    not an answer: name the label and the ones that exist."""
    for col, value in filters.items():
        if value is None:
            continue
        _, rows = catalog.run_select(
            db, f'SELECT DISTINCT "{col}" AS v FROM files '
                "WHERE dataset_relpath = ? AND datatype = 'func' "
                f'AND "{col}" IS NOT NULL ORDER BY 1', [deriv])
        known = [r["v"] for r in rows]
        if value not in known:
            raise LookupError(
                f"{col}={value!r} does not occur in {deriv} (catalog {db}); "
                f"known values: {known}")


def _latest_run_level(history: list[dict]) -> Optional[dict]:
    # duckbrain's per-domain notes share the file and must never read as the
    # run's verdict (duckbrain catalog/qc_decisions.py does the same).
    run_level = [e for e in history if not e.get("domain")]
    return run_level[-1] if run_level else None


def runs(
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: int | str | None = None,
    space: str = DEFAULT_SPACE,
    include_excluded: bool = False,
    variant: str = DEFAULT_VARIANT,
    bids_root: Path | str | None = None,
    allow_mixed_designs: bool = False,
) -> list[Run]:
    """Preprocessed BOLD runs, from the catalog, with QC exclusions applied.

    A run is one fMRIPrep confounds TSV in the catalog (the same anchor as
    ``find_fmriprep_runs``); its BOLD, mask, boldref and surfaces in
    *space*, and its raw events file, are attached when catalogued. Nothing
    is globbed: a run the catalog does not list is not returned, so rebuild
    the catalog after new preprocessing.

    Runs a person has marked ``exclude`` in the QC record are dropped unless
    ``include_excluded=True``, in which case they are returned with
    ``excluded=True`` and the call is recorded as non-standard.

    Parameters
    ----------
    subject, session : str, optional
        Labels, bare (``"##"``) or prefixed (``"sub-##"``, ``"ses-##"``).
    task : str, optional
        BIDS task label.
    run : int or str, optional
        Run index (``1``, ``"01"`` or ``"run-01"``).
    space : str
        Output space, as fMRIPrep names it with any resolution
        (``"MNI152NLin2009cAsym_res-2"``, ``"T1w"``) or ``"func"`` for the
        run's native grid.
    include_excluded : bool
        Return excluded runs too (flagged), instead of dropping them.
    variant : str
        Which fMRIPrep derivative tree.
    bids_root : path, optional
        Overrides ``paths.bids_project_dir`` from config.
    allow_mixed_designs : bool
        Some localizer task labels cover two protocols acquired in different
        session groups (``LOCALIZER_DESIGNS`` in ``neuroimaging.constants``).
        When ``task`` names one and the returned runs span both, this raises
        ``MixedLocalizerDesignError`` unless True, as ``find_fmriprep_runs``
        does. Pooling that way is recorded as non-standard; the usual fix is
        a ``session`` filter.

    Returns
    -------
    list of Run
        Sorted by subject, session, task, run.

    Raises
    ------
    FileNotFoundError
        The catalog does not exist (the message names how to build it).
    LookupError
        Nothing matched and a filter names a label the dataset lacks.
    MixedLocalizerDesignError
        See ``allow_mixed_designs``. Only when ``task`` is given; an
        unfiltered sweep is not a design selection.
    """
    if variant not in FMRIPREP_VARIANTS:
        raise ValueError(f"variant {variant!r} is not one of {FMRIPREP_VARIANTS}")
    root = _config.bids_root(bids_root)
    db = catalog.catalog_path(root)
    deriv = DERIVATIVES_DIRS[variant]
    sub = catalog.bare_label(subject, "sub-")
    ses = catalog.bare_label(session, "ses-")
    run_i = _run_int(run)
    filters = {"sub": sub, "ses": ses, "task": task, "run": run_i}

    # the raw dataset's root has more than one spelling (RAW_RELPATHS)
    where = ["(dataset_relpath IN (?, '.', '') OR dataset_relpath IS NULL)",
             "datatype = 'func'", "task IS NOT NULL"]
    params: list[Any] = [deriv]
    for col, value in filters.items():
        if value is not None:
            where.append(f'"{col}" = ?')
            params.append(value)
    _, rows = catalog.run_select(
        db,
        'SELECT dataset_relpath, sub, ses, task, run, space, res, hemi, "desc", '
        "suffix, ext, path FROM files WHERE " + " AND ".join(where),
        params,
    )

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["sub"], r["ses"], r["task"], r["run"]), []).append(r)

    cat_space, cat_res = split_space(space)
    decisions = load_decisions(decision_search_dirs(root), subject=sub)

    out: list[Run] = []
    for key, group in groups.items():
        d = [r for r in group if r["dataset_relpath"] == deriv]
        raw = [r for r in group if r["dataset_relpath"] in RAW_RELPATHS]
        conf = _pick(d, "confounds TSV", key, desc="confounds",
                     suffix="timeseries", ext=".tsv")
        if conf is None:
            continue
        sp = dict(space=cat_space, res=cat_res, hemi=None)
        bold = _pick(d, "preproc BOLD", key, desc="preproc", suffix="bold",
                     ext=".nii.gz", **sp)
        mask = _pick(d, "brain mask", key, desc="brain", suffix="mask",
                     ext=".nii.gz", **sp)
        if cat_space is None:  # native: the coregistered reference
            boldref = _pick(d, "boldref", key, desc="coreg", suffix="boldref",
                            ext=".nii.gz", **sp)
        else:
            boldref = _pick(d, "boldref", key, desc=None, suffix="boldref",
                            ext=".nii.gz", **sp)
        conf_json = _pick(d, "confounds JSON", key, desc="confounds",
                          suffix="timeseries", ext=".json")
        surf = {h: _pick(d, f"hemi-{h} surface", key, hemi=h, space="fsaverage6",
                         suffix="bold", ext=".func.gii") for h in ("L", "R")}
        events = _pick(raw, "events", key, suffix="events", ext=".tsv")

        m = _RUN_LABEL.search(Path(conf["path"]).name)
        run_label = m.group(1) if m else None
        rk = _build_run_key(key[0], key[1], key[2], run_label, "bold")
        latest = _latest_run_level(decisions.get(rk, {}).get("history", []))
        excluded = is_signed_off(latest) and latest.get("decision") == "exclude"

        def p(row):
            return _abs(root, row) if row else None

        out.append(Run(
            subject=key[0], session=key[1], task=key[2], run=run_label,
            variant=variant, space=space,
            bold=p(bold), mask=p(mask), boldref=p(boldref),
            confounds=p(conf), confounds_json=p(conf_json), events=p(events),
            surface_L=p(surf["L"]), surface_R=p(surf["R"]),
            qc_decision=latest, excluded=excluded,
        ))

    if not out:
        _check_filters(db, deriv, filters)

    out.sort(key=lambda r: (r.subject, r.session, r.task, r.run or ""))
    n_excluded = sum(r.excluded for r in out)
    kept = out if include_excluded else [r for r in out if not r.excluded]

    reasons = []
    if include_excluded:
        reasons.append("QC exclusions not applied (include_excluded=True)")
    if task is not None:
        try:
            check_single_design(task, (r.session for r in kept))
        except MixedLocalizerDesignError as exc:
            if not allow_mixed_designs:
                raise MixedLocalizerDesignError(
                    f"{exc} Add a session= filter, or pass "
                    "allow_mixed_designs=True to pool deliberately.") from None
            reasons.append(f"pooled the designs of split task-{task} (allow_mixed_designs=True)")
    record(
        "runs",
        standard=not reasons,
        reason="; ".join(reasons) or None,
        variant=variant, space=space,
        query={k: v for k, v in filters.items() if v is not None} or None,
        n_runs=len(kept), n_excluded=n_excluded,
    )
    return kept

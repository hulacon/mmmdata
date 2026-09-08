"""The one GLM configuration every condition-level runner imports.

Before this module, ``glmsingle_tbencoding.py``, ``glmsingle_natencoding.py``
and friends each declared their own ``SPACE`` and ``TR`` constants that
matched ``neuroimaging.constants`` by coincidence (glm-strategy log,
OBSERVED 2026-08-21). Space and confound groups come from ``constants`` here;
the repetition time is read from the run's own sidecar, because it is a fact
about the acquisition, not a constant of the code.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

from ..constants import ACOMPCOR_6, COSINE_PREFIX, DEFAULT_SPACE, DEFAULT_VARIANT, MOTION_6, MOTION_24
from ..io import FmriprepRun


@dataclasses.dataclass(frozen=True)
class GlmConfig:
    """Settings shared by every condition-level GLM fit.

    Attributes
    ----------
    space, variant
        Which fMRIPrep tree and output space to read, from ``constants``.
    hrf_model
        nilearn HRF name; ``"spm"`` matches the ``Convolve`` model the shipped
        BIDS Stats Models declare. Derivatives are off by default: block
        designs with 4–20 s blocks gain little, and a derivative column per
        condition doubles the design width for the fLoc's ten conditions.
        ``"glmsingle:<k>"`` (k in 0..19) is kernel k of GLMsingle's canonical
        HRF library (:mod:`.hrf`); the bake-off's per-voxel HRF arm fits one
        design per kernel over the voxels whose ``HRFindex`` chose it
        (:mod:`.voxelwise_hrf`).
    noise_model
        ``"ar1"`` is nilearn's AR(1) prewhitening — the fixed-effects
        production candidate. ``"ols"`` is the iid control arm. ``"arma11"``
        is AFNI 3dREMLfit's ARMA(1,1) and only the ``remlfit`` estimator
        accepts it.
    smoothing_fwhm
        Applied by the estimator. ``None`` disables it; the archived plan used
        5 mm for ROI definition.
    high_pass
        Only used when ``drift_model`` is set. Default is to rely on fMRIPrep's
        cosine regressors instead, which is why ``drift_model`` is ``None``.
    confounds
        Confound columns taken from fMRIPrep, plus every ``cosine*`` column
        when ``include_cosine`` is set. Six motion parameters is the
        deliberate default for a localizer: aCompCor and the 24-parameter
        expansion eat degrees of freedom that a 300 s run does not have to
        spare. The bake-off's confound factor is the three
        :data:`CONFOUND_PRESETS`; ``with_confounds`` applies one.
    acompcor_n
        How many aCompCor components to append (``a_comp_cor_00`` upward,
        fMRIPrep's variance order over the combined WM+CSF mask). 0 is none.
    output_tree
        Derivative directory name under ``<bids_root>/derivatives``.
    """

    space: str = DEFAULT_SPACE
    variant: str = DEFAULT_VARIANT
    hrf_model: str = "spm"
    noise_model: str = "ar1"
    smoothing_fwhm: Optional[float] = 5.0
    drift_model: Optional[str] = None
    high_pass: float = 0.01
    confounds: tuple[str, ...] = tuple(MOTION_6)
    include_cosine: bool = True
    acompcor_n: int = 0
    output_tree: str = "glm_localizer"

    def confound_columns(self, available: list[str]) -> list[str]:
        """The confound columns to regress, given what a confounds TSV has.

        Raises KeyError naming the missing ones rather than silently
        dropping them — a design matrix short a motion regressor is a
        different model, not a degraded one.
        """
        missing = [c for c in self.confounds if c not in available]
        if missing:
            raise KeyError(
                f"Confound columns not in the confounds TSV: {missing}. "
                f"Available: {available[:12]}..."
            )
        cols = list(self.confounds)
        if self.acompcor_n:
            wanted = [f"{ACOMPCOR_PREFIX}{i:02d}" for i in range(self.acompcor_n)]
            absent = [c for c in wanted if c not in available]
            if absent:
                raise KeyError(
                    f"aCompCor columns not in the confounds TSV: {absent}. fMRIPrep writes "
                    "them only when its aCompCor step ran; check the confounds sidecar."
                )
            cols.extend(wanted)
        if self.include_cosine:
            cols.extend(c for c in available if c.startswith(COSINE_PREFIX))
        return cols

    def with_confounds(self, preset: str) -> "GlmConfig":
        """A copy using one of :data:`CONFOUND_PRESETS`."""
        try:
            columns, n_acompcor = CONFOUND_PRESETS[preset]
        except KeyError:
            raise KeyError(f"Unknown confound preset {preset!r}; available: {sorted(CONFOUND_PRESETS)}") from None
        return dataclasses.replace(self, confounds=columns, acompcor_n=n_acompcor)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_CONFIG = GlmConfig()

ACOMPCOR_PREFIX = "a_comp_cor_"

#: The bake-off's confound factor (glm-strategy log, DECIDED 2026-09-08):
#: preset -> (motion columns, number of aCompCor components). Cosines ride
#: along in every preset via ``include_cosine``.
CONFOUND_PRESETS: dict[str, tuple[tuple[str, ...], int]] = {
    "motion6": (tuple(MOTION_6), 0),
    "motion24": (tuple(MOTION_24), 0),
    "acompcor": (tuple(MOTION_6), len(ACOMPCOR_6)),
}


def repetition_time(run: FmriprepRun, bids_root: Optional[Path] = None) -> float:
    """The run's TR in seconds, from a sidecar — never from a constant.

    Looks at the raw BOLD sidecar under ``bids_root`` first (the acquisition
    record), then at the sidecar beside the preprocessed BOLD (fMRIPrep
    copies ``RepetitionTime`` through). Raises FileNotFoundError naming both
    paths when neither has it, because a guessed TR mis-times every regressor
    and nothing downstream would notice.
    """
    candidates: list[Path] = []
    if bids_root is not None:
        candidates.append(
            Path(bids_root)
            / f"sub-{run.subject}"
            / f"ses-{run.session}"
            / "func"
            / f"{run.entity_prefix}_bold.json"
        )
    if run.bold is not None:
        name = run.bold.name
        for suffix in (".nii.gz", ".nii"):
            if name.endswith(suffix):
                candidates.append(run.bold.with_name(name[: -len(suffix)] + ".json"))
                break
    for path in candidates:
        if path.exists():
            with open(path) as f:
                meta = json.load(f)
            tr = meta.get("RepetitionTime")
            if tr is not None:
                return float(tr)
    raise FileNotFoundError(
        f"No RepetitionTime for {run.entity_prefix}; looked in "
        + ", ".join(str(p) for p in candidates)
        + ". Pass bids_root so the raw sidecar can be read, or check the tree."
    )

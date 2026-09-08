"""The frozen bake-off harness: halves, N sets, thresholds, and the scores.

glm-strategy Settles-when 2 (as re-scoped 2026-09-08): every estimator cell
is scored on the same split-half protocol, and the protocol is written to
the output tree once and hashed, so a later code change cannot re-score
earlier cells under different rules without the mismatch being loud. Any
late-arriving arm (braintwill's GLS core, another package) is one more row
against the same frozen table.

Protocol
--------
* **Halves.** Runs sorted by (session, run); odd positions form half 1,
  even positions half 2. fLoc (6 runs) splits 3/3, motor (2 runs) is run-01
  vs run-02, TBencoding (42 runs over 14 sessions) alternates so both halves
  span every session. Within a half, run-level estimates pool by the same
  precision-weighted fixed effects the production runner uses; the half's
  t map is what gets scored.
* **Dice at an N set** — top-N voxels of each half's t map, per contrast,
  at several N (a face patch, a motor strip and diffuse repetition
  suppression have no common N), plus one threshold-based mask (z > 3.1,
  one-sided in the contrast's direction). A ranking that flips across N or
  between Dice and the correlation is reported as unstable, not resolved.
* **Split-half spatial correlation** of the two t maps inside the analysis
  mask — the threshold-free companion.
* **Mask.** Intersection of the subject's run brain masks for the task.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .reliability import dice, top_n_mask

HARNESS_VERSION = "1"
Z_THRESHOLD = 3.1  # one-sided p < .001

#: Pre-registered N sets per model (voxels at 2 mm; log entry 2026-09-08).
N_SETS: dict[str, tuple[int, ...]] = {
    "floc": (250, 500, 1000, 2000),
    "motor": (500, 1000, 2000, 4000),
    "tbrepetition": (1000, 2500, 5000, 10000),
}

#: The factorial (glm-strategy log, DECIDED 2026-09-08).
HRF_LEVELS: dict[str, str] = {
    "spm": "spm",
    "spmderiv": "spm + derivative",
    "voxelwise": "voxelwise",
}
CONFOUND_LEVELS: tuple[str, ...] = ("motion6", "motion24", "acompcor")
ENGINE_LEVELS: tuple[str, ...] = ("nilearn-ols", "nilearn-ar1", "remlfit-arma11")
STANDALONE_ARMS: dict[str, tuple[str, ...]] = {
    # arm -> models it applies to (GLMsingle needs >1 run per half; motor has 1)
    "glmsingle": ("floc",),
    "glmsingle-betas": ("tbrepetition",),
}
MODELS: tuple[str, ...] = ("floc", "motor", "tbrepetition")


@dataclasses.dataclass(frozen=True)
class Cell:
    """One point of the design: a model and an estimator configuration."""

    model: str
    hrf: str  # HRF_LEVELS key, or the standalone arm name
    confounds: str  # CONFOUND_LEVELS entry, or "na" for standalone arms
    engine: str  # ENGINE_LEVELS entry, or "na"

    @property
    def id(self) -> str:
        return f"model-{self.model}_hrf-{self.hrf}_conf-{self.confounds}_engine-{self.engine}"

    @property
    def standalone(self) -> bool:
        return self.hrf in STANDALONE_ARMS

    @classmethod
    def parse(cls, cell_id: str) -> "Cell":
        parts = dict(p.split("-", 1) for p in cell_id.split("_") if "-" in p)
        try:
            return cls(parts["model"], parts["hrf"], parts["conf"], parts["engine"])
        except KeyError as exc:
            raise ValueError(f"cell id {cell_id!r} lacks {exc}; expected model-…_hrf-…_conf-…_engine-…") from None


def factorial_cells(models: Sequence[str] = MODELS) -> list[Cell]:
    cells = [
        Cell(m, h, c, e) for m in models for h in HRF_LEVELS for c in CONFOUND_LEVELS for e in ENGINE_LEVELS
    ]
    for arm, arm_models in STANDALONE_ARMS.items():
        cells += [Cell(m, arm, "na", "na") for m in models if m in arm_models]
    return cells


def harness_spec() -> dict[str, Any]:
    """The protocol as data: what gets frozen into the output tree."""
    return {
        "version": HARNESS_VERSION,
        "halves": "runs sorted by (session, run); odd positions = half 1, even = half 2; "
                  "fixed effects within a half; t map scored",
        "n_sets": {k: list(v) for k, v in N_SETS.items()},
        "z_threshold": Z_THRESHOLD,
        "mask": "intersection of the subject's run brain masks for the task",
        "metrics": ["dice@N", f"dice@z{Z_THRESHOLD}", "r"],
        "hrf_levels": HRF_LEVELS,
        "confound_levels": list(CONFOUND_LEVELS),
        "engine_levels": list(ENGINE_LEVELS),
        "standalone_arms": {k: list(v) for k, v in STANDALONE_ARMS.items()},
        "models": list(MODELS),
    }


def spec_digest(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def freeze(out_base: Path) -> Path:
    """Write ``harness.json`` once; refuse to overwrite a different frozen spec."""
    path = Path(out_base) / "harness.json"
    spec = harness_spec()
    spec["sha256"] = spec_digest(spec)
    if path.exists():
        check_frozen(out_base)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2))
    return path


def check_frozen(out_base: Path) -> dict[str, Any]:
    """The frozen spec, or an error naming the drift between code and tree."""
    path = Path(out_base) / "harness.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `glm_bakeoff.py plan` first to freeze the harness")
    frozen = json.loads(path.read_text())
    content = {k: v for k, v in frozen.items() if k != "sha256"}
    if spec_digest(content) != frozen.get("sha256"):
        raise RuntimeError(f"{path} was edited after freezing (content hash != sha256); restore it from git or the log")
    live = harness_spec()
    if frozen.get("sha256") != spec_digest(live):
        drift = sorted(k for k in set(content) | set(live) if content.get(k) != live.get(k))
        raise RuntimeError(
            f"harness in code differs from the frozen {path} on {drift}. Cells already scored "
            "used the frozen rules; either revert the code or start a new output tree."
        )
    return frozen


def split_runs(n_runs: int) -> tuple[list[int], list[int]]:
    """Indices (0-based, in sorted run order) of half 1 and half 2."""
    if n_runs < 2:
        raise ValueError("a split-half needs at least two runs")
    return list(range(0, n_runs, 2)), list(range(1, n_runs, 2))


def score_halves(
    t_half1: np.ndarray,
    t_half2: np.ndarray,
    mask: np.ndarray,
    n_set: Sequence[int],
    z_half1: Optional[np.ndarray] = None,
    z_half2: Optional[np.ndarray] = None,
    z_threshold: float = Z_THRESHOLD,
) -> dict[str, Any]:
    """Every pre-registered metric for one contrast.

    ``r`` is Pearson over voxels in ``mask`` with finite values in both
    halves; ``dice@N`` uses top-N masks within ``mask``; ``dice@z`` uses
    ``z > z_threshold`` in each half (the z maps default to the t maps).
    """
    m = np.asarray(mask, dtype=bool)
    a = np.asarray(t_half1, dtype=float)
    b = np.asarray(t_half2, dtype=float)
    valid = m & np.isfinite(a) & np.isfinite(b)
    n_valid = int(valid.sum())
    out: dict[str, Any] = {"n_mask": int(m.sum()), "n_valid": n_valid}
    out["r"] = float(np.corrcoef(a[valid], b[valid])[0, 1]) if n_valid > 2 else float("nan")
    for n in n_set:
        out[f"dice@{n}"] = dice(top_n_mask(a, n, valid), top_n_mask(b, n, valid))
    za = a if z_half1 is None else np.asarray(z_half1, dtype=float)
    zb = b if z_half2 is None else np.asarray(z_half2, dtype=float)
    ma = valid & (za > z_threshold)
    mb = valid & (zb > z_threshold)
    out[f"dice@z{z_threshold}"] = dice(ma, mb)
    out[f"n@z{z_threshold}"] = [int(ma.sum()), int(mb.sum())]
    return out


def half_name(subject: str, task: str, space: str, half: int, contrast: str, stat: str) -> str:
    """``sub-XX_task-T_space-S_half-H_contrast-C_stat-X_statmap.nii.gz`` (bake-off tree only)."""
    return f"sub-{subject}_task-{task}_space-{space}_half-{half}_contrast-{contrast}_stat-{stat}_statmap.nii.gz"


def cell_dir(out_base: Path, subject: str, cell: Cell) -> Path:
    return Path(out_base) / f"sub-{subject}" / cell.id


def write_scores(path: Path, record: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def collect_scores(out_base: Path) -> list[dict[str, Any]]:
    """Long-format rows from every ``scores.json`` under the tree."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(out_base).glob("sub-*/model-*/scores.json")):
        rec = json.loads(path.read_text())
        cell = Cell.parse(rec["cell"])
        for contrast, metrics in rec["contrasts"].items():
            for metric, value in metrics.items():
                if metric.startswith("n"):
                    continue
                rows.append({
                    "subject": rec["subject"], "model": cell.model, "hrf": cell.hrf,
                    "confounds": cell.confounds, "engine": cell.engine, "cell": rec["cell"],
                    "contrast": contrast, "metric": metric, "value": value,
                })
    return rows

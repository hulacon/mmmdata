"""Where contrast maps go and what they are called.

Contract A keys in every filename (``subject, session, task, space``) plus
the two entities that make a statistical map self-describing, ``contrast``
and ``stat``, following the BIDS derivatives convention for ``statmap``
files. The tree gets a ``dataset_description.json`` on first write so the
nightly catalog rebuild indexes it instead of finding an undeclared
directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

STATS = ("effect", "variance", "t", "z")


def statmap_name(
    subject: str,
    task: str,
    space: str,
    contrast: str,
    stat: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    ext: str = ".nii.gz",
) -> str:
    """``sub-XX[_ses-YY]_task-T[_run-RR]_space-S_contrast-C_stat-X_statmap.nii.gz``.

    Bare labels in, prefixes added here — the same rule the QC tools use.
    A fixed-effects map over runs carries no ``run``; one pooled over sessions
    carries no ``session`` either.
    """
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}, got {stat!r}")
    parts = [f"sub-{_bare(subject, 'sub')}"]
    if session:
        parts.append(f"ses-{_bare(session, 'ses')}")
    parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{_bare(run, 'run')}")
    parts += [f"space-{space}", f"contrast-{contrast}", f"stat-{stat}", "statmap"]
    return "_".join(parts) + ext


def _bare(label: str, prefix: str) -> str:
    return label[len(prefix) + 1 :] if label.startswith(prefix + "-") else label


def output_dir(derivatives_dir: Path, tree: str, subject: str, session: Optional[str] = None) -> Path:
    d = Path(derivatives_dir) / tree / f"sub-{_bare(subject, 'sub')}"
    if session:
        d = d / f"ses-{_bare(session, 'ses')}"
    return d / "func"


def ensure_dataset_description(
    out_base: Path, fmriprep_dir: Path, model_name: str, estimator: str
) -> Path:
    """Write the tree-level description once, the way ``glmsingle_tb.py`` does."""
    dd = Path(out_base) / "dataset_description.json"
    if dd.exists():
        return dd
    try:
        from importlib.metadata import version

        nilearn_version = version("nilearn")
    except Exception:
        nilearn_version = "unknown"
    out_base.mkdir(parents=True, exist_ok=True)
    dd.write_text(
        json.dumps(
            {
                "Name": "Condition-level GLM contrast maps (localizers)",
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "nilearn",
                        "Version": nilearn_version,
                        "Description": f"estimator '{estimator}' via neuroimaging.glm",
                    },
                    {
                        "Name": "glm_contrast_maps.py",
                        "Description": "mmmdata/scripts/glm_contrast_maps.py; models in "
                        "mmmdata/models/ (BIDS Stats Models); design record in "
                        "mmmdata-agents docs/workbench/glm-strategy/",
                    },
                ],
                "SourceDatasets": [{"URL": str(fmriprep_dir)}],
                "HowToAcknowledge": f"First fit written by model {model_name}",
            },
            indent=2,
        )
    )
    return dd


def write_run_metadata(path: Path, payload: dict[str, Any]) -> Path:
    """The fit's own record: model, config, runs, estimator, versions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path

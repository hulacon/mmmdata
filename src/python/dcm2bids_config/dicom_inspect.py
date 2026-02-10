"""Inspect DICOM directories to extract fieldmap series information.

This is the only module that touches the filesystem.  It scans the DICOM
directory structure (``Series_##_<description>/``) to determine:
- Which fieldmap strategy applies (series_description vs series_number)
- The actual series numbers for each AP/PA pair
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FieldmapDetection:
    """Result of inspecting a DICOM session directory for fieldmaps.

    Attributes
    ----------
    strategy : str
        ``"series_description"`` if encoding/retrieval suffixes are present,
        ``"series_number"`` if only generic ``se_epi_ap``/``se_epi_pa`` names,
        ``"none"`` if no fieldmaps found.
    groups : dict
        Mapping from group name to ``{"ap": series_num, "pa": series_num}``.
        For ``series_description`` strategy, series numbers are still
        populated from the directory names.
    warnings : list[str]
        Any issues detected (extra pairs, missing pairs, etc.).
    """

    strategy: str = "none"
    groups: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# Matches directory names like "Series_10_se_epi_ap" or "Series_5_se_epi_pa_encoding"
_SERIES_RE = re.compile(
    r"Series_(\d+)_(se_epi_(ap|pa)(?:_(encoding|retrieval))?)",
    re.IGNORECASE,
)


def inspect_fieldmaps(dicom_dir: Path) -> FieldmapDetection:
    """Scan a DICOM session directory for spin-echo fieldmap series.

    Parameters
    ----------
    dicom_dir : Path
        Path to the session DICOM directory containing ``Series_*/`` dirs.

    Returns
    -------
    FieldmapDetection
        Detected strategy, groups, and any warnings.
    """
    result = FieldmapDetection()

    if not dicom_dir.is_dir():
        result.warnings.append(f"Directory not found: {dicom_dir}")
        return result

    # Parse all series directories
    entries: list[dict] = []
    for item in sorted(dicom_dir.iterdir()):
        if not item.is_dir():
            continue
        m = _SERIES_RE.match(item.name)
        if m:
            entries.append({
                "series_number": int(m.group(1)),
                "full_desc": m.group(2),
                "direction": m.group(3).lower(),  # "ap" or "pa"
                "suffix": (m.group(4) or "").lower(),  # "encoding", "retrieval", or ""
            })

    if not entries:
        return result

    # Determine strategy based on whether encoding/retrieval suffixes exist
    has_suffixes = any(e["suffix"] for e in entries)

    if has_suffixes:
        result.strategy = "series_description"
        # Group by suffix
        for suffix in ("encoding", "retrieval"):
            ap = [e for e in entries if e["direction"] == "ap" and e["suffix"] == suffix]
            pa = [e for e in entries if e["direction"] == "pa" and e["suffix"] == suffix]
            if len(ap) == 1 and len(pa) == 1:
                result.groups[suffix] = {
                    "ap": ap[0]["series_number"],
                    "pa": pa[0]["series_number"],
                }
            elif len(ap) > 1 or len(pa) > 1:
                result.warnings.append(
                    f"Multiple {suffix} fieldmap pairs found "
                    f"(AP: {[e['series_number'] for e in ap]}, "
                    f"PA: {[e['series_number'] for e in pa]}). "
                    f"Use overrides to specify which to use."
                )
            # Missing pairs for a suffix are not warned — may be single-group session
    else:
        result.strategy = "series_number"
        ap_all = sorted(
            [e for e in entries if e["direction"] == "ap"],
            key=lambda e: e["series_number"],
        )
        pa_all = sorted(
            [e for e in entries if e["direction"] == "pa"],
            key=lambda e: e["series_number"],
        )

        if len(ap_all) == 2 and len(pa_all) == 2:
            # Standard case: first pair → encoding, second → retrieval
            result.groups["encoding"] = {
                "ap": ap_all[0]["series_number"],
                "pa": pa_all[0]["series_number"],
            }
            result.groups["retrieval"] = {
                "ap": ap_all[1]["series_number"],
                "pa": pa_all[1]["series_number"],
            }
        elif len(ap_all) == 1 and len(pa_all) == 1:
            # Single pair (e.g., localizer or final session)
            result.groups["encoding"] = {
                "ap": ap_all[0]["series_number"],
                "pa": pa_all[0]["series_number"],
            }
        elif len(ap_all) > 2 or len(pa_all) > 2:
            # More than expected — likely re-entry, needs override
            result.warnings.append(
                f"Found {len(ap_all)} AP and {len(pa_all)} PA fieldmap series. "
                f"AP series: {[e['series_number'] for e in ap_all]}, "
                f"PA series: {[e['series_number'] for e in pa_all]}. "
                f"Use overrides to specify group assignments."
            )
            # Still assign first two pairs as default guess
            if len(ap_all) >= 2 and len(pa_all) >= 2:
                result.groups["encoding"] = {
                    "ap": ap_all[0]["series_number"],
                    "pa": pa_all[0]["series_number"],
                }
                result.groups["retrieval"] = {
                    "ap": ap_all[1]["series_number"],
                    "pa": pa_all[1]["series_number"],
                }
        else:
            result.warnings.append(
                f"Unexpected fieldmap count: {len(ap_all)} AP, {len(pa_all)} PA."
            )

    return result

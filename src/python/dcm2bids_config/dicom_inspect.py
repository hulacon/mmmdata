"""Inspect DICOM directories to extract fieldmap series information.

This is the only module that touches the filesystem.  It scans the DICOM
directory structure (``Series_##_<description>/``) to determine:
- Which fieldmap strategy applies (series_description vs series_number)
- The actual series numbers for each AP/PA pair

It also provides post-conversion validation:
- Detect truncated/aborted BOLD series (duplicate protocol names with few volumes)
- Flag duplicate protocol names that may cause run-numbering issues
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


# ---------------------------------------------------------------------------
# BOLD series inspection — detect aborted / duplicate acquisitions
# ---------------------------------------------------------------------------


@dataclass
class BoldSeriesInfo:
    """Metadata for a single BOLD DICOM series directory."""

    series_number: int
    protocol_name: str
    dicom_count: int
    dir_name: str


@dataclass
class BoldInspection:
    """Result of scanning BOLD series in a DICOM session directory.

    Attributes
    ----------
    series : list[BoldSeriesInfo]
        All non-SBRef, non-PhysioLog series found.
    duplicates : dict[str, list[BoldSeriesInfo]]
        Protocol names that appear more than once (potential aborts).
    truncated : list[BoldSeriesInfo]
        Series with fewer DICOMs than ``min_volumes``.
    warnings : list[str]
        Human-readable warnings about detected issues.
    """

    series: list[BoldSeriesInfo] = field(default_factory=list)
    duplicates: dict[str, list[BoldSeriesInfo]] = field(default_factory=dict)
    truncated: list[BoldSeriesInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def inspect_bold_series(
    dicom_dir: Path,
    *,
    min_volumes: int = 20,
    protocol_filter: str | None = None,
) -> BoldInspection:
    """Scan a DICOM session directory for BOLD series anomalies.

    Detects:
    - Duplicate protocol names (same scan run twice, one likely aborted)
    - Truncated series (fewer DICOMs than ``min_volumes``)

    Parameters
    ----------
    dicom_dir : Path
        Path to ``sourcedata/sub-XX/ses-YY/dicom/``.
    min_volumes : int
        Minimum DICOM count to consider a series complete. Series below
        this threshold are flagged as truncated.
    protocol_filter : str, optional
        If provided, only inspect series whose directory name contains this
        substring (e.g. ``"localizer_floc"``).

    Returns
    -------
    BoldInspection
        Detected series, duplicates, truncated entries, and warnings.
    """
    result = BoldInspection()

    if not dicom_dir.is_dir():
        result.warnings.append(f"Directory not found: {dicom_dir}")
        return result

    # Collect all non-SBRef, non-PhysioLog, non-scout, non-fmap series
    skip_patterns = {"_SBRef", "_PhysioLog", "AAhead_scout", "se_epi_"}
    for item in sorted(dicom_dir.iterdir()):
        if not item.is_dir():
            continue
        if any(pat in item.name for pat in skip_patterns):
            continue
        m = re.match(r"Series_(\d+)_(.+)", item.name)
        if not m:
            continue
        if protocol_filter and protocol_filter not in item.name:
            continue

        series_num = int(m.group(1))
        protocol = m.group(2)
        dcm_count = sum(1 for f in item.iterdir() if f.suffix == ".dcm")

        info = BoldSeriesInfo(
            series_number=series_num,
            protocol_name=protocol,
            dicom_count=dcm_count,
            dir_name=item.name,
        )
        result.series.append(info)

    # Detect duplicates (same protocol name, multiple series)
    by_protocol: dict[str, list[BoldSeriesInfo]] = {}
    for s in result.series:
        by_protocol.setdefault(s.protocol_name, []).append(s)
    for proto, entries in by_protocol.items():
        if len(entries) > 1:
            result.duplicates[proto] = entries
            counts = [
                f"Series {e.series_number} ({e.dicom_count} DICOMs)"
                for e in entries
            ]
            result.warnings.append(
                f"Duplicate protocol '{proto}': {', '.join(counts)}. "
                f"Likely aborted scan — use run_series override to pin "
                f"the correct series."
            )

    # Detect truncated series
    for s in result.series:
        if s.dicom_count < min_volumes:
            result.truncated.append(s)
            result.warnings.append(
                f"Truncated series: {s.dir_name} has only {s.dicom_count} "
                f"DICOMs (minimum: {min_volumes}). Likely an aborted "
                f"acquisition."
            )

    return result

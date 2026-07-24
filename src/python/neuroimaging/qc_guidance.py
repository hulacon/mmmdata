"""Explicit review guidance for the QC dashboard.

Every measure the dashboard surfaces is paired here with four things:

* **why** it is on the dashboard at all,
* **what a human should look for** in the MRIQC/fMRIPrep visual report,
* **what is flagged automatically**, and on what rule,
* **where the guidance comes from** (papers, MRIQC docs, forum threads).

The registry is the single source of truth. The dashboard renders it into
tooltips and a glossary; :func:`guidance_markdown` renders the same content
as a standalone document. Adding a measure in one place updates both.

Typical usage::

    from neuroimaging.qc_guidance import get_guidance, guidance_for_modality

    g = get_guidance("fd_mean")
    print(g.look_for)

    for g in guidance_for_modality("bold"):
        print(g.key, g.direction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

#: Direction of "better" for a measure. ``context`` means the value is
#: diagnostic but has no universally good direction.
VALID_DIRECTIONS = {"lower_better", "higher_better", "target_range", "context"}

#: What kind of source a reference is, so the UI can style it.
VALID_REF_KINDS = {"paper", "docs", "forum", "software"}


@dataclass(frozen=True)
class Reference:
    """A citable source for a piece of QC guidance.

    Parameters
    ----------
    label : str
        Short human-readable citation, e.g. ``'Power et al. (2012)'``.
    detail : str
        Venue/locator, e.g. ``'NeuroImage 59(3):2142-2154'``.
    url : str
        Resolvable link (DOI preferred).
    kind : str
        One of ``'paper'``, ``'docs'``, ``'forum'``, ``'software'``.
    note : str
        What this source is being cited *for* — the specific claim it backs.
    """

    label: str
    detail: str
    url: str
    kind: str = "paper"
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_REF_KINDS:
            raise ValueError(
                f"Invalid reference kind {self.kind!r}; "
                f"expected one of {sorted(VALID_REF_KINDS)}"
            )


@dataclass(frozen=True)
class MeasureGuidance:
    """Review guidance for one QC measure.

    Parameters
    ----------
    key : str
        Metric name as it appears in MRIQC output / dashboard columns.
    label : str
        Human-readable name.
    modalities : tuple of str
        Which dashboard modalities this measure appears in.
    direction : str
        One of :data:`VALID_DIRECTIONS`.
    why : str
        Why the measure earns a column — what failure mode it catches.
    look_for : str
        What the reviewer should check by eye to confirm or dismiss a flag.
    auto_flag : str
        What the dashboard flags without human input, and by what rule.
    units : str, optional
        Physical units, if any.
    literature_threshold : str, optional
        An a priori cutoff with published backing, stated with its source.
        ``None`` when no defensible universal cutoff exists.
    caveats : str, optional
        Where the measure misleads, or why a threshold does not transfer.
    references : tuple of Reference
        Sources backing the above.
    """

    key: str
    label: str
    modalities: tuple[str, ...]
    direction: str
    why: str
    look_for: str
    auto_flag: str
    units: str | None = None
    literature_threshold: str | None = None
    caveats: str | None = None
    references: tuple[Reference, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction {self.direction!r} for {self.key!r}; "
                f"expected one of {sorted(VALID_DIRECTIONS)}"
            )

    @property
    def direction_label(self) -> str:
        """Short phrase describing which way is better."""
        return {
            "lower_better": "lower is better",
            "higher_better": "higher is better",
            "target_range": "target range",
            "context": "context only — no good direction",
        }[self.direction]

    def tooltip(self) -> str:
        """One-paragraph plain-text summary for a column-header tooltip."""
        parts = [f"{self.label} ({self.direction_label})."]
        if self.units:
            parts.append(f"Units: {self.units}.")
        parts.append(f"Why: {self.why}")
        parts.append(f"Look for: {self.look_for}")
        if self.literature_threshold:
            parts.append(f"A priori: {self.literature_threshold}")
        return " ".join(parts)


@dataclass(frozen=True)
class ProcessNote:
    """Guidance about the review *process* rather than a single measure.

    Covers the things a reviewer needs to know that are not attached to any
    one column: how automatic flagging works, what the decision vocabulary
    means, and where the auto-populated decisions come from.

    Parameters
    ----------
    key : str
        Stable identifier, e.g. ``'outlier_rule'``.
    title : str
        Section heading.
    body : str
        The guidance itself. Plain text; rendered as a paragraph.
    references : tuple of Reference
        Sources backing it.
    """

    key: str
    title: str
    body: str
    references: tuple[Reference, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Shared references
# ---------------------------------------------------------------------------

_MRIQC_MEASURES = Reference(
    label="MRIQC documentation — Measures",
    detail="mriqc.readthedocs.io",
    url="https://mriqc.readthedocs.io/en/latest/measures.html",
    kind="docs",
    note="canonical definition of every IQM below",
)

_MRIQC_PAPER = Reference(
    label="Esteban et al. (2017)",
    detail="PLoS ONE 12(9):e0184661",
    url="https://doi.org/10.1371/journal.pone.0184661",
    note="MRIQC; IQM distributions do not transfer across sites unmodified",
)

_FMRIPREP_PAPER = Reference(
    label="Esteban et al. (2019)",
    detail="Nature Methods 16:111-116",
    url="https://doi.org/10.1038/s41592-018-0235-x",
    note="fMRIPrep; source of the confounds this dashboard summarises",
)

_POWER_2012 = Reference(
    label="Power et al. (2012)",
    detail="NeuroImage 59(3):2142-2154",
    url="https://doi.org/10.1016/j.neuroimage.2011.10.018",
    note="defines framewise displacement and motion-induced distance artifact",
)

_POWER_2014 = Reference(
    label="Power et al. (2014)",
    detail="NeuroImage 84:320-341",
    url="https://doi.org/10.1016/j.neuroimage.2013.08.048",
    note="FD/DVARS censoring; stricter thresholds than the 2012 paper",
)

_PARKES_2018 = Reference(
    label="Parkes et al. (2018)",
    detail="NeuroImage 171:415-436",
    url="https://doi.org/10.1016/j.neuroimage.2017.12.073",
    note="benchmark of denoising strategies and participant-exclusion criteria",
)

_QAP_FBER = Reference(
    label="Shehzad et al. (2015)",
    detail="Frontiers in Neuroscience, conference abstract (QAP)",
    url="https://doi.org/10.3389/conf.fnins.2015.91.00047",
    note="origin of FBER; a conference abstract, not a peer-reviewed paper",
)

_PROVINS_2023 = Reference(
    label="Provins et al. (2023)",
    detail="Frontiers in Neuroimaging 1:1073734",
    url="https://doi.org/10.3389/fnimg.2022.1073734",
    kind="paper",
    note="practical QC workflow for MRIQC and fMRIPrep visual reports",
)

_MRIQC_PROTOCOL = Reference(
    label="Esteban et al. — MRIQC protocol",
    detail="Nature Protocols (2026); preprint bioRxiv 2024.10.21.619532",
    url="https://doi.org/10.1101/2024.10.21.619532",
    note="visual assessment checklists and pre-specified exclusion criteria",
)

_MRIQC_REPORTS = Reference(
    label="MRIQC documentation — Visual reports",
    detail="mriqc.readthedocs.io",
    url="https://mriqc.readthedocs.io/en/latest/reports.html",
    kind="docs",
    note="what each reportlet in the visual report shows",
)

_NS_IQM_INTERPRETATION = Reference(
    label="NeuroStars — MRIQC: interpretation of IQMs",
    detail="neurostars.org/t/mriqc-interpretation-of-iqms/17690",
    url="https://neurostars.org/t/mriqc-interpretation-of-iqms/17690",
    kind="forum",
    note="community discussion of how to read IQMs without fixed cutoffs",
)

_NS_EXCLUDING = Reference(
    label="NeuroStars — Excluding subjects based on MRIQC",
    detail="neurostars.org/t/excluding-subjects-based-on-mriqc/2662",
    url="https://neurostars.org/t/excluding-subjects-based-on-mriqc/2662",
    kind="forum",
    note="recurring question about exclusion criteria; no consensus threshold",
)

_NS_FD_MISMATCH = Reference(
    label="NeuroStars — Different FDs in MRIQC vs fMRIPrep",
    detail="neurostars.org/t/different-fds-in-mriqc-vs-fmriprep/1494",
    url="https://neurostars.org/t/different-fds-in-mriqc-vs-fmriprep/1494",
    kind="forum",
    note="why the two pipelines report different FD for the same run",
)

_NS_TSNR = Reference(
    label="NeuroStars — Proper tSNR value for multiband BOLD?",
    detail="neurostars.org/t/what-is-the-proper-value-to-the-snr-tsnr-"
           "of-multi-band-bold-image/24763",
    url="https://neurostars.org/t/what-is-the-proper-value-to-the-snr-tsnr-"
        "of-multi-band-bold-image/24763",
    kind="forum",
    note="the absolute-tSNR-threshold question, asked and left unresolved",
)

_TUKEY = Reference(
    label="Tukey (1977)",
    detail="Exploratory Data Analysis, Addison-Wesley",
    url="https://archive.org/details/exploratorydataa0000tuke",
    kind="paper",
    note="origin of the 1.5x IQR fence used for automatic flagging",
)


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

def get_guidance(key: str) -> MeasureGuidance | None:
    """Return guidance for a measure, or ``None`` if it has no entry.

    Parameters
    ----------
    key : str
        Metric name, e.g. ``'fd_mean'``.
    """
    return MEASURE_GUIDANCE.get(key)


def guidance_for_modality(modality: str) -> list[MeasureGuidance]:
    """Return every measure documented for *modality*, in registry order.

    Parameters
    ----------
    modality : str
        ``'bold'``, ``'T1w'``, ``'T2w'``, or ``'dwi'``.
    """
    return [g for g in MEASURE_GUIDANCE.values() if modality in g.modalities]


def guidance_for_keys(keys: Iterable[str]) -> list[MeasureGuidance]:
    """Return guidance for *keys*, skipping any that are undocumented.

    Preserves the caller's ordering so the glossary can follow the column
    order of the table it explains.
    """
    return [MEASURE_GUIDANCE[k] for k in keys if k in MEASURE_GUIDANCE]


def undocumented_keys(keys: Iterable[str]) -> list[str]:
    """Return the subset of *keys* with no registry entry.

    Used by the test suite to assert that every column the dashboard
    renders carries guidance.
    """
    return [k for k in keys if k not in MEASURE_GUIDANCE]


def all_references() -> list[Reference]:
    """Return every distinct reference across the registry and process notes.

    Deduplicated by URL, ordered by first appearance, so the dashboard can
    render a single bibliography.
    """
    seen: dict[str, Reference] = {}
    for g in MEASURE_GUIDANCE.values():
        for ref in g.references:
            seen.setdefault(ref.url, ref)
    for note in PROCESS_GUIDANCE:
        for ref in note.references:
            seen.setdefault(ref.url, ref)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Measure registry
# ---------------------------------------------------------------------------

_IQR_FLAG = (
    "Flagged when outside the 1.5x IQR fence of the runs in scope. "
    "No absolute cutoff is applied."
)

_ALL_MODALITIES = ("bold", "T1w", "T2w", "dwi")

MEASURE_GUIDANCE: dict[str, MeasureGuidance] = {}


def _register(g: MeasureGuidance) -> None:
    """Add an entry to the registry, rejecting duplicate keys."""
    if g.key in MEASURE_GUIDANCE:
        raise ValueError(f"Duplicate guidance entry for {g.key!r}")
    MEASURE_GUIDANCE[g.key] = g


# --- Motion (BOLD and DWI) -------------------------------------------------

_register(MeasureGuidance(
    key="fd_mean",
    label="Mean framewise displacement",
    modalities=("bold", "dwi"),
    direction="lower_better",
    units="mm",
    why=(
        "Head motion is the dominant nuisance in functional MRI and the one "
        "most likely to masquerade as a real effect. It biases connectivity "
        "estimates systematically rather than randomly, and because motion "
        "correlates with age, clinical status and task difficulty, it can "
        "manufacture group differences outright."
    ),
    look_for=(
        "Open the MRIQC carpet plot and the FD trace together. A few tall "
        "spikes on an otherwise flat trace is transient motion and is usually "
        "salvageable by censoring; a trace that drifts upward across the run "
        "means the participant is settling or sliding and the later volumes "
        "are misregistered. Check whether FD spikes line up with dark or "
        "bright bands running across the whole carpet plot — that coincidence "
        "is what distinguishes real motion from a noisy estimate."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "For excluding a whole run, the citable rule is Parkes et al. (2018), "
        "whose stringent criterion excludes a participant for mean FD above "
        "0.25 mm, more than 20% of frames above 0.2 mm, or any single frame "
        "above 5 mm. Note what this is not: Power's widely quoted 0.5 mm "
        "(2012) and 0.2 mm (2014) are thresholds for censoring individual "
        "frames, not for excluding runs. Applying them to a run's mean FD is "
        "a category error, and a common one."
    ),
    caveats=(
        "FD sums absolute rotational and translational displacements, "
        "converting rotations to millimetres on a sphere of radius 50 mm. "
        "Its value depends on TR — shorter TRs sample motion more finely and "
        "report higher FD for identical movement — so it does not compare "
        "across studies with different TRs. In multiband or fast-TR data, "
        "respiration enters the motion estimates as pseudo-motion from "
        "B0 changes rather than true head displacement, inflating FD; "
        "neither MRIQC nor fMRIPrep filters this by default, so thresholds "
        "derived from single-band data are systematically too strict here."
    ),
    references=(_PARKES_2018, _POWER_2012, _POWER_2014, _MRIQC_MEASURES,
                _NS_EXCLUDING),
))

_register(MeasureGuidance(
    key="fd_num",
    label="Number of high-motion frames",
    modalities=("bold", "dwi"),
    direction="lower_better",
    units="volumes",
    why=(
        "Distinguishes a run ruined by a few discrete jerks from one degraded "
        "throughout. The former can be censored and still yield usable data; "
        "the latter cannot. Mean FD alone conflates the two."
    ),
    look_for=(
        "Compare against the run's total volume count to judge how much data "
        "survives censoring, and check on the FD trace whether the flagged "
        "frames are clustered — a single burst at the start of a run is far "
        "less costly than the same count scattered across it, because "
        "censoring removes neighbouring frames too."
    ),
    auto_flag=_IQR_FLAG,
    caveats=(
        "MRIQC counts frames above its own FD threshold (0.2 mm by default), "
        "which is not necessarily the threshold used elsewhere in this "
        "dashboard. Read it as a count under MRIQC's definition."
    ),
    references=(_POWER_2014, _MRIQC_MEASURES),
))

_register(MeasureGuidance(
    key="fd_perc",
    label="Percent of high-motion frames",
    modalities=("bold", "dwi"),
    direction="lower_better",
    units="%",
    why=(
        "The scale-free version of fd_num, and the quantity that actually "
        "determines whether enough data survives censoring. Runs of different "
        "lengths are only comparable on this measure."
    ),
    look_for=(
        "Decide whether the surviving volumes still cover the design. For "
        "task data, check that censoring does not fall disproportionately on "
        "one condition — losing 20% of volumes matters much more if they are "
        "concentrated in a single trial type than if they are spread evenly."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "fd_perc above 20 is directly backed by Parkes et al. (2018), whose "
        "stringent criterion excludes a participant when more than 20% of "
        "frames exceed 0.2 mm — the same threshold MRIQC uses by default, so "
        "the two line up exactly. Derived on resting-state connectivity data."
    ),
    caveats=(
        "This is MRIQC's percentage, computed at 0.2 mm over N frames. It is "
        "not interchangeable with this dashboard's pct_high_motion, which "
        "uses a different threshold and denominator."
    ),
    references=(_PARKES_2018, _POWER_2014, _MRIQC_MEASURES),
))

_register(MeasureGuidance(
    key="mean_fd",
    label="Mean FD (from fMRIPrep confounds)",
    modalities=("bold",),
    direction="lower_better",
    units="mm",
    why=(
        "Computed here from the fMRIPrep confounds of the variant actually "
        "feeding the analysis streams, so it describes the data being "
        "analysed rather than the raw series MRIQC saw. It is also the only "
        "measure the auto-stub decision looks at."
    ),
    look_for=(
        "Cross-check it against MRIQC's fd_mean in the same row. Modest "
        "differences are expected and do not indicate a problem. A large "
        "discrepancy points at a registration failure in one pipeline or the "
        "other and is worth opening both reports over."
    ),
    auto_flag=(
        "Not flagged by the IQR fence. Used by the auto-stub generator: runs "
        "above its threshold are written as 'investigate' rather than 'keep'. "
        "The threshold appears in the decision's reason string."
    ),
    caveats=(
        "Averaging hides the shape of the motion. A run with one enormous "
        "spike and a run with constant low-level movement can share a mean; "
        "always read it alongside the percentage of high-motion frames. It "
        "will not exactly equal MRIQC's fd_mean even though both use the "
        "Power formula with a 50 mm radius: MRIQC estimates motion with "
        "AFNI 3dvolreg on despiked, deobliqued data, while fMRIPrep uses its "
        "own head-motion correction referenced to the image centre of "
        "gravity. The difference is in the motion estimation, not the "
        "formula."
    ),
    references=(_POWER_2012, _PARKES_2018, _FMRIPREP_PAPER, _NS_FD_MISMATCH),
))

_register(MeasureGuidance(
    key="pct_high_motion",
    label="Percent of volumes above the FD threshold",
    modalities=("bold",),
    direction="lower_better",
    units="%",
    why=(
        "Estimates directly how much of the run would be lost to censoring, "
        "which is the practical question behind a keep/exclude decision."
    ),
    look_for=(
        "Read it together with mean FD. High percentage with low mean means "
        "many small excursions — often respiration in multiband data rather "
        "than true head movement, and less damaging than the number suggests. "
        "Low percentage with high mean means a few large events, which "
        "censoring handles well."
    ),
    auto_flag=(
        "Not flagged automatically. Reported for context alongside mean FD; "
        "the threshold used is shown on the motion chart."
    ),
    caveats=(
        "The threshold is a dashboard parameter defaulting to 0.5 mm, not a "
        "property of the data, so this number is only comparable across runs "
        "generated with the same setting. Do not read Parkes et al.'s 20% "
        "criterion against it: that criterion counts frames above 0.2 mm, "
        "and MRIQC's fd_perc column is the one that matches it. This measure "
        "also divides by the number of frames with a defined FD, which is "
        "one fewer than MRIQC uses."
    ),
    references=(_POWER_2012, _PARKES_2018, _FMRIPREP_PAPER),
))

# --- Temporal quality (BOLD) -----------------------------------------------

_register(MeasureGuidance(
    key="tsnr",
    label="Temporal signal-to-noise ratio",
    modalities=("bold",),
    direction="higher_better",
    why=(
        "Sets the ceiling on what any analysis can detect: the mean of each "
        "voxel's timeseries divided by its standard deviation over time, "
        "reported as the median across the brain mask. Low tSNR means the "
        "effect you are looking for is buried in noise regardless of how the "
        "data are modelled. It is also the most direct check on whether a "
        "denoising step helped."
    ),
    look_for=(
        "Localise the loss on the MRIQC report before acting. Global tSNR "
        "collapse across the brain suggests a coil or reconstruction problem "
        "and often affects a whole session. Loss confined to orbitofrontal "
        "and anterior temporal regions is ordinary susceptibility dropout and "
        "is not a defect. Loss following the motion trace is motion, and "
        "should be judged on the FD measures instead."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None. The widely repeated 'below 20 is poor' has no published "
        "derivation and should not be used as a cutoff. tSNR scales with "
        "voxel volume, TR, field strength, multiband factor and coil — 30 can "
        "be excellent at 1.5 mm isotropic multiband and poor at 4 mm "
        "single-band — so a defensible expectation exists only within a fixed "
        "protocol. MRIQC states a direction and no threshold. Where a number "
        "is genuinely needed, Murphy et al. (2007) frame tSNR against the "
        "effect size and scan duration you need rather than a pass mark. "
        "Judge it against this dataset's own distribution."
    ),
    caveats=(
        "Do not read fBIRN stability figures as human-subject cutoffs; those "
        "are phantom QA measures for cross-site scanner monitoring. Also note "
        "that tSNR rises with spatial smoothing and with any denoising that "
        "removes variance, including denoising that removes signal. A NORDIC "
        "or ICA-denoised series will show higher tSNR than its raw "
        "counterpart almost by construction, so an increase is not by itself "
        "proof of better data."
    ),
    references=(
        _MRIQC_MEASURES,
        Reference(
            label="Krüger & Glover (2001)",
            detail="Magnetic Resonance in Medicine 46(4):631-637",
            url="https://doi.org/10.1002/mrm.1240",
            note="the tSNR definition MRIQC simplifies",
        ),
        Reference(
            label="Murphy et al. (2007)",
            detail="NeuroImage 34(2):565-574",
            url="https://doi.org/10.1016/j.neuroimage.2006.09.032",
            note="relates tSNR to detectable effect size and scan duration "
                 "rather than to a pass/fail cutoff",
        ),
        _MRIQC_PAPER,
        _NS_TSNR,
    ),
))

_register(MeasureGuidance(
    key="dvars_std",
    label="Standardized DVARS",
    modalities=("bold",),
    direction="lower_better",
    why=(
        "Measures how much the image intensity changes from one volume to the "
        "next, standardized so that pure noise sits near 1. Unlike FD, which "
        "is derived from realignment parameters, DVARS looks at the images "
        "themselves, so it catches intensity artifacts that leave no trace in "
        "the motion estimates — spikes, dropout, reconstruction glitches."
    ),
    look_for=(
        "Read it against FD. Excursions in both together are head motion. "
        "DVARS excursions with a flat FD trace are the interesting case: "
        "hardware spikes, RF interference, or a reconstruction fault, none of "
        "which censoring by FD alone would remove. Confirm on the carpet "
        "plot, where these appear as bright or dark full-width bands."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "Standardized DVARS is scaled so that clean data sits near 1; values "
        "meaningfully above 1 indicate frame-to-frame change beyond noise. "
        "fMRIPrep's own default spike threshold is 1.5 standardized, which is "
        "consistent with that scaling. Afyouni and Nichols (2018) give the "
        "formal null distribution, and argue for p-values in place of "
        "arbitrary cutoffs altogether."
    ),
    caveats=(
        "Do not apply Power et al.'s DVARS threshold of 5 to this column. "
        "That value belongs to non-standardized, mode-1000-scaled DVARS "
        "(the dvars_nstd column), and mixing the two is consequential — 5 is "
        "unreachable on a standardized scale where 1.5 is already a spike. "
        "DVARS also depends on the preprocessing stage at which it is "
        "computed, so values only compare across identically processed runs."
    ),
    references=(
        Reference(
            label="Afyouni & Nichols (2018)",
            detail="NeuroImage 172:291-312, PII S1053811917311229",
            url="https://www.sciencedirect.com/science/article/pii/S1053811917311229",
            note="standardization and formal inference for DVARS",
        ),
        _POWER_2012,
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="dvars_nstd",
    label="Non-standardized DVARS",
    modalities=("bold",),
    direction="lower_better",
    why=(
        "The same frame-to-frame intensity change in raw image units, before "
        "standardization. Useful for checking whether an unusual "
        "standardized value reflects genuinely large intensity swings or "
        "merely an unusually quiet baseline in the denominator."
    ),
    look_for=(
        "Only interpretable next to dvars_std. If the standardized value is "
        "high but this one is ordinary for the dataset, the run's baseline "
        "variance is low rather than its excursions being large."
    ),
    auto_flag=_IQR_FLAG,
    caveats=(
        "Expressed in mode-1000-scaled units, roughly ten times percent BOLD "
        "change, so it is not comparable across sequences or scaling "
        "conventions. This is the column Power et al.'s DVARS threshold of 5 "
        "refers to, not dvars_std."
    ),
    references=(_POWER_2012, _MRIQC_MEASURES),
))

_register(MeasureGuidance(
    key="gcor",
    label="Global correlation",
    modalities=("bold",),
    direction="context",
    why=(
        "The average correlation between every pair of voxel timeseries. It "
        "summarises how much of the signal is a single brain-wide fluctuation "
        "rather than regional structure. Elevated values are a known signature "
        "of motion and respiratory contamination, and they bias connectivity "
        "estimates upward across the board."
    ),
    look_for=(
        "Treat an unusual value as a prompt to check motion and physiological "
        "confounds rather than as a defect in itself. On the carpet plot, "
        "high global correlation appears as broad synchronous banding across "
        "all tissue types at once."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None. Saad et al. (2013) introduced GCOR as a comparative measure "
        "and specifically to reason about the consequences of global signal "
        "regression; there is no absolute good value."
    ),
    caveats=(
        "Strongly affected by whether global signal regression has been "
        "applied, so it says as much about the preprocessing as about the "
        "data. It is also legitimately task-dependent — a strongly driven "
        "task raises it."
    ),
    references=(
        Reference(
            label="Saad et al. (2013)",
            detail="Brain Connectivity 3(4):339-352",
            url="https://doi.org/10.1089/brain.2013.0156",
            note="defines GCOR as a comparative summary of global correlation",
        ),
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="aqi",
    label="AFNI quality index",
    modalities=("bold",),
    direction="lower_better",
    why=(
        "One minus the rank correlation between each volume and the run's "
        "median volume, averaged over the run. It catches volumes that are "
        "structurally wrong rather than merely displaced, complementing the "
        "realignment-based motion measures."
    ),
    look_for=(
        "An elevated value with unremarkable FD points at intensity or "
        "reconstruction artifacts. Confirm on the carpet plot and check "
        "whether the affected volumes cluster at the start of the run, where "
        "incomplete magnetisation steady state is the usual explanation and "
        "the fix is dropping dummy volumes rather than excluding the run."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None, and AFNI says so in the tool itself: 3dTqual's documentation "
        "states it is intended as a guide to help find data problems and no "
        "more, leaving 'much higher than others' for the user to define. The "
        "one quantitative rule it does offer is relative — treat values "
        "outside the median plus or minus 3.5 times the median absolute "
        "deviation as worth examining. Prefer that to any absolute cutoff."
    ),
    references=(
        _MRIQC_MEASURES,
        Reference(
            label="AFNI 3dTqual",
            detail="AFNI program documentation",
            url="https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTqual.html",
            kind="software",
            note="defines the quality index and disclaims absolute interpretation",
        ),
    ),
))

_register(MeasureGuidance(
    key="aor",
    label="AFNI outlier ratio",
    modalities=("bold",),
    direction="lower_better",
    why=(
        "The mean fraction of voxels flagged as outliers per volume. Where "
        "AQI asks how different a volume is overall, this asks how much of it "
        "is affected, separating a localised artifact from a whole-volume one."
    ),
    look_for=(
        "Localise the outlier voxels on the MRIQC report. Confined to the "
        "brain edge, this is motion or ghosting; distributed through the "
        "interior, suspect a reconstruction or spike artifact."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None. AFNI states only that some outliers are expected and that a "
        "large fraction warrants investigating the dataset more fully. As "
        "with AQI, the usable rule is relative: median plus or minus 3.5 "
        "median absolute deviations across runs."
    ),
    caveats=(
        "A small non-zero value is expected by construction, not a defect: "
        "the outlier test carries a built-in per-voxel false-positive rate. "
        "Judge this measure by its position in the distribution, never by "
        "its distance from zero."
    ),
    references=(
        _MRIQC_MEASURES,
        Reference(
            label="AFNI 3dToutcount",
            detail="AFNI program documentation",
            url="https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dToutcount.html",
            kind="software",
            note="defines the per-volume outlier count underlying AOR",
        ),
    ),
))

# --- Ghosting (EPI) --------------------------------------------------------

_register(MeasureGuidance(
    key="gsr_x",
    label="Ghost-to-signal ratio (x)",
    modalities=("bold", "dwi"),
    direction="lower_better",
    why=(
        "Quantifies Nyquist (N/2) ghosting, the replica of the brain that EPI "
        "sequences project half a field of view away along the phase-encoding "
        "direction. Ghosts overlap real tissue and inject signal that is "
        "correlated with the brain but spatially wrong."
    ),
    look_for=(
        "Look for a faint offset copy of the brain in the background of the "
        "MRIQC mosaic, displaced along the phase-encode axis. Confirm that "
        "the displacement direction matches the axis this metric names — a "
        "ghost along the other axis suggests the phase-encoding direction is "
        "mislabelled in the BIDS metadata, which matters for fieldmap "
        "correction."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None. MRIQC publishes no cutoff, and the source it cites is a "
        "phantom study of how sequence parameters affect ghosting rather "
        "than a proposal for excluding human data."
    ),
    caveats=(
        "Only the axis that is actually phase-encoded is meaningful; the "
        "other reflects background noise. Confirm which applies from the "
        "PhaseEncodingDirection field in the BIDS sidecar before reading "
        "either value — MRIQC computes both regardless."
    ),
    references=(
        Reference(
            label="Giannelli et al. (2010)",
            detail="Journal of Applied Clinical Medical Physics 11(4):3237",
            url="https://doi.org/10.1120/jacmp.v11i4.3237",
            note="phantom characterisation of Nyquist ghosting in EPI; "
                 "proposes no exclusion cutoff",
        ),
        _MRIQC_MEASURES,
        _MRIQC_REPORTS,
    ),
))

_register(MeasureGuidance(
    key="gsr_y",
    label="Ghost-to-signal ratio (y)",
    modalities=("bold", "dwi"),
    direction="lower_better",
    why=(
        "As gsr_x, for the other in-plane axis. Which of the two carries the "
        "signal depends on the phase-encoding direction of the acquisition."
    ),
    look_for=(
        "Same check as gsr_x: an offset replica of the brain in the "
        "background of the mosaic, displaced along the phase-encode axis."
    ),
    auto_flag=_IQR_FLAG,
    caveats=(
        "Interpret only for the axis that is actually phase-encoded; the "
        "other is dominated by background noise."
    ),
    references=(
        Reference(
            label="Giannelli et al. (2010)",
            detail="Journal of Applied Clinical Medical Physics 11(4):3237",
            url="https://doi.org/10.1120/jacmp.v11i4.3237",
            note="phantom characterisation of Nyquist ghosting in EPI; "
                 "proposes no exclusion cutoff",
        ),
        _MRIQC_MEASURES,
        _MRIQC_REPORTS,
    ),
))

# --- Shared image-quality measures ----------------------------------------

_register(MeasureGuidance(
    key="efc",
    label="Entropy focus criterion",
    modalities=_ALL_MODALITIES,
    direction="lower_better",
    why=(
        "Measures the Shannon entropy of voxel intensities as a proxy for "
        "blurring and ghosting. Motion during acquisition spreads signal "
        "across voxels, which raises entropy — so this catches within-scan "
        "motion in structural images, where there is no timeseries and "
        "therefore no FD to inspect."
    ),
    look_for=(
        "Open the mosaic and look for ringing at tissue boundaries, blurred "
        "grey/white contrast, and ghost replicas. On a T1w image, check "
        "whether cortical ribbon detail is crisp at the occipital pole, where "
        "motion blur shows earliest."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None absolute. Atkinson et al. (1997) introduced EFC as a focus "
        "criterion to be minimised during motion correction, not as a "
        "pass/fail cutoff. MRIQC scales it by the maximum entropy of an image "
        "of the same size so that differently sized images are comparable."
    ),
    references=(
        Reference(
            label="Atkinson et al. (1997)",
            detail="IEEE Transactions on Medical Imaging 16(6):903-910",
            url="https://doi.org/10.1109/42.650886",
            note="introduces the entropy focus criterion for motion artifacts",
        ),
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="fber",
    label="Foreground-background energy ratio",
    modalities=_ALL_MODALITIES,
    direction="higher_better",
    why=(
        "The ratio of mean energy inside the head to mean energy outside it. "
        "Air should be empty; when it is not, the background is carrying "
        "ghosts, motion smear, or reconstruction artifacts, and the head "
        "signal is correspondingly compromised."
    ),
    look_for=(
        "Inspect the background of the mosaic with the intensity window "
        "stretched. Structured background — an offset brain replica, stripes, "
        "or a bright rim — is diagnostic; uniform faint speckle is ordinary "
        "thermal noise and harmless."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None. The measure originates in the Quality Assessment Protocol, "
        "cited by MRIQC to a conference abstract rather than a peer-reviewed "
        "paper, and MRIQC documents only that higher is better."
    ),
    caveats=(
        "Meaningless if the background has been masked or denoised, which "
        "some vendor reconstructions do by default. MRIQC returns -1 when "
        "there is no signal outside the head mask, i.e. for an already "
        "skull-stripped or defaced image; that is a sentinel value, not a "
        "catastrophic score."
    ),
    references=(_MRIQC_MEASURES, _QAP_FBER),
))

_register(MeasureGuidance(
    key="snr",
    label="Signal-to-noise ratio",
    modalities=("bold", "dwi"),
    direction="higher_better",
    why=(
        "A static, single-volume estimate of signal against background noise. "
        "For functional data it is a weaker guide than tSNR, which captures "
        "the temporal stability that actually limits detection, but it does "
        "catch coil and reconstruction faults that tSNR can mask."
    ),
    look_for=(
        "Check for coil dropout: a region of the brain that is systematically "
        "darker than its counterpart on the other side, usually adjacent to a "
        "failed coil element. This is visible on the mosaic and is a scanner "
        "fault worth reporting, not just a data-quality issue."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None that transfers across protocols; it scales with voxel volume, "
        "field strength and coil."
    ),
    caveats=(
        "For BOLD data prefer tSNR. Read this measure mainly as a check on "
        "hardware, not on usability."
    ),
    references=(_MRIQC_MEASURES,),
))

_register(MeasureGuidance(
    key="fwhm_avg",
    label="Estimated smoothness (average FWHM)",
    modalities=_ALL_MODALITIES,
    direction="lower_better",
    units="voxels",
    why=(
        "Estimates the spatial smoothness actually present in the image. "
        "Higher means blurrier: resolution has been lost somewhere — to "
        "motion, to interpolation, or to smoothing applied upstream without "
        "being recorded. It is the check that the data have the resolution "
        "the protocol claims."
    ),
    look_for=(
        "Look for blurring on the mosaic — soft tissue boundaries and a "
        "cortical ribbon that is hard to resolve — and check whether the run "
        "was resampled or smoothed before MRIQC saw it."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold="None. MRIQC documents only that lower is better.",
    caveats=(
        "Reported in voxel units, not millimetres: MRIQC divides the AFNI "
        "estimate by the voxel dimensions, and fwhm_avg is the mean across x, "
        "y and z. That makes it roughly comparable across protocols with "
        "different voxel sizes, but it also means it cannot be read directly "
        "against a millimetre smoothing kernel. The estimator also assumes "
        "stationary Gaussian smoothness, which holds poorly near tissue "
        "boundaries."
    ),
    references=(
        _MRIQC_MEASURES,
        Reference(
            label="Forman et al. (1995)",
            detail="Magnetic Resonance in Medicine 33(5):636-647",
            url="https://doi.org/10.1002/mrm.1910330508",
            note="the smoothness estimation MRIQC cites for FWHM",
        ),
        Reference(
            label="AFNI 3dFWHMx",
            detail="AFNI program documentation",
            url="https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dFWHMx.html",
            kind="software",
            note="the smoothness estimator MRIQC calls",
        ),
    ),
))

# --- Anatomical ------------------------------------------------------------

_register(MeasureGuidance(
    key="cnr",
    label="Contrast-to-noise ratio",
    modalities=("T1w", "T2w"),
    direction="higher_better",
    why=(
        "Measures how separable grey and white matter are relative to noise. "
        "This is the property every downstream anatomical step depends on: "
        "poor grey/white contrast produces bad segmentation, which produces "
        "bad surfaces, bad registration, and bad tissue-based confound "
        "regressors in the functional data."
    ),
    look_for=(
        "Rather than judging the number, judge its consequence. Open the "
        "fMRIPrep or FreeSurfer segmentation overlay and check whether the "
        "white-matter boundary follows the actual boundary, especially in "
        "temporal poles and near the insula. If segmentation is correct, low "
        "CNR is tolerable; if it is wrong, the run needs attention regardless "
        "of what CNR says."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None that transfers. CNR depends on sequence, field strength and "
        "acceleration, so it is comparative within a protocol only."
    ),
    references=(
        Reference(
            label="Magnotta & Friedman (2006)",
            detail="Journal of Digital Imaging 19(2):140-147",
            url="https://doi.org/10.1007/s10278-006-0264-x",
            note="fBIRN SNR/CNR measurement conventions",
        ),
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="cjv",
    label="Coefficient of joint variation",
    modalities=("T1w", "T2w"),
    direction="lower_better",
    why=(
        "Combines the spread of grey- and white-matter intensities relative to "
        "their separation. It is sensitive to both head motion and intensity "
        "non-uniformity at once, which makes it one of the most useful single "
        "summaries of structural image quality — and a good early warning that "
        "segmentation will fail."
    ),
    look_for=(
        "Because it responds to two different problems, identify which one is "
        "driving it. Look at the mosaic for motion signs — ringing, blurred "
        "boundaries, ghosting — and separately for a smooth brightness "
        "gradient across the image indicating an uncorrected bias field. The "
        "INU measures help distinguish them."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None. Ganzetti et al. (2016) used CJV as an objective function for "
        "tuning intensity non-uniformity correction — a quantity to minimise "
        "during optimization, never a pass/fail cutoff. The additional "
        "sensitivity to head motion is MRIQC's characterisation rather than "
        "a claim of that paper."
    ),
    references=(
        Reference(
            label="Ganzetti et al. (2016)",
            detail="Frontiers in Neuroinformatics 10:10",
            url="https://doi.org/10.3389/fninf.2016.00010",
            note="uses CJV as the objective for tuning INU correction",
        ),
        _MRIQC_MEASURES,
        _PROVINS_2023,
    ),
))

_register(MeasureGuidance(
    key="snr_total",
    label="Signal-to-noise ratio (mean across tissues)",
    modalities=("T1w", "T2w"),
    direction="higher_better",
    why=(
        "The headline SNR figure, useful for catching gross acquisition "
        "problems such as coil dropout or an aborted and restarted scan."
    ),
    look_for=(
        "Check for regional darkening on the mosaic that follows coil "
        "geometry rather than anatomy, and confirm the scan covers the whole "
        "head without wrap-around."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None that transfers across protocols. Magnotta and Friedman (2006) "
        "reported grey-matter SNR spanning roughly 44 to 109 within 3T "
        "scanners alone, which is why a fixed cutoff cannot be stated."
    ),
    caveats=(
        "Despite the name this is not a whole-brain measurement: MRIQC "
        "computes it as the arithmetic mean of the per-tissue SNRs for CSF, "
        "grey matter and white matter. It therefore inherits the "
        "segmentation dependency of its components."
    ),
    references=(
        Reference(
            label="Magnotta & Friedman (2006)",
            detail="Journal of Digital Imaging 19(2):140-147",
            url="https://doi.org/10.1007/s10278-006-0264-x",
            note="fBIRN SNR/CNR measurement conventions",
        ),
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="snr_gm",
    label="Signal-to-noise ratio (grey matter)",
    modalities=("T1w", "T2w"),
    direction="higher_better",
    why=(
        "Tissue-specific SNR. Grey matter is the harder tissue to image well "
        "and the one cortical analyses depend on, so it degrades before the "
        "whole-brain figure does."
    ),
    look_for=(
        "Inspect cortical detail directly on the mosaic — whether the ribbon "
        "is resolvable from surrounding CSF — rather than relying on the "
        "number, which is computed within a segmentation that may itself be "
        "wrong."
    ),
    auto_flag=_IQR_FLAG,
    caveats=(
        "Computed inside tissue masks derived from the image, so a "
        "segmentation failure corrupts this measure and can make a bad image "
        "look acceptable."
    ),
    references=(_MRIQC_MEASURES,),
))

_register(MeasureGuidance(
    key="snr_wm",
    label="Signal-to-noise ratio (white matter)",
    modalities=("T1w", "T2w"),
    direction="higher_better",
    why=(
        "White matter is the most homogeneous tissue in the brain, so its SNR "
        "is the cleanest available estimate of acquisition noise, relatively "
        "uncontaminated by genuine anatomical variation."
    ),
    look_for=(
        "Check that deep white matter looks uniform on the mosaic. Visible "
        "mottling or banding there is an acquisition or reconstruction "
        "problem, since the underlying tissue is not."
    ),
    auto_flag=_IQR_FLAG,
    caveats=(
        "Shares the segmentation dependency of the other tissue-specific "
        "measures."
    ),
    references=(_MRIQC_MEASURES,),
))

_register(MeasureGuidance(
    key="qi_1",
    label="Mortamet quality index 1",
    modalities=("T1w", "T2w"),
    direction="lower_better",
    why=(
        "The proportion of voxels in the air background that carry structured "
        "artifact rather than noise. Because it looks only outside the head, "
        "it detects motion and ghosting independently of anatomy and of any "
        "segmentation, which makes it a useful cross-check on the "
        "tissue-based measures."
    ),
    look_for=(
        "Window the mosaic hard toward the dark end and examine the air. "
        "Structured content — ghosts, ripples, or a bright rim around the "
        "head — is what this measure is counting."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None. MRIQC states only that near-zero values are better, and "
        "publishes no cutoff. Mortamet et al. (2009) is the source of the "
        "index itself; do not treat any circulating number as their "
        "threshold without checking the paper."
    ),
    caveats=(
        "Meaningless if the background has been masked or denoised by the "
        "scanner reconstruction — the artifact it counts has been erased "
        "along with the noise. MRIQC returns -1 when the background mask "
        "comes out empty, which happens for skull-stripped or defaced "
        "images. That -1 is a sentinel, not a very good score, and it will "
        "sort to the extreme of any ranking."
    ),
    references=(
        Reference(
            label="Mortamet et al. (2009)",
            detail="Magnetic Resonance in Medicine 62(2):365-372",
            url="https://doi.org/10.1002/mrm.21992",
            note="defines QI1/QI2 for automatic structural quality screening",
        ),
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="qi_2",
    label="Mortamet quality index 2",
    modalities=("T1w", "T2w"),
    direction="lower_better",
    why=(
        "Complements QI1 by testing how well the background intensity "
        "distribution matches the chi-squared distribution expected from pure "
        "noise. Departure from that shape indicates the background contains "
        "something other than noise."
    ),
    look_for=(
        "Same inspection as QI1 — structured content in the air. Treat the "
        "two together; agreement between them is stronger evidence than "
        "either alone."
    ),
    auto_flag=_IQR_FLAG,
    caveats=(
        "Shares QI1's dependence on an unmasked, undenoised background."
    ),
    references=(
        Reference(
            label="Mortamet et al. (2009)",
            detail="Magnetic Resonance in Medicine 62(2):365-372",
            url="https://doi.org/10.1002/mrm.21992",
            note="defines QI1/QI2 for automatic structural quality screening",
        ),
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="inu_range",
    label="Intensity non-uniformity range",
    modalities=("T1w", "T2w"),
    direction="lower_better",
    why=(
        "Summarises the spread of the estimated bias field — the smooth, "
        "anatomically meaningless brightness gradient produced by coil "
        "sensitivity. Because segmentation algorithms key on absolute "
        "intensity, a strong bias field makes the same tissue classify "
        "differently in different parts of the brain."
    ),
    look_for=(
        "Look for a smooth brightness gradient across the mosaic, typically "
        "brighter near the coil elements at the surface and dimmer centrally. "
        "Then check whether bias correction handled it, by inspecting the "
        "segmentation overlay in the affected region."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "No published cutoff. A perfectly uniform image would give a range of "
        "zero; values are otherwise comparative within a protocol."
    ),
    caveats=(
        "Describes the bias field N4 estimated, so it characterises the "
        "uncorrected image. A large range that was correctly removed is far "
        "less concerning than a small one that was not."
    ),
    references=(
        Reference(
            label="Tustison et al. (2010)",
            detail="IEEE Transactions on Medical Imaging 29(6):1310-1320",
            url="https://doi.org/10.1109/TMI.2010.2046908",
            note="N4ITK bias field correction, the estimator behind these values",
        ),
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="inu_med",
    label="Intensity non-uniformity median",
    modalities=("T1w", "T2w"),
    direction="target_range",
    why=(
        "The median of the estimated bias field. Where inu_range describes "
        "how much the field varies, this describes its overall level, "
        "separating a globally mis-scaled image from a spatially distorted "
        "one."
    ),
    look_for=(
        "Interpret alongside inu_range; on its own it says little. A median "
        "near 1 with a wide range means strong spatial variation around a "
        "correct overall level."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "A median close to 1 corresponds to a multiplicative field that "
        "neither brightens nor dims the image overall. Not a quality cutoff."
    ),
    references=(
        Reference(
            label="Tustison et al. (2010)",
            detail="IEEE Transactions on Medical Imaging 29(6):1310-1320",
            url="https://doi.org/10.1109/TMI.2010.2046908",
            note="N4ITK bias field correction, the estimator behind these values",
        ),
        _MRIQC_MEASURES,
    ),
))

_register(MeasureGuidance(
    key="tpm_overlap_gm",
    label="Tissue probability map overlap (grey matter)",
    modalities=("T1w", "T2w"),
    direction="higher_better",
    why=(
        "Compares this image's grey-matter segmentation against the ICBM "
        "template's tissue priors after spatial normalization. It is the "
        "dashboard's most direct check that registration to standard space "
        "succeeded — which matters because every group-level result depends "
        "on that registration being right."
    ),
    look_for=(
        "This is a registration check, so check the registration. Open the "
        "fMRIPrep normalization report and confirm the outlines follow the "
        "brain edge and the ventricles line up. Do not exclude a run on a low "
        "value before confirming the alignment is genuinely wrong."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None. Overlap with a population template is legitimately lower for "
        "atypical anatomy, so a low value is not by itself evidence of poor "
        "image quality."
    ),
    caveats=(
        "Confounds three separate things — image quality, segmentation "
        "accuracy and registration accuracy. Diagnose which before acting."
    ),
    references=(_MRIQC_MEASURES,),
))

_register(MeasureGuidance(
    key="tpm_overlap_wm",
    label="Tissue probability map overlap (white matter)",
    modalities=("T1w", "T2w"),
    direction="higher_better",
    why=(
        "As the grey-matter version, for white matter. White matter is the "
        "more stable tissue across individuals, so a low value here is "
        "stronger evidence of a registration failure than the same value in "
        "grey matter."
    ),
    look_for=(
        "Check the normalization report, and specifically whether the corpus "
        "callosum and ventricular horns align with the template."
    ),
    auto_flag=_IQR_FLAG,
    references=(_MRIQC_MEASURES,),
))

_register(MeasureGuidance(
    key="tpm_overlap_csf",
    label="Tissue probability map overlap (CSF)",
    modalities=("T1w", "T2w"),
    direction="higher_better",
    why=(
        "As above, for cerebrospinal fluid. CSF overlap is the most sensitive "
        "of the three to ventricular size, so it flags atrophy and normal "
        "anatomical variation as readily as it flags misregistration."
    ),
    look_for=(
        "Compare ventricle size and shape against the template on the "
        "normalization report. Genuinely large ventricles are anatomy, not a "
        "defect, and should not be excluded on this basis."
    ),
    auto_flag=_IQR_FLAG,
    caveats=(
        "The least specific of the three overlap measures; weigh it least "
        "heavily."
    ),
    references=(_MRIQC_MEASURES,),
))

_register(MeasureGuidance(
    key="wm2max",
    label="White-matter to maximum intensity ratio",
    modalities=("T1w", "T2w"),
    direction="target_range",
    why=(
        "The ratio of median white-matter intensity to a high percentile of "
        "the whole image. It detects hyperintense structures brighter than "
        "white matter — fat, vessels, or contrast agent — which distort "
        "intensity normalization and therefore segmentation."
    ),
    look_for=(
        "Scan the mosaic for very bright non-brain structures: scalp fat, "
        "orbital fat, or vasculature. Confirm they fall outside the brain "
        "mask; if they are inside it, segmentation is compromised."
    ),
    auto_flag=_IQR_FLAG,
    literature_threshold=(
        "None you should rely on. MRIQC's documentation quotes an expected "
        "interval of roughly [0.6, 0.8], but its own function docstring "
        "instead says values close to 1.0 are better, and the two are "
        "attached to different definitions of the denominator than the code "
        "actually computes. Neither figure carries a citation. Use this "
        "measure comparatively within the dataset only."
    ),
    caveats=(
        "The least trustworthy entry in this registry: MRIQC's documentation "
        "and implementation disagree about how it is computed. Corroborate "
        "any flag visually before acting on it."
    ),
    references=(_MRIQC_MEASURES,),
))


# ---------------------------------------------------------------------------
# Process guidance
# ---------------------------------------------------------------------------

PROCESS_GUIDANCE: tuple[ProcessNote, ...] = (
    ProcessNote(
        key="division_of_labour",
        title="What is automatic, and what is yours to judge",
        body=(
            "The dashboard does exactly two things on its own: it flags runs "
            "whose IQMs sit outside this dataset's own distribution, and it "
            "reports motion summaries computed from fMRIPrep confounds. It "
            "does not decide whether a run is usable. A flag is a request to "
            "look, not a verdict — most flags resolve to 'keep' once you open "
            "the MRIQC report. Conversely, an unflagged run can still be "
            "unusable: relative flagging cannot catch a defect shared by every "
            "run, such as a coil fault or a wrong prescription affecting the "
            "whole session. Open the visual report for anything you exclude."
        ),
        references=(_MRIQC_PAPER, _PROVINS_2023, _MRIQC_PROTOCOL),
    ),
    ProcessNote(
        key="visual_report",
        title="Reading the visual report",
        body=(
            "The carpet plot is the highest-yield panel for functional runs. "
            "It stacks voxel timeseries by compartment — cortical grey, deep "
            "grey, white matter and CSF, cerebellum, and a 'crown' of "
            "brain-edge voxels — with the FD and DVARS traces above it. "
            "Banding in the crown region time-locked to FD spikes is the "
            "strongest single visual confirmation that motion is corrupting "
            "the data. For structural runs the workhorse is the mosaic: check "
            "tissue contrast and boundary sharpness, then window hard toward "
            "the dark end and look at the air, where ghosting and ringing "
            "show most clearly. The artifact classes worth learning to "
            "recognise are ringing, ghosting, motion waves emanating from the "
            "skull, wrap-around where the head exceeds the field of view, and "
            "orientation errors from mislabelled axes."
        ),
        references=(_MRIQC_REPORTS, _PROVINS_2023, _MRIQC_PROTOCOL),
    ),
    ProcessNote(
        key="outlier_rule",
        title="How runs get flagged (the 1.5x IQR fence)",
        body=(
            "For each metric the dashboard computes the first and third "
            "quartiles across the runs in scope and flags any value below "
            "Q1 - 1.5*IQR or above Q3 + 1.5*IQR. This is Tukey's fence. It is "
            "relative, not absolute: thresholds are recomputed from whichever "
            "runs are in scope, so the same run can be flagged in a "
            "single-subject dashboard and unflagged across all subjects, or "
            "vice versa. A flag therefore means 'unusual for this dataset', "
            "never 'bad in absolute terms'. Because every metric is tested "
            "independently, some flags are expected by chance alone — weigh a "
            "run flagged on one metric differently from one flagged on five."
        ),
        references=(_TUKEY, _MRIQC_PAPER),
    ),
    ProcessNote(
        key="no_universal_cutoffs",
        title="Why there are so few absolute thresholds here",
        body=(
            "MRIQC publishes no reference ranges for its measures, and that "
            "absence is deliberate rather than an oversight. IQMs carry "
            "strong site, scanner and protocol batch effects — Magnotta and "
            "Friedman reported grey-matter SNR ranging from roughly 44 to 109 "
            "within 3T scanners alone — so a value that is normal under one "
            "protocol is alarming under another. Esteban et al. found MRIQC's "
            "own quality classifier dropped sharply on unseen sites for the "
            "same reason. Identify outliers within this dataset rather than "
            "against fixed cutoffs. Where an a priori threshold does appear "
            "below, it is stated with its source and the population it came "
            "from; treat everything else as comparative."
        ),
        references=(_MRIQC_PAPER, _MRIQC_MEASURES, _NS_IQM_INTERPRETATION,
                    _NS_EXCLUDING),
    ),
    ProcessNote(
        key="sentinel_values",
        title="Values that look catastrophic but are not",
        body=(
            "MRIQC writes -1 for FBER and for QI1 when the image has no "
            "usable background — which is what a skull-stripped or defaced "
            "scan looks like to it. These are sentinels meaning 'not "
            "computable', not scores. They will sort to the extreme of any "
            "ranking and can be caught by the IQR fence, so check whether a "
            "run flagged on either measure simply arrived already stripped "
            "before reading anything into it. The same caution applies to "
            "every background-based measure, QI2 included: if the scanner "
            "reconstruction denoised the air, the artifact these measures "
            "count has been erased along with the noise."
        ),
        references=(_MRIQC_MEASURES,),
    ),
    ProcessNote(
        key="decision_vocabulary",
        title="What keep / investigate / exclude / pending mean downstream",
        body=(
            "Decisions are consumed by the pipeline gate, so the word you pick "
            "changes what gets analysed. 'keep' admits the run. 'exclude' "
            "removes it. 'investigate' is treated as 'exclude' by default, so "
            "it is a safe holding state but not a neutral one. 'pending' means "
            "nobody has signed off yet, and is also held out by default. Only "
            "the first three can be recorded by a person; 'pending' is what "
            "automation writes. Every decision is appended to a per-run JSON "
            "history with your identifier and a UTC timestamp, and nothing is "
            "overwritten — recording a reason is what makes an exclusion "
            "defensible later."
        ),
    ),
    ProcessNote(
        key="auto_stub_provenance",
        title="Decisions you did not make",
        body=(
            "Every run is pre-populated with a stub so the pipeline gate is "
            "never blocked by a missing file. Stubs are recorded as 'pending' "
            "under the reviewer identifier 'auto-stub', and the tooling is not "
            "permitted to write keep, exclude or investigate at all. What the "
            "stub thinks is stored separately as a recommendation and shown "
            "next to the PENDING badge as an advisory note — it gates nothing. "
            "Bear in mind the stub only ever saw mean FD: it inspected no IQM "
            "and no image. The dashboard's 'Signed Off' count therefore means "
            "what it says, and a run shows as reviewed only once a named "
            "person has recorded a call on it."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Rendering — HTML (consumed by qc_dashboard)
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def _render_references(refs: tuple[Reference, ...]) -> str:
    """Render a reference list as an HTML ``<ul>``."""
    if not refs:
        return ""
    items = []
    for r in refs:
        note = f' — <span class="ref-note">{_esc(r.note)}</span>' if r.note else ""
        items.append(
            f'<li class="ref ref-{r.kind}">'
            f'<a href="{_esc(r.url)}" target="_blank" rel="noopener">{_esc(r.label)}</a>'
            f' <span class="ref-detail">{_esc(r.detail)}</span>{note}</li>'
        )
    return f'<ul class="ref-list">{"".join(items)}</ul>'


def render_measure_card(g: MeasureGuidance) -> str:
    """Render one measure's guidance as a collapsible HTML block."""
    units = f' <span class="g-units">({_esc(g.units)})</span>' if g.units else ""
    rows = [
        ("Why it is here", g.why),
        ("What to look for", g.look_for),
        ("Flagged automatically", g.auto_flag),
    ]
    if g.literature_threshold:
        rows.insert(2, ("A priori threshold", g.literature_threshold))
    if g.caveats:
        rows.append(("Caveats", g.caveats))

    body = "".join(
        f'<div class="g-row"><span class="g-label">{label}</span>'
        f'<span class="g-text">{_esc(text)}</span></div>'
        for label, text in rows
    )

    return f"""<details class="guidance-card" id="guidance-{_esc(g.key)}">
  <summary><code>{_esc(g.key)}</code> — {_esc(g.label)}{units}
    <span class="g-dir g-dir-{g.direction}">{_esc(g.direction_label)}</span>
  </summary>
  <div class="guidance-body">{body}{_render_references(g.references)}</div>
</details>"""


def render_guidance_section(keys: Iterable[str]) -> str:
    """Render the measure glossary for the columns a dashboard displays.

    Parameters
    ----------
    keys : iterable of str
        Metric names, in the order they appear in the run table.
    """
    cards = [render_measure_card(g) for g in guidance_for_keys(keys)]
    if not cards:
        return "<p>No guidance registered for these measures.</p>"
    return "\n".join(cards)


def render_process_section() -> str:
    """Render the process-level guidance (how to review, what is automatic)."""
    blocks = []
    for note in PROCESS_GUIDANCE:
        blocks.append(
            f'<details class="guidance-card process-card" id="guidance-{_esc(note.key)}">'
            f"<summary>{_esc(note.title)}</summary>"
            f'<div class="guidance-body"><p class="g-text">{_esc(note.body)}</p>'
            f"{_render_references(note.references)}</div></details>"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Rendering — Markdown (standalone document)
# ---------------------------------------------------------------------------

def guidance_markdown(modalities: Iterable[str] = ("bold", "T1w", "T2w", "dwi")) -> str:
    """Render the whole registry as a standalone Markdown document.

    Lets the same guidance that appears in the dashboard be reviewed,
    diffed, and cited outside it.

    Parameters
    ----------
    modalities : iterable of str
        Which modality sections to include.
    """
    lines = [
        "# QC review guidance",
        "",
        "Generated from `src/python/neuroimaging/qc_guidance.py`. Edit the",
        "registry there rather than this file.",
        "",
        "## How to review",
        "",
    ]

    for note in PROCESS_GUIDANCE:
        lines += [f"### {note.title}", "", note.body, ""]
        for r in note.references:
            note_txt = f" — {r.note}" if r.note else ""
            lines.append(f"- [{r.label}]({r.url}), {r.detail}{note_txt}")
        lines.append("")

    for modality in modalities:
        entries = guidance_for_modality(modality)
        if not entries:
            continue
        lines += [f"## {modality}", ""]
        for g in entries:
            units = f" ({g.units})" if g.units else ""
            lines += [
                f"### `{g.key}` — {g.label}{units}",
                "",
                f"*{g.direction_label}*",
                "",
                f"**Why it is here.** {g.why}",
                "",
                f"**What to look for.** {g.look_for}",
                "",
                f"**Flagged automatically.** {g.auto_flag}",
                "",
            ]
            if g.literature_threshold:
                lines += [f"**A priori threshold.** {g.literature_threshold}", ""]
            if g.caveats:
                lines += [f"**Caveats.** {g.caveats}", ""]
            if g.references:
                lines.append("**Sources.**")
                lines.append("")
                for r in g.references:
                    note_txt = f" — {r.note}" if r.note else ""
                    lines.append(f"- [{r.label}]({r.url}), {r.detail}{note_txt}")
                lines.append("")

    return "\n".join(lines)

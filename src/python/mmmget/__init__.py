"""mmmget: the lab's getters for MMMData analysis inputs, and its result writer.

Call these instead of globbing derivatives or listing confound columns, and
the lab's standards come with the data:

    import mmmget

    for run in mmmget.runs(subject="##", task="<label>"):   # catalog, QC applied
        design = mmmget.confounds(run, regime="<name>")      # registry regime
        ...
    v1 = mmmget.roi("##", "schaefer400-7:LH_Vis_1")
    mmmget.write_result(table, spec, out_dir, "<stem>")      # mmmview-valid + provenance

Each getter records what it was asked for (``provenance_records()``), and
``write_result`` stamps that record, with ``__version__``, into the result's
spec, so a result says whether it followed the standard path. Escape hatches
(``confounds(run, columns=[...])``, ``runs(include_excluded=True)``) work and
are recorded as non-standard.

Where a standard is not decided yet the getter raises
:class:`~mmmget.errors.OpenDecisionError` naming the decision, rather than
guessing: ``confounds(run, regimes="standard")`` (the comparison set) and
``events(run, scheme=...)`` (where condition definitions live).
"""

from ._version import __version__
from .catalog_runs import Run, runs
from .conditions import events
from .confound_regimes import confounds
from .provenance import provenance_records, reset_provenance
from .results import provenance_block, write_result
from .rois import ROI, roi
from .errors import OpenDecisionError, SourceNotConfigured

__all__ = [
    "__version__",
    "runs", "Run",
    "confounds",
    "roi", "ROI",
    "events",
    "write_result", "provenance_block",
    "provenance_records", "reset_provenance",
    "OpenDecisionError", "SourceNotConfigured",
]

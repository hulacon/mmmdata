#!/usr/bin/env python3
"""MIGRATED 2026-08-17 into duckbrain: ``duckbrain.catalog.sweep``.

The Contract A engine moved out of mmmdata under the contracts §3.2
ownership ruling (the dataset owns the artifact and its declarations;
duckbrain owns the generic engine). This stub remains so old command
lines fail with the fix rather than a FileNotFoundError. Run instead:

    /gpfs/projects/hulacon/shared/envs/duckbrain/bin/python \\
        -m duckbrain.catalog sweep --root <bids_root>

(or the ``rebuild`` verb for all four tiers; ``scripts/catalog_rebuild.sbatch``
wraps it). Indexing scope now comes from the dataset declaration's
``[catalog]`` section (``<root>/expectations/dataset.toml``), not from flags.
The pre-migration implementation is in git history at this path.
"""

import sys

sys.exit(__doc__)

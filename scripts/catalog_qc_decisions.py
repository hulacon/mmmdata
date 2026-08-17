#!/usr/bin/env python3
"""MIGRATED 2026-08-17 into duckbrain: ``duckbrain.catalog.qc_decisions``.

The Contract A engine moved out of mmmdata under the contracts §3.2
ownership ruling — and this module is the one that gained from it: the
in-duckbrain version imports ``core/qc.py``'s reading semantics directly
instead of replicating the two on-disk decision schemas here. Run instead:

    /gpfs/projects/hulacon/shared/envs/duckbrain/bin/python \\
        -m duckbrain.catalog qc --root <bids_root>

The pre-migration implementation is in git history at this path.
"""

import sys

sys.exit(__doc__)

#!/usr/bin/env python3
"""MIGRATED 2026-08-17 into duckbrain: ``duckbrain.catalog.supplemental``.

The Contract A engine moved out of mmmdata under the contracts §3.2
ownership ruling. The raw-tree walk scope (previously hard-coded here) is
now declared in the dataset's ``[catalog] raw_scope``
(``<root>/expectations/dataset.toml``). Run instead:

    /gpfs/projects/hulacon/shared/envs/duckbrain/bin/python \\
        -m duckbrain.catalog supplemental --root <bids_root>

The pre-migration implementation is in git history at this path.
"""

import sys

sys.exit(__doc__)

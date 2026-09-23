"""Block energy of a fitted map at the union rungs (design record 6(a)).

The orthogonal map Q (k, k) lifted to voxel space is M = W Q W^T (V, V),
never materialised. For source block a and target block b,
||M_ab||_F^2 = tr(G_a Q G_b Q^T) with G_a = W_a^T W_a (k, k), so a
(n_blocks, n_blocks) energy matrix costs n_blocks^2 small products. Each row
is normalised by its source block's total: "rotation within" is diagonal
mass, "movement across" is off-diagonal mass.

Level predictions from the design record: level 1 (single-ROI rotation)
cross ~ 0; level 2 (remapping across ROIs) cross high; level 3
(retinotopic pos -> neg remapping) the predicted map within the CI of the
free fit.
"""

from __future__ import annotations

import numpy as np


def block_energy(Q: np.ndarray, W: np.ndarray, block_ids: np.ndarray) -> dict:
    """Returns blocks (labels), energy (n_b, n_b) row-normalised,
    energy_raw, source_totals and cross_fraction = off-diagonal share of the
    total energy (source-weighted)."""
    block_ids = np.asarray(block_ids)
    labels = np.unique(block_ids)
    G = [W[block_ids == b].T @ W[block_ids == b] for b in labels]
    n = len(labels)
    raw = np.zeros((n, n))
    for a in range(n):
        GaQ = G[a] @ Q
        for b in range(n):
            raw[a, b] = float(np.trace(GaQ @ G[b] @ Q.T))
    totals = raw.sum(axis=1)
    norm = raw / np.where(totals > 0, totals, 1.0)[:, None]
    diag = np.trace(raw)
    cross = float(1 - diag / totals.sum()) if totals.sum() > 0 else float("nan")
    return {"blocks": labels, "energy": norm, "energy_raw": raw,
            "source_totals": totals, "cross_fraction": cross}


def predicted_retinotopic_map(*args, **kwargs):
    """Level-3 predicted map from retinotopic correspondence between the
    positive and negative pRF populations. Needs the warped pRF angle and
    eccentricity maps under functional_rois/sub-##/space-MNI152NLin2009cAsym_res-2/
    (written by build_prf_masks.py) and a stated correspondence rule; the
    rule is not decided in the design record yet."""
    raise NotImplementedError(
        "level-3 predicted map: correspondence rule between pos and neg pRF "
        "populations is not decided; see the neural-rotation-pilot charter, "
        "Settles-when 6(b)")

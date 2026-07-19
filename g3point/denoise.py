"""Point-cloud denoising for the G3Point pipeline.

The MATLAB reference removes outliers with the proprietary `pcdenoise`. That is NOT reproducible
by standard statistical outlier removal: the best a standard SOR achieves against MATLAB's exact
inlier set is retained-set Jaccard ~0.99, but the removed-set Jaccard is much lower and varies tile
to tile (0.275, 0.530, 0.853, 0.927 on the four fixtures at the default parameters), and the removal
counts are inconsistent -- so `pcdenoise` is a different algorithm in the same family, not merely
different parameters.

This module therefore provides an explicit, documented standard SOR as a deliberate OPEN-SOURCE
SUBSTITUTE for `pcdenoise`. It does not reproduce `pcdenoise` bit-for-bit and is NOT tuned to
match MATLAB's output: `std_ratio` / `n_neighbors` are ordinary parameters (defaults below) set
per run in the `.ini`. Because SOR != `pcdenoise`, the port's grain-size distribution runs slightly
finer than MATLAB; held-out validation (40 field tiles, port end-to-end vs the existing MATLAB
`granulo`) at the default `std_ratio=3.0` gave median dD50 -0.8 mm / dD84 -3.4 mm -- the expected,
quantified difference between two legitimate denoisers, not a defect. The full account (and the
held-out reproduction scripts, which use field data and live outside this repository) is in
PARITY.md divergence #1. `denoise_concordance` reports retained- and removed-set agreement against
a MATLAB fixture inlier set.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# Default denoise strength. This is a plain parameter (like knn or rad_factor) that a user
# overrides in the .ini per run (denoise_n_neighbors / denoise_std_ratio); it is NOT tuned to
# reproduce MATLAB pcdenoise. At these defaults SOR removes ~1-1.5% of points on the bed tiles,
# the points whose mean neighbour distance is farthest above the cloud-wide mean (its outlier
# tail). std_ratio=3.0 (vs a tighter 2.0-2.5) is deliberately conservative given the small
# neighbour count: it removes only clear outliers and avoids trimming legitimately-sparse grain
# edges. See PARITY.md divergence #1 -- this is an open-source substitute for pcdenoise and does
# not reproduce it bit-for-bit.
DEFAULT_N_NEIGHBORS = 4
DEFAULT_STD_RATIO = 3.0


def statistical_outlier_removal(xyz: np.ndarray,
                                n_neighbors: int = DEFAULT_N_NEIGHBORS,
                                std_ratio: float = DEFAULT_STD_RATIO):
    """Remove statistical outliers by mean distance to the `n_neighbors` nearest points.

    A point is kept iff its mean distance to its `n_neighbors` nearest neighbours (excluding
    itself) is within ``mean + std_ratio * std`` of the global distribution. This is the
    standard SOR rule (same family as MATLAB `pcdenoise`, not bit-identical to it).

    Parameters
    ----------
    xyz : (n, 3) array
        Point coordinates.
    n_neighbors : int
        Number of nearest neighbours used for the local mean-distance statistic.
    std_ratio : float
        Threshold in standard deviations of the global mean-distance distribution.

    Returns
    -------
    xyz_kept : (m, 3) array
        The retained points.
    kept_indexes : (m,) int array
        Indices of the retained points into the input `xyz` (ascending).
    """
    n_int = int(n_neighbors)
    if n_int != n_neighbors or n_int < 1:
        raise ValueError(f"n_neighbors must be a positive integer, got {n_neighbors!r}")
    if not np.isfinite(std_ratio) or std_ratio < 0:
        raise ValueError(f"std_ratio must be finite and >= 0, got {std_ratio}")
    n_neighbors = n_int
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be an (n, 3) array, got shape {xyz.shape}")
    # need > n_neighbors points to form the n_neighbors+1 query (self + n_neighbors); at exactly
    # n_neighbors+1 the query is valid and SOR applies.
    if len(xyz) <= n_neighbors:
        return xyz, np.arange(len(xyz))
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, n_neighbors + 1)  # +1: the first neighbour is the point itself
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = mean_distance.mean() + std_ratio * mean_distance.std(ddof=0)
    kept_indexes = np.where(mean_distance <= threshold)[0]
    return xyz[kept_indexes], kept_indexes


def denoise_concordance(xyz: np.ndarray, matlab_inlier_indexes: np.ndarray,
                        n_neighbors: int = DEFAULT_N_NEIGHBORS,
                        std_ratio: float = DEFAULT_STD_RATIO) -> dict:
    """Audit SOR against a MATLAB `pcdenoise` inlier set (0-based indices into `xyz`).

    Returns the Jaccard overlap of the two kept sets plus both removal counts -- the quantity
    tracked in PARITY.md divergence #1.
    """
    n = len(xyz)
    all_idx = set(range(n))
    _, kept = statistical_outlier_removal(xyz, n_neighbors, std_ratio)
    a, b = set(kept.tolist()), set(np.asarray(matlab_inlier_indexes).tolist())
    ra, rb = all_idx - a, all_idx - b  # removed sets
    # removed-set Jaccard is the discriminating metric: retained-set Jaccard is dominated by
    # the many points both methods keep and looks deceptively high.
    removed_jacc = len(ra & rb) / len(ra | rb) if (ra or rb) else 1.0
    return {"jaccard": len(a & b) / len(a | b),
            "removed_jaccard": removed_jacc,
            "sor_removed": n - len(a),
            "matlab_removed": n - len(b)}

"""Point-cloud denoising for the G3Point pipeline.

The MATLAB reference removes outliers with `pcdenoise`. That is NOT reproducible by standard
statistical outlier removal: sweeping neighbour count and threshold, the closest a standard
SOR gets to MATLAB's exact inlier set is ~0.99 Jaccard, and the removal counts stay
inconsistent tile to tile -- so MATLAB's denoiser is a different (likely iterative /
robust-spread) algorithm in the same family, not merely different parameters. See PARITY.md
divergence #1.

This module therefore provides an explicit, documented standard SOR rather than trying to
replicate `pcdenoise`. It is ~99% concordant with MATLAB's inliers; the residual disagreement
is confined to borderline points near the outlier threshold. The Python port's equivalence to
MATLAB is certified at the grain-size-distribution level (not at bit-exact denoise) -- see the
Phase-4 acceptance gate. `denoise_concordance` supports auditing that ~99% against a MATLAB
fixture inlier set.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# Defaults chosen to maximise inlier concordance with MATLAB pcdenoise across the fixture
# tiles (min Jaccard ~0.99). They are an empirical approximation, certified downstream at the
# GSD level, NOT a claim of pcdenoise equivalence.
DEFAULT_N_NEIGHBORS = 4
DEFAULT_STD_RATIO = 2.5


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
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) <= n_neighbors + 1:
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

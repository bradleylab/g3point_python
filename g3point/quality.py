"""Ellipsoid fit-quality (Acover / Aqualityok) -- parity with the MATLAB G3Point.

The MATLAB reference (fitellipsoidtograins.m, randsamplingellipsoid.m) keeps a grain
in the grain-size distribution only if the fitted ellipsoid is actually SUPPORTED by
the grain's points: it samples the ellipsoid surface and measures the fraction of the
surface that has a grain point as its nearest neighbour (`Acover`), then requires
`Acover > a_quality_thresh` (`Aqualityok`). This rejects oversized / degenerate fits
whose ellipsoid is far larger than the point support -- exactly the grains that
otherwise inflate the coarse tail of the GSD.

The upstream python port parses `a_quality_thresh` but never computes `Acover`, so its
GSD retains those poor fits. This module restores the filter. Acover is stochastic in
the reference (MATLAB `rand`); we replicate the METHOD (200 random surface samples),
not the exact per-grain value, and expose an optional seeded RNG for reproducibility.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def grain_rng(run_seed: int, source_point_indexes) -> np.random.Generator:
    """A deterministic per-grain RNG stream, stable under grain reordering.

    Acover is stochastic; a single shared generator would make every grain's Acover depend on
    how many grains (and failed fits) preceded it, so results would shift under reordering or
    parallelisation. Instead seed each grain independently from ``(run_seed, grain_key)`` where
    the key is the grain's smallest source-point index -- unique across grains (their point sets
    are disjoint) and independent of iteration order.
    """
    key = int(np.min(source_point_indexes))
    return np.random.default_rng([int(run_seed), key])


def sample_ellipsoid_surface(center, radii, rotation_matrix, n=200, rng=None):
    """`n` random points on an ellipsoid surface -- port of randsamplingellipsoid.m.

    `radii` are the three semi-axes; `rotation_matrix` is the fit's rotation with the
    radii directions as ROWS (as returned by ellipsoid.implicit_to_explicit), so the
    axis-aligned local point maps to world as ``world = local @ rotation_matrix + center``.
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    v = rng.random(n)
    theta = u * 2.0 * np.pi
    phi = np.arccos(2.0 * v - 1.0)
    sin_phi = np.sin(phi)
    local = np.column_stack([
        radii[0] * sin_phi * np.cos(theta),
        radii[1] * sin_phi * np.sin(theta),
        radii[2] * np.cos(phi),
    ])
    return local @ np.asarray(rotation_matrix) + np.asarray(center)


def acover(xyz_grain, center, radii, rotation_matrix, n=200, rng=None):
    """Percentage of `n` ellipsoid-surface samples nearest to >=1 grain point.

    Mirrors fitellipsoidtograins.m lines 83-85: sample the surface, `knnsearch` each
    grain point to its nearest sample, ``Acover = 100 * unique(nearest) / n``. High
    Acover => the point cloud wraps the ellipsoid (good fit); low => oversized fit.
    """
    samples = sample_ellipsoid_surface(center, radii, rotation_matrix, n, rng)
    _, idx = cKDTree(samples).query(np.asarray(xyz_grain))
    return 100.0 * np.unique(idx).size / n


def aquality_ok(xyz_grain, center, radii, rotation_matrix, a_quality_thresh=10.0,
                n=200, rng=None):
    """MATLAB `Aqualityok`: True iff Acover exceeds `a_quality_thresh` (default 10)."""
    return acover(xyz_grain, center, radii, rotation_matrix, n, rng) > a_quality_thresh

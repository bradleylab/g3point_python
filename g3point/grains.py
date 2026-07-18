"""Typed per-grain results and the grain-size distribution.

MATLAB's `grainsizedistribution.m` keeps a grain in the GSD only if `fitok & Aqualityok`
(no diameter cut -- `min_diam` belongs to the separate grid-by-number workflow). This module
mirrors that: `compute_grains` fits an ellipsoid to every grain, runs the Acover fit-quality
test with a deterministic per-grain RNG, and `grain_size_distribution` returns the b-axis
diameters of the grains that pass `fitok & aqualityok`.

Coordinates in `GrainResult` are in the ANALYSIS frame (the point set the ellipsoids were fit
on, i.e. min-shifted if the class shifted). Add the class `mins` offset for the loaded/scan
frame. MATLAB's `centers` field is the fitted ellipsoid centre (`Ellipsoidm.c`) -- compare it
to `ellipsoid_center`, NOT to `point_centroid`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ellipsoid import fit_ellipsoid_to_grain
from .quality import acover, grain_rng

A_QUALITY_THRESH_DEFAULT = 10.0
MIN_POINTS_FOR_FIT = 4  # an ellipsoid fit needs at least this many points


@dataclass
class GrainResult:
    label: int
    n_points: int
    point_indexes: np.ndarray            # 0-based into the analysis cloud
    source_point_indexes: np.ndarray     # 0-based into the loaded source file
    point_centroid: np.ndarray           # arithmetic mean of the grain points (analysis frame)
    fitok: bool
    fail_reason: str | None = None
    radii: np.ndarray | None = None            # semi-axes, sorted descending [a, b, c]
    ellipsoid_center: np.ndarray | None = None # fitted ellipsoid centre (analysis frame)
    rotation: np.ndarray | None = None         # 3x3, rows = axis directions
    acover: float | None = None
    aqualityok: bool = False

    @property
    def b_axis_diameter(self) -> float:
        """Intermediate-axis grain diameter (2 * middle semi-axis), or NaN if not fitted."""
        return float(2.0 * self.radii[1]) if self.radii is not None else float("nan")


def compute_grains(xyz: np.ndarray, stacks, source_indexes: np.ndarray,
                   fit_method: str = "direct", a_quality_thresh: float = A_QUALITY_THRESH_DEFAULT,
                   run_seed: int = 42) -> list[GrainResult]:
    """Fit an ellipsoid + Acover to every stack and return a typed per-grain table.

    Parameters
    ----------
    xyz : (n, 3) array
        The analysis cloud the grains index into.
    stacks : sequence of index lists
        Per-grain point indexes into `xyz`.
    source_indexes : (n,) int array
        Map from analysis-cloud row -> loaded-file row, so grains can be joined back to source
        points (and to a MATLAB inlier set). Pass ``np.arange(len(xyz))`` if there is no
        upstream row removal.
    fit_method : str
        Ellipsoid fit method threaded through to `fit_ellipsoid_to_grain` ('direct'|'inertia').
    a_quality_thresh : float
        Aqualityok threshold: keep grains with Acover > this value.
    run_seed : int
        Base seed for the deterministic per-grain Acover RNG.
    """
    source_indexes = np.asarray(source_indexes)
    grains: list[GrainResult] = []
    for label, stack in enumerate(stacks):
        stack = np.asarray(stack, dtype=int)
        src = source_indexes[stack]
        pts = xyz[stack, :]
        centroid = pts.mean(axis=0)
        base = dict(label=label, n_points=len(stack), point_indexes=stack,
                    source_point_indexes=src, point_centroid=centroid)

        if len(stack) < MIN_POINTS_FOR_FIT:
            grains.append(GrainResult(fitok=False, fail_reason="too_few_points", **base))
            continue

        center, radii, _quat, rotation, _params = fit_ellipsoid_to_grain(pts, method=fit_method)
        if center is None or radii is None or rotation is None:
            grains.append(GrainResult(fitok=False, fail_reason="fit_not_positive_definite", **base))
            continue

        rng = grain_rng(run_seed, src)
        cover = acover(pts, center, radii, rotation, rng=rng)
        grains.append(GrainResult(
            fitok=True, radii=np.asarray(radii), ellipsoid_center=np.asarray(center),
            rotation=np.asarray(rotation), acover=cover,
            aqualityok=bool(cover > a_quality_thresh), **base))
    return grains


def grain_size_distribution(grains: list[GrainResult]) -> np.ndarray:
    """b-axis diameters of grains passing `fitok & aqualityok` (matches grainsizedistribution.m).

    No `min_diam` cut is applied here -- MATLAB's raw GSD filters on fit quality only.
    """
    return np.array([g.b_axis_diameter for g in grains if g.fitok and g.aqualityok])


def percentiles(diameters: np.ndarray, ps=(16, 50, 84)) -> dict:
    """Grain-size percentiles Dxx from a diameter array (empty -> NaNs)."""
    if len(diameters) == 0:
        return {f"D{p}": float("nan") for p in ps}
    return {f"D{p}": float(np.percentile(diameters, p)) for p in ps}

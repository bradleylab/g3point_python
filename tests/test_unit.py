"""MATLAB-free unit tests for the G3Point port.

These lock in the correctness fixes (F1-F8, see PARITY.md) and the class contract using
synthetic inputs and the committed Otira example cloud -- NO MATLAB fixtures required, so they
run in CI on every push. The MATLAB stage-parity oracle lives in test_stage_parity.py and
auto-skips when its (out-of-band) fixtures are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"


# --------------------------------------------------------------------------------------------- #
# F1 -- check_stacks no longer enforces the false "min index == 0" invariant
# --------------------------------------------------------------------------------------------- #
def test_check_stacks_allows_dropped_point_zero():
    from g3point.tools import check_stacks
    # point 0 has been removed by a small/flat filter; the stacks are still coherent.
    stacks = [[1, 2], [3, 4]]
    assert check_stacks(stacks, 4) is True


def test_check_stacks_rejects_wrong_count():
    from g3point.tools import check_stacks
    with pytest.raises(ValueError):
        check_stacks([[0, 1], [2]], 4)  # covers 3 points, claims 4


def test_check_stacks_rejects_overlap():
    from g3point.tools import check_stacks
    with pytest.raises(ValueError):
        check_stacks([[0, 1, 2], [2, 3]], 4)  # point 2 appears in two grains


# --------------------------------------------------------------------------------------------- #
# get_sink_indexes -- highest-z point per grain, first on ties
# --------------------------------------------------------------------------------------------- #
def test_get_sink_indexes_picks_highest_z_first_on_tie():
    from g3point.cluster import get_sink_indexes
    xyz = np.array([[0, 0, 0.0], [0, 0, 5.0], [0, 0, 5.0], [0, 0, 1.0]])
    # grain 0 = points {0,1,2}: 1 and 2 tie at z=5 -> argmax returns the first (index 1)
    sinks = get_sink_indexes([[0, 1, 2], [3]], xyz)
    assert list(sinks) == [1, 3]


# --------------------------------------------------------------------------------------------- #
# merge_labels_dbscan -- the asymmetric merge matrix is traversed in the MATLAB (transpose)
# direction. This directed 3-node case gives a different partition with vs without the
# transpose, so it fails loudly if the .T is ever dropped.
# --------------------------------------------------------------------------------------------- #
def test_merge_labels_dbscan_asymmetric_transpose_direction():
    from g3point.cluster import merge_labels_dbscan
    labels = np.array([0, 1, 2])
    stacks = [[0], [1], [2]]
    # condition == True means "do NOT merge". Allow-merge (finite/0) only for the directed
    # edges 0->1 and 1->2; every reverse and 0<->2 pair is blocked.
    condition = np.ones((3, 3), dtype=bool)
    condition[0, 1] = False
    condition[1, 2] = False
    new_labels, new_stacks = merge_labels_dbscan(labels, stacks, condition)
    # In the transpose (MATLAB) direction this directed chain does NOT collapse to one cluster:
    # each node's traversal row isolates it -> three singleton grains.
    assert len(np.unique(new_labels)) == 3


# --------------------------------------------------------------------------------------------- #
# F2b -- the small-grain filter keeps grains with EXACTLY n_min points (>=, not strict >)
# --------------------------------------------------------------------------------------------- #
def test_small_grain_filter_keeps_exactly_n_min():
    from g3point.cluster import keep_labels
    n_min = 50
    nstack = np.array([n_min - 1, n_min, n_min + 1])  # below / at / above threshold
    # this is the exact rule clean_labels applies (clean_labels.m uses >=)
    condition = nstack >= n_min
    labels = np.array([0, 0, 1, 1, 2, 2])
    stacks = [[0, 1], [2, 3], [4, 5]]
    sink_indexes = np.array([0, 2, 4])
    new_labels, new_stacks, _ = keep_labels(labels, stacks, condition, sink_indexes)
    assert condition.tolist() == [False, True, True]      # the =n_min grain is kept
    assert len(new_stacks) == 2                           # grains 1 and 2 survive


# --------------------------------------------------------------------------------------------- #
# F2a / F3 -- ellipsoid rotation reorders by ROWS with the radii, and implicit_to_explicit does
# not mutate its input parameter vector.
# --------------------------------------------------------------------------------------------- #
def _orthonormal_rows(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q  # rows orthonormal


def test_ellipsoid_explicit_implicit_roundtrip_row_ordering():
    from g3point.ellipsoid import explicit_to_implicit, implicit_to_explicit
    center = np.array([1.0, -2.0, 0.5])
    radii = np.array([3.0, 2.0, 1.4])          # already sorted descending
    rotation = _orthonormal_rows(seed=3)        # rows = axis directions
    p = explicit_to_implicit(center, radii, rotation)
    p_before = p.copy()
    c2, r2, _q2, R2 = implicit_to_explicit(p)
    assert np.allclose(p, p_before)             # F3: input vector not mutated in place
    assert np.allclose(np.sort(r2)[::-1], radii, rtol=1e-6)
    assert np.allclose(c2, center, atol=1e-6)
    # each recovered row aligns (up to sign) with an input axis row -> rows moved with radii
    for row in rotation:
        cosines = np.abs(R2 @ row)
        assert np.max(cosines) > 1 - 1e-6


def test_fit_ellipsoid_recovers_known_axes():
    from g3point.ellipsoid import fit_ellipsoid_to_grain
    from g3point.quality import sample_ellipsoid_surface
    center = np.array([10.0, 20.0, 30.0])
    radii = np.array([3.0, 2.0, 1.6])           # c >= a/2 (direct-fit constraint)
    rotation = _orthonormal_rows(seed=7)
    pts = sample_ellipsoid_surface(center, radii, rotation, n=600,
                                   rng=np.random.default_rng(1))
    c, r, _q, _R, _p = fit_ellipsoid_to_grain(pts, method="direct")
    assert c is not None
    assert np.allclose(np.sort(r)[::-1], radii, rtol=0.05)


# --------------------------------------------------------------------------------------------- #
# F3 -- a failed / degenerate fit returns a typed failure, never crashes the tile
# --------------------------------------------------------------------------------------------- #
def test_fit_ellipsoid_zero_extent_raises_and_is_guarded_upstream():
    # The raw fit divides by the max extent, so a zero-extent grain makes it blow up (raise)
    # rather than return None -- which is exactly why compute_grains guards the extent BEFORE
    # calling it (see test_compute_grains_degenerate_is_failed_not_error).
    from g3point.ellipsoid import fit_ellipsoid_to_grain
    pts = np.tile([1.0, 2.0, 3.0], (10, 1))      # zero extent
    with pytest.raises((ValueError, FloatingPointError, np.linalg.LinAlgError)):
        with np.errstate(divide="raise", invalid="raise"):
            fit_ellipsoid_to_grain(pts, method="direct")


def test_compute_grains_degenerate_is_failed_not_error():
    from g3point.grains import compute_grains
    xyz = np.zeros((10, 3))                       # identical points -> zero extent
    grains = compute_grains(xyz, [np.arange(10)], np.arange(10))
    assert len(grains) == 1
    assert grains[0].fitok is False and grains[0].fail_reason == "degenerate_input"


def test_real_guard_drops_complex_keeps_noise():
    # A marginal quadric fit can return complex radii; _real drops a genuinely-complex fit but
    # keeps one whose imaginary part is only numerical noise (casting to the real part).
    from g3point.grains import _real
    assert _real(np.array([3.0, 2.0, 1.0])) is not None
    noise = np.array([3.0 + 1e-13j, 2.0 + 0j, 1.0 - 1e-13j])
    got = _real(noise)
    assert got is not None and np.allclose(got, [3.0, 2.0, 1.0]) and not np.iscomplexobj(got)
    assert _real(np.array([3.0 + 0.5j, 2.0, 1.0])) is None      # real imaginary component
    assert _real(np.array([np.inf, 2.0, 1.0])) is None          # non-finite


def test_compute_grains_too_few_points():
    from g3point.grains import compute_grains
    xyz = np.random.default_rng(0).random((3, 3))
    grains = compute_grains(xyz, [np.arange(3)], np.arange(3))
    assert grains[0].fitok is False and grains[0].fail_reason == "too_few_points"


def test_compute_grains_rejects_unsupported_fit_method():
    # a config typo must fail loudly, not silently empty the GSD via per-grain fit_error
    from g3point.grains import compute_grains
    xyz = np.random.default_rng(0).random((20, 3))
    with pytest.raises(ValueError):
        compute_grains(xyz, [np.arange(20)], np.arange(20), fit_method="bogus")


# --------------------------------------------------------------------------------------------- #
# E2 (scale) -- the catchment stack builder is iterative and byte-identical to the recursion it
# replaces, and no longer overflows the call stack on deep donor chains.
# --------------------------------------------------------------------------------------------- #
def _recursive_bw(index, delta, di_list, local_maximum, out):
    out.append(index)
    for k in range(delta[index], delta[index + 1]):
        if di_list[k] != local_maximum:
            _recursive_bw(di_list[k], delta, di_list, local_maximum, out)


def test_add_to_stack_bw_matches_recursion_order():
    from g3point.segment import add_to_stack_bw
    # tree rooted at local max 0: 1->0, 2->0, 3->1, 4->2, 5->2 (0 is its own receiver)
    delta = np.array([0, 3, 4, 6, 6, 6, 6])
    Di = np.array([0, 1, 2, 3, 4, 5])
    reference = []
    _recursive_bw(0, delta, Di, 0, reference)
    got = []
    add_to_stack_bw(0, delta, Di, got, 0)
    assert got == reference == [0, 1, 3, 2, 4, 5]


def test_add_to_stack_matches_recursion_order():
    from g3point.segment import add_to_stack
    n_donors = np.array([2, 1, 2, 0, 0, 0])
    donors = np.array([[1, 2], [3, 0], [4, 5], [0, 0], [0, 0], [0, 0]])
    got = []
    add_to_stack(0, n_donors, donors, got)
    assert got == [0, 1, 3, 2, 4, 5]


def test_segment_handles_coincident_points():
    # float32-quantised tiles can contain coincident points (zero-distance neighbours -> 0/0 slope).
    # Segmentation must stay coherent (every point in exactly one stack), not crash on the NaN.
    from scipy.spatial import KDTree
    from g3point.segment import segment_labels
    rng = np.random.default_rng(0)
    xyz = rng.random((60, 3))
    xyz[11] = xyz[10]                      # exact duplicate
    knn = 6
    _, nbr = KDTree(xyz).query(xyz, knn + 1)
    labels, stacks, _ndon, _lmax = segment_labels(xyz, knn, nbr[:, 1:])   # must not raise
    assert len(labels) == 60
    assert sorted(int(i) for s in stacks for i in s) == list(range(60))    # coherent partition


def test_add_to_stack_bw_deep_chain_no_recursion_error():
    from g3point.segment import add_to_stack_bw
    # a single catchment 6000 points deep -- a recursive builder would blow the ~1000 call-stack
    n = 6000
    delta = np.arange(n + 1)          # each node i has exactly one donor at Di[i] = i + 1
    Di = np.append(np.arange(1, n), 0)  # last node's donor is the root 0 -> skipped by the guard
    stack = []
    add_to_stack_bw(0, delta, Di, stack, 0)
    assert len(stack) == n and stack[:3] == [0, 1, 2]


# --------------------------------------------------------------------------------------------- #
# F4 -- the .laz loader reads scaled/offset metres (lowercase x/y/z), not raw integer records
# --------------------------------------------------------------------------------------------- #
def test_laz_loader_reads_scaled_coordinates(tmp_path):
    import laspy
    from g3point.tools import load_data
    coords = np.array([[100.12, 200.34, 300.56],
                       [101.00, 201.00, 301.00],
                       [102.99, 202.99, 302.99]])
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 300.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    path = tmp_path / "scaled.laz"
    las.write(path)
    xyz = load_data(str(path))
    assert np.allclose(xyz, coords, atol=1e-3)   # metres, not the raw ~integer record values
    assert np.all(xyz.max(axis=0) < 1000)        # raw X/Y/Z would be ~1e5


# --------------------------------------------------------------------------------------------- #
# F8 -- Acover is stochastic but seedable; the per-grain RNG is deterministic and reorder-stable
# --------------------------------------------------------------------------------------------- #
def test_acover_seeded_is_reproducible():
    from g3point.quality import acover
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    center, radii, rotation = np.zeros(3), np.array([2.0, 1.5, 1.2]), np.eye(3)
    pts = np.random.default_rng(0).standard_normal((80, 3)) * 0.5
    assert acover(pts, center, radii, rotation, rng=rng_a) == \
        acover(pts, center, radii, rotation, rng=rng_b)


def test_run_result_is_immutable():
    import dataclasses
    from g3point.grains import RunResult
    r = RunResult(grains=[], provenance={"merge_version": "matlab_dbscan"})
    assert r.provenance["merge_version"] == "matlab_dbscan"
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.grains = [1]                                   # frozen: cannot rebind fields


def test_grain_rng_depends_only_on_min_index_and_seed():
    from g3point.quality import grain_rng
    a = grain_rng(42, np.array([7, 3, 9])).random(5)     # min index 3
    b = grain_rng(42, np.array([9, 3, 7])).random(5)     # reordered, same min
    c = grain_rng(42, np.array([3])).random(5)
    assert np.allclose(a, b) and np.allclose(a, c)


# --------------------------------------------------------------------------------------------- #
# GSD filters on fitok & aqualityok only
# --------------------------------------------------------------------------------------------- #
def test_grain_size_distribution_filters_on_quality():
    from g3point.grains import GrainResult, grain_size_distribution, percentiles
    good = GrainResult(label=0, n_points=100, point_indexes=np.array([]),
                       source_point_indexes=np.array([0]), point_centroid=np.zeros(3),
                       fitok=True, radii=np.array([0.05, 0.03, 0.02]), aqualityok=True)
    poor = GrainResult(label=1, n_points=100, point_indexes=np.array([]),
                       source_point_indexes=np.array([1]), point_centroid=np.zeros(3),
                       fitok=True, radii=np.array([0.09, 0.07, 0.05]), aqualityok=False)
    failed = GrainResult(label=2, n_points=2, point_indexes=np.array([]),
                         source_point_indexes=np.array([2]), point_centroid=np.zeros(3),
                         fitok=False)
    gsd = grain_size_distribution([good, poor, failed])
    assert gsd.tolist() == [pytest.approx(0.06)]         # only the good grain (2 * b-axis 0.03)
    assert percentiles(np.array([]))["D50"] != percentiles(np.array([]))["D50"]  # NaN on empty


# --------------------------------------------------------------------------------------------- #
# Denoise SOR removes a planted isolated outlier and keeps the bulk
# --------------------------------------------------------------------------------------------- #
def test_sor_removes_isolated_outlier():
    from g3point.denoise import statistical_outlier_removal
    rng = np.random.default_rng(0)
    cloud = rng.standard_normal((500, 3)) * 0.1          # tight blob
    outlier = np.array([[50.0, 50.0, 50.0]])             # far away
    xyz = np.vstack([cloud, outlier])
    kept_xyz, kept_idx = statistical_outlier_removal(xyz, n_neighbors=4, std_ratio=3.0)
    assert 500 not in kept_idx                           # the outlier (last row) is removed
    assert len(kept_idx) >= 495                           # the bulk is retained


def test_sor_rejects_bad_parameters():
    from g3point.denoise import statistical_outlier_removal
    xyz = np.random.default_rng(0).random((50, 3))
    for bad in (dict(n_neighbors=0), dict(std_ratio=-1.0), dict(std_ratio=np.nan)):
        with pytest.raises(ValueError):
            statistical_outlier_removal(xyz, **bad)


# --------------------------------------------------------------------------------------------- #
# LAZ export preserves precision (P1-LAZ-EXPORT): a PLY source uses a 1 mm policy, not laspy's
# 1 cm default, so coordinates survive the round trip.
# --------------------------------------------------------------------------------------------- #
def test_laz_export_preserves_precision(tmp_path):
    from g3point.tools import load_data, save_data_with_colors
    cloud = str(tmp_path / "src.ply")                    # PLY source -> 1 mm precision policy
    xyz = np.array([[100.000, 200.000, 300.000],
                    [100.003, 200.007, 300.011],
                    [100.015, 200.024, 300.042]])
    out = save_data_with_colors(cloud, xyz, [[0, 1, 2]], np.array([0, 0, 0]), "_T")
    back = load_data(out)
    assert np.allclose(back, xyz, atol=6e-4)             # ~1 mm; laspy's 0.01 default would lose 1 cm


# --------------------------------------------------------------------------------------------- #
# Quaternion is consistent with the reordered rotation (P2-QUATERNION) and does not raise.
# --------------------------------------------------------------------------------------------- #
def test_quaternion_matches_reordered_rotation():
    from scipy.spatial.transform import Rotation
    from g3point.ellipsoid import explicit_to_implicit, implicit_to_explicit
    center, radii = np.zeros(3), np.array([3.0, 2.0, 1.4])
    p = explicit_to_implicit(center, radii, _orthonormal_rows(seed=5))
    _c, _r, quat, R = implicit_to_explicit(p, ignore_quaternions=False)
    assert quat is not None
    ref = np.real(R).astype(float)
    if np.linalg.det(ref) < 0:                           # the quaternion encodes the right-handed frame
        ref = ref.copy()
        ref[2] = -ref[2]
    assert np.allclose(Rotation.from_quat(quat).as_matrix(), ref, atol=1e-6)


def test_cli_json_emits_null_for_empty_percentiles():
    import json
    from g3point.grains import percentiles
    pct = percentiles(np.array([]))                      # all NaN
    pct_json = {k: (v if v == v else None) for k, v in pct.items()}
    doc = json.dumps({"percentiles_m": pct_json}, allow_nan=False)   # must not raise on NaN
    assert json.loads(doc)["percentiles_m"]["D50"] is None


# --------------------------------------------------------------------------------------------- #
# End-to-end smoke test on the committed Otira example (F2d frame separation + provenance).
# Needs open3d + the committed PLY; skips only if the example cloud is missing.
# --------------------------------------------------------------------------------------------- #
def _otira_ini(tmp_path) -> str:
    """Otira parameters, but with the port's unsupported/plot options disabled for a headless
    smoke run (minima resampling is not implemented; iplot/saveplot off)."""
    vals = dict(
        iplot=0, saveplot=0, denoise=1, decimate=0, minima=0, rot_detrend=1, clean=1,
        grid_by_number=0, save_granulo=0, save_grain=0, res=0.002, n_scale=4, min_scale=0.04,
        max_scale=2, knn=20, rad_factor=0.6, max_angle1=60, max_angle2=10, min_flatness=0.1,
        n_min=50, fit_method="direct", a_quality_thresh=10, min_diam=0.04, n_axis=2, dx_gbn=0)
    ini = tmp_path / "otira.ini"
    ini.write_text("[DEFAULT]\n" + "\n".join(f"{k} = {v}" for k, v in vals.items()) + "\n")
    return str(ini)


@pytest.mark.skipif(not (DATA / "Otira_1cm_grains.ply").exists(),
                    reason="Otira example cloud not present")
def test_otira_end_to_end_and_frame_separation(tmp_path):
    from g3point import G3Point
    g = G3Point(str(DATA / "Otira_1cm_grains.ply"), _otira_ini(tmp_path), remove_mins=True)
    result = g.run(version="matlab_dbscan", run_seed=42)
    grains = result.grains

    assert len(grains) > 0
    assert any(gr.fitok for gr in grains)
    # F2d: the analysis frame is NOT overwritten by the detrended segmentation copy.
    assert g.xyz_detrended is not g.xyz
    assert not np.allclose(g.xyz, g.xyz_detrended)
    # analysis frame is the min-shifted cloud (all coords >= 0); denoise only drops points, so
    # the minimum stays >= 0 even after the extreme point is removed.
    assert (g.xyz >= -1e-9).all()
    # source_indexes map every analysis-cloud row back to a loaded-file row.
    assert g.source_indexes.shape[0] == g.xyz.shape[0]
    # provenance records ACTUAL execution, not just configured values (identifies the result).
    prov = result.provenance
    assert prov["denoise_backend"] == "statistical_outlier_removal"
    assert prov["acover_run_seed"] == 42
    assert prov["merge_version"] == "matlab_dbscan"      # the mode that actually ran
    assert prov["denoise_applied"] is True               # ini sets denoise=1
    assert prov["clean_applied"] is True                 # ini sets clean=1
    assert prov["n_points_analysis"] == g.xyz.shape[0]
    assert prov["n_grains_in_gsd"] == len(g.grain_size_distribution())
    assert prov["parameters"]["knn"] == 20

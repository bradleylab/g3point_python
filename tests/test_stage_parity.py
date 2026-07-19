"""Stage-by-stage parity of the g3point_python port against the MATLAB oracle.

The MATLAB oracle (tests/matlab_oracle/g3_fixture_export.m) dumps every pipeline-stage
intermediate per tile. This harness feeds each MATLAB stage input into the corresponding
PORT stage and scores the port output against MATLAB -- so a stage's parity is measured in
isolation, not contaminated by an upstream divergence. Run it before/after a change and
watch the per-stage score move.

Isolation matters more than the end GSD: two wrong pipelines can still pass a D50 gate.

Fixtures embed field point-cloud coordinates and are NOT redistributed with the repo.
Point the harness at a local fixture directory (defaults to tests/fixtures_matlab):
    G3_FIXTURE_DIR=/path/to/fixtures_matlab python tests/test_stage_parity.py

MATLAB is 1-based; every index/label pulled from a fixture is converted to 0-based here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
from sklearn.metrics import adjusted_rand_score

# import the port under test (the fork clone that contains this tests/ dir)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from g3point import segment, cluster as cluster_stage, clean  # noqa: E402
from g3point.cluster import merge_labels_dbscan  # noqa: E402
from g3point.ellipsoid import fit_ellipsoid_to_grain  # noqa: E402
from g3point.quality import acover  # noqa: E402
from g3point.G3Point import SENSOR_HEIGHT  # noqa: E402

FIXTURE_DIR = Path(os.environ.get(
    "G3_FIXTURE_DIR",
    Path(__file__).resolve().parent / "fixtures_matlab"))

PARITY_SEED = 42  # seed the stochastic Acover sampler so ablations are comparable


def load_fixture(path: Path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def canonical_partition(labels: np.ndarray) -> np.ndarray:
    """Relabel a partition by first-appearance order; -1 (unlabelled) stays -1.

    Two partitions are IDENTICAL iff their canonical forms are array-equal -- an exact check,
    stronger than ARI>=0.9999 (which tolerates a swapped point and does not pin the unlabelled set).
    """
    labels = np.asarray(labels).ravel()
    out = np.full(labels.shape, -1, dtype=int)
    remap: dict[int, int] = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == -1:
            continue
        if lab not in remap:
            remap[lab] = len(remap)
        out[i] = remap[lab]
    return out


def partitions_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(canonical_partition(a), canonical_partition(b))


def params_from_meta(meta) -> SimpleNamespace:
    p = meta.param
    return SimpleNamespace(
        knn=int(p.nnptCloud),
        rad_factor=float(p.radfactor),
        max_angle1=float(p.maxangle1),
        max_angle2=float(p.maxangle2),
        min_flatness=float(p.minflatness),
        n_min=int(p.minnpoint),
        a_quality_thresh=float(p.Aquality_thresh),
    )


def stacks_from_labels(labels0: np.ndarray, nlabels: int) -> list[list[int]]:
    """Reconstruct per-label point-index stacks (0-based) from a label vector."""
    stacks: list[list[int]] = [[] for _ in range(nlabels)]
    for idx, lab in enumerate(labels0):
        if lab >= 0:
            stacks[lab].append(int(idx))
    return stacks


# --------------------------------------------------------------------------- #
# Stage 0a — denoise concordance (port SOR vs MATLAB pcdenoise inliers)
# --------------------------------------------------------------------------- #
def check_denoise(fx) -> dict:
    from g3point.denoise import denoise_concordance
    xyz = fx["xyz_loaded"].astype(float)
    inliers0 = fx["inlierIdx"].astype(int) - 1  # 1-based -> 0-based into loaded
    r = denoise_concordance(xyz, inliers0)
    r["stage"] = "denoise"
    return r


# --------------------------------------------------------------------------- #
# Stage 0 — normals (port Open3D pcnormals vs MATLAB pcnormals)
# --------------------------------------------------------------------------- #
def check_normals(fx) -> dict:
    import open3d as o3d
    from g3point.detrend import orient_normals
    xyz = fx["xyz_denoised"].astype(float)
    knn = int(fx["meta"].param.nnptCloud)
    n_ml = fx["normals"].astype(float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn))
    # Orient toward the sensor the PRODUCTION code uses: SENSOR_HEIGHT (10000) above the cloud
    # floor, not an ad-hoc 1000 -- otherwise the test could pass while the shipped orientation is
    # wrong.
    c = np.mean(xyz, axis=0)
    sensor = np.array([c[0], c[1], float(np.amin(xyz[:, 2])) + SENSOR_HEIGHT])
    n_py = orient_normals(xyz, np.asarray(pcd.normals), sensor)
    dots = np.clip(np.sum(n_py * n_ml, axis=1), -1, 1)   # SIGNED: a reversed normal must NOT pass
    ang = np.degrees(np.arccos(dots))
    return {"stage": "normals", "median_deg": float(np.median(ang)),
            "p99_deg": float(np.percentile(ang, 99)), "frac_gt5deg": float(np.mean(ang > 5)),
            "frac_flipped": float(np.mean(dots < 0))}


# --------------------------------------------------------------------------- #
# Stage 1 — initial segmentation (tests F5: sink test >= vs MATLAB >)
# --------------------------------------------------------------------------- #
def check_segmentation(fx) -> dict:
    xyz_detrended = fx["xyz_detrended"]
    neigh0 = fx["indNeighbors"].astype(int) - 1  # 1-based -> 0-based
    knn = int(fx["meta"].param.nnptCloud)
    labels_ml = fx["labels_seg"].astype(int) - 1  # every point labelled

    labels_py, stacks_py, ndon_py, locmax_py = segment(xyz_detrended, knn, neigh0)

    ari = adjusted_rand_score(labels_ml, labels_py)
    # ndon differs by convention (braun_willett self-counts at local maxima) without changing
    # the partition; report the exact-match flag but do not treat it as a failure.
    ndon_ml = fx["ndon"].astype(int)
    ndon_match = bool(np.array_equal(ndon_py.astype(int), ndon_ml))
    # NB: the fixture's `isink` is the POST-CLEAN sink array (the exporter overwrote it), so it
    # is NOT comparable to the port's segmentation seeds -- a sink metric here is meaningless.
    return {"stage": "segment", "ari": ari,
            "n_ml": int(fx["meta"].nlabels_seg), "n_py": len(stacks_py),
            "ndon_exact": ndon_match, "labels_py": labels_py, "labels_ml": labels_ml}


# --------------------------------------------------------------------------- #
# Stage 2A — DBSCAN merge primitive on the ACTUAL MATLAB Mmerge (both stages)
# --------------------------------------------------------------------------- #
def check_dbscan_matrix(fx) -> list[dict]:
    out = []
    for stage, dbg in (("cluster", fx["DBG_cluster"]), ("clean", fx["DBG_clean"])):
        Mmerge = np.asarray(dbg.Mmerge, dtype=float)
        idx_ml = np.asarray(dbg.dbscan_idx, dtype=int)
        # merge_labels_dbscan expects the DON'T-MERGE mask (MATLAB Mmerge==Inf): it sets
        # those cells to a large distance and merges the finite (==0) pairs under DBSCAN.
        cond = ~np.isfinite(Mmerge)  # Inf entries = "do not merge"
        n = Mmerge.shape[0]
        dummy_labels = np.arange(n)
        dummy_stacks = [[i] for i in range(n)]
        # direct
        lab_direct, _ = merge_labels_dbscan(dummy_labels, dummy_stacks, cond)
        # transpose
        lab_trans, _ = merge_labels_dbscan(dummy_labels, dummy_stacks, cond.T)
        out.append({"stage": f"dbscan[{stage}]",
                    "ari_direct": adjusted_rand_score(idx_ml, lab_direct),
                    "ari_transpose": adjusted_rand_score(idx_ml, lab_trans),
                    "n_ml": len(np.unique(idx_ml))})
    return out


# --------------------------------------------------------------------------- #
# Stage 2B/3B — full cluster / clean stage, MATLAB inputs -> port -> vs MATLAB out
# (tests the DBSCAN transpose end-to-end, plus F1 crash-guard and F2b n_min)
# --------------------------------------------------------------------------- #
def _labels_to_partition(labels_ml_raw) -> np.ndarray:
    """MATLAB label vector (1-based, NaN for removed) -> 0-based, -1 for removed."""
    v = np.asarray(labels_ml_raw, dtype=float)
    return np.where(np.isnan(v), -1, v - 1).astype(int)


def check_cluster_stage(fx, params) -> dict:
    # Feeds the port's own seg outputs (label numbering, ndon, sink seeds). This is NOT a
    # contamination: Stage 1 proves the port's segmentation partition is EXACTLY equal to MATLAB's
    # (test_segmentation_partition asserts canonical-partition equality, not just ARI), so
    # cluster(seg_port) == cluster(seg_matlab). The fixture's own segmentation-stage sink seeds are
    # unusable anyway (the exporter overwrote `isink` with the post-clean array). F2d: cluster/clean
    # run on the original (denoised) frame, not detrended.
    xyz = fx["xyz_denoised"]
    neigh0 = fx["indNeighbors"].astype(int) - 1
    surface = fx["surface"].astype(float)
    normals = fx["normals"].astype(float)

    labels0, stacks0, ndon0, sink0 = segment(fx["xyz_detrended"], params.knn, neigh0)

    new_labels, _stk, _snk = cluster_stage(
        xyz, params, neigh0, labels0, stacks0, ndon0, sink0, surface, normals,
        version="matlab_dbscan")
    ml = fx["labels_cl"].astype(int) - 1
    ari = adjusted_rand_score(ml, new_labels)
    return {"stage": "cluster", "ari": ari,
            "n_ml": int(fx["meta"].nlabels_cluster),
            "n_py": len(np.unique(new_labels)), "labels_py": new_labels, "labels_ml": ml}


def check_clean_stage(fx, params) -> dict:
    xyz = fx["xyz_denoised"]
    neigh0 = fx["indNeighbors"].astype(int) - 1
    labels_cl0 = fx["labels_cl"].astype(int) - 1      # cluster output, all points labelled
    nlab_cl = int(fx["meta"].nlabels_cluster)
    stacks = stacks_from_labels(labels_cl0, nlab_cl)
    ndon = fx["ndon"].astype(int)
    normals = fx["normals"].astype(float)

    try:
        labels_out, _stk, _snk = clean(
            xyz, params, neigh0, labels_cl0, stacks, ndon, normals,
            version="matlab_dbscan")
    except ValueError as e:
        return {"stage": "clean", "ari": float("nan"), "crash": str(e)[:40],
                "n_ml": int(fx["meta"].nlabels_clean), "n_py": -1}
    ml = _labels_to_partition(fx["labels_clean"])
    py = _labels_to_partition(labels_out + 1)
    ari = adjusted_rand_score(ml, py)
    return {"stage": "clean", "ari": ari, "crash": "",
            "n_ml": int(fx["meta"].nlabels_clean),
            "n_py": len(np.unique(labels_out[labels_out != -1])),
            "labels_py": py, "labels_ml": ml}


# --------------------------------------------------------------------------- #
# Stage 4 — ellipsoid geometry + Acover per clean grain (tests F2a/F2 rotation)
# --------------------------------------------------------------------------- #
def check_ellipsoids(fx) -> dict:
    xyz = fx["xyz_denoised"]                      # MATLAB fits on the original/denoised frame
    labels_clean = fx["labels_clean"]            # 1-based, NaN for removed
    nlab = int(fx["meta"].nlabels_clean)
    radii_ml = np.atleast_2d(fx["radii"])        # (3, nlab)
    R_ml = np.atleast_2d(fx["Rmats"])            # (9, nlab)
    acover_ml = np.atleast_1d(fx["acover"]).astype(float)
    fitok_ml = np.atleast_1d(fx["fitok"]).astype(bool)
    thresh = float(fx["meta"].param.Aquality_thresh)
    rng = np.random.default_rng(PARITY_SEED)

    radii_relerr, R_fro, acov_abs, aq_agree, n_fit, ml_fit_missed = [], [], [], [], 0, 0
    for k in range(nlab):
        stack = np.where(np.nan_to_num(labels_clean, nan=-1).astype(int) == (k + 1))[0]
        ml_is_fit = bool(fitok_ml[k]) if k < fitok_ml.size else False
        if stack.size < 4:
            ml_fit_missed += ml_is_fit
            continue
        center, radii, _q, Rm, _ep = fit_ellipsoid_to_grain(xyz[stack, :])
        if center is None:
            ml_fit_missed += ml_is_fit  # MATLAB fit this grain but the port failed -> a real gap
            continue
        n_fit += 1
        # radii: both sorted desc; relative error on b-axis (middle) — the GSD axis
        r_ml = np.sort(radii_ml[:, k])[::-1]
        r_py = np.sort(np.asarray(radii))[::-1]
        radii_relerr.append(abs(r_py[1] - r_ml[1]) / max(r_ml[1], 1e-9))
        # rotation: Frobenius distance up to column sign (compare |R| structure).
        # MATLAB saved R with column-major (:) flatten -> reshape order='F'.
        R_ml_k = R_ml[:, k].reshape(3, 3, order="F")
        R_fro.append(_rot_dist(np.asarray(Rm), R_ml_k))
        # Acover (stochastic): compare value + Aqualityok decision
        av = acover(xyz[stack, :], center, radii, Rm, n=200, rng=rng)
        acov_abs.append(abs(av - acover_ml[k]) if np.isfinite(acover_ml[k]) else np.nan)
        aq_agree.append((av > thresh) == (acover_ml[k] > thresh))
    return {"stage": "ellipsoid", "n_fit": n_fit, "n_ml": nlab, "ml_fit_missed": int(ml_fit_missed),
            "baxis_relerr_med": float(np.nanmedian(radii_relerr)) if radii_relerr else np.nan,
            "R_frobenius_med": float(np.nanmedian(R_fro)) if R_fro else np.nan,
            "acover_absdiff_med": float(np.nanmedian(acov_abs)) if acov_abs else np.nan,
            "aqualityok_agree": float(np.mean(aq_agree)) if aq_agree else np.nan}


# --------------------------------------------------------------------------- #
# Stage 5 — full class end-to-end: GSD vs MATLAB granulo (denoise included -> tolerance test)
# --------------------------------------------------------------------------- #
_INI_KEYS = [
    ("iplot", 0), ("saveplot", 0), ("grid_by_number", 0), ("save_granulo", 0),
    ("save_grain", 0), ("dx_gbn", 0),
]


def _synth_ini(param, path):
    p = param
    vals = dict(_INI_KEYS)
    vals.update(dict(
        denoise=int(p.denoise), decimate=int(p.decimate), minima=int(p.minima),
        rot_detrend=int(p.rotdetrend), clean=int(p.clean), res=float(p.res),
        n_scale=int(p.nscale), min_scale=float(p.minscale), max_scale=float(p.maxscale),
        knn=int(p.nnptCloud), rad_factor=float(p.radfactor), max_angle1=float(p.maxangle1),
        max_angle2=float(p.maxangle2), min_flatness=float(p.minflatness), n_min=int(p.minnpoint),
        fit_method=str(p.fitmethod), a_quality_thresh=float(p.Aquality_thresh),
        min_diam=float(p.mindiam), n_axis=int(p.naxis)))
    with open(path, "w") as fh:
        fh.write("[DEFAULT]\n" + "\n".join(f"{k} = {v}" for k, v in vals.items()) + "\n")


def check_end_to_end(fx, tmpdir) -> dict | None:
    import os
    from g3point import G3Point
    p = fx["meta"].param
    tile = os.path.join(str(p.ptCloudpathname), str(p.ptCloudname))
    if not os.path.exists(tile):
        return None  # tile PLY not available locally
    ini = os.path.join(tmpdir, "params.ini")
    _synth_ini(p, ini)
    g = G3Point(tile, ini, remove_mins=True)
    g.run(version="matlab_dbscan")
    gsd = g.grain_size_distribution()
    bml = np.atleast_2d(fx["granulo"].diameter)[1, :]  # MATLAB b-axis (median-diameter row)
    out = {"n_py": len(gsd), "n_ml": bml.size}
    for q in (16, 50, 84):
        out[f"d{q}"] = float(np.percentile(gsd, q) - np.percentile(bml, q))
    return out


def _rot_dist(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Orientation distance between two rotation matrices whose ROWS are the ellipsoid
    axis directions (R = evecs', per ellipsoid_im2ex.m). Invariant to axis order and to
    per-axis sign (eigenvectors have free sign): match each MATLAB row to the best-aligned
    port row and return the mean residual vector norm (0 = axes coincide)."""
    best = 0.0
    for j in range(3):
        row_b = Rb[j, :]
        dots = Ra @ row_b
        i = int(np.argmax(np.abs(dots)))
        row_a = Ra[i, :] * np.sign(dots[i] if dots[i] != 0 else 1.0)
        best += np.linalg.norm(row_a - row_b)
    return best / 3.0


def main():
    fixtures = sorted(FIXTURE_DIR.glob("*_fixture.mat"))
    if not fixtures:
        raise SystemExit(f"no fixtures in {FIXTURE_DIR}")
    print(f"fixtures: {FIXTURE_DIR}  ({len(fixtures)} tiles)\n")

    print("== Stage 0a: denoise concordance (SOR vs MATLAB pcdenoise inliers) ==")
    print(f"{'tile':<22} {'keptJ':>7} {'remJ':>6} {'SORrem':>7} {'MLrem':>7}")
    for f in fixtures:
        fx = load_fixture(f)
        r = check_denoise(fx)
        print(f"{fx['meta'].tile:<22} {r['jaccard']:>7.4f} {r['removed_jaccard']:>6.3f} "
              f"{r['sor_removed']:>7} {r['matlab_removed']:>7}")

    print("\n== Stage 0: normals (Open3D vs MATLAB pcnormals) ==")
    print(f"{'tile':<22} {'median°':>8} {'p99°':>7} {'>5°frac':>8}")
    for f in fixtures:
        fx = load_fixture(f)
        r = check_normals(fx)
        print(f"{fx['meta'].tile:<22} {r['median_deg']:>8.4f} {r['p99_deg']:>7.3f} {r['frac_gt5deg']:>8.4f}")

    print("\n== Stage 1: initial segmentation (partition; F5 no-op on float) ==")
    print(f"{'tile':<22} {'ARI':>7} {'n_ml':>5} {'n_py':>5} {'ndon=':>6}")
    for f in fixtures:
        fx = load_fixture(f)
        r = check_segmentation(fx)
        print(f"{fx['meta'].tile:<22} {r['ari']:>7.4f} {r['n_ml']:>5} {r['n_py']:>5} "
              f"{str(r['ndon_exact']):>6}")

    # merge_labels_dbscan now transposes Mmerge internally, so the PRODUCTION call (natural
    # condition) reproduces MATLAB (ARI=1.0); passing an already-transposed condition
    # double-transposes back to the wrong direction (the pre-fix behaviour), shown for contrast.
    print("\n== Stage 2A: DBSCAN merge on MATLAB Mmerge (production vs double-transpose) ==")
    print(f"{'tile':<22} {'stage':<14} {'ARIprod':>8} {'ARI2xT':>7} {'n_ml':>5}")
    for f in fixtures:
        fx = load_fixture(f)
        for r in check_dbscan_matrix(fx):
            print(f"{fx['meta'].tile:<22} {r['stage']:<14} {r['ari_direct']:>8.4f} "
                  f"{r['ari_transpose']:>7.4f} {r['n_ml']:>5}")

    print("\n== Stage 2B/3B: full cluster / clean stage (DBSCAN transpose, F1, F2b) ==")
    print(f"{'tile':<22} {'stage':<8} {'ARI':>7} {'n_ml':>5} {'n_py':>5}  {'note':<20}")
    for f in fixtures:
        fx = load_fixture(f)
        params = params_from_meta(fx["meta"])
        rc = check_cluster_stage(fx, params)
        print(f"{fx['meta'].tile:<22} {rc['stage']:<8} {rc['ari']:>7.4f} {rc['n_ml']:>5} {rc['n_py']:>5}")
        rk = check_clean_stage(fx, params)
        note = rk.get("crash", "")
        ari_s = f"{rk['ari']:>7.4f}" if rk["ari"] == rk["ari"] else "  CRASH"
        print(f"{fx['meta'].tile:<22} {rk['stage']:<8} {ari_s} {rk['n_ml']:>5} {rk['n_py']:>5}  {note:<20}")

    print("\n== Stage 4: ellipsoid geometry + Acover per clean grain (F2a) ==")
    print(f"{'tile':<22} {'nfit/nml':>9} {'bRELerr':>8} {'Rfrob':>7} {'AcovΔ':>7} {'AQok=':>6}")
    for f in fixtures:
        fx = load_fixture(f)
        r = check_ellipsoids(fx)
        print(f"{fx['meta'].tile:<22} {str(r['n_fit'])+'/'+str(r['n_ml']):>9} "
              f"{r['baxis_relerr_med']:>8.4f} {r['R_frobenius_med']:>7.4f} "
              f"{r['acover_absdiff_med']:>7.2f} {r['aqualityok_agree']:>6.3f}")

    print("\n== Stage 5: full class end-to-end GSD vs MATLAB granulo (denoise INCLUDED) ==")
    print(f"{'tile':<22} {'n_py':>5} {'n_ml':>5} {'dD16':>7} {'dD50':>7} {'dD84':>7}  (metres)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for f in fixtures:
            fx = load_fixture(f)
            r = check_end_to_end(fx, td)
            if r is None:
                print(f"{fx['meta'].tile:<22}   (tile PLY not available locally)")
                continue
            print(f"{fx['meta'].tile:<22} {r['n_py']:>5} {r['n_ml']:>5} "
                  f"{r['d16']:>+7.3f} {r['d50']:>+7.3f} {r['d84']:>+7.3f}")


# --------------------------------------------------------------------------- #
# pytest gates — the printed tables above are a diagnostic report; these ASSERT with
# predeclared tolerances so a numerical regression fails CI (not just prints a worse number).
# --------------------------------------------------------------------------- #
try:
    import pytest
except ImportError:  # allow the script to run standalone without pytest installed
    pytest = None

_FIXTURES = sorted(FIXTURE_DIR.glob("*_fixture.mat"))
_IDS = [f.stem.replace("_fixture", "") for f in _FIXTURES]
_ALLOW_MISSING_TILES = os.environ.get("G3_ALLOW_MISSING_TILES") == "1"

if pytest is not None:
    if not _FIXTURES:
        pytest.skip(f"no MATLAB fixtures in {FIXTURE_DIR}", allow_module_level=True)

    @pytest.fixture(params=_FIXTURES, ids=_IDS)
    def fx(request):
        return load_fixture(request.param)

    def test_normals_parity(fx):
        r = check_normals(fx)
        assert r["p99_deg"] < 0.1               # signed angle (a reversed normal would fail here)
        assert r["frac_flipped"] < 1e-3         # essentially no reversed normals vs MATLAB

    def test_segmentation_partition(fx):
        r = check_segmentation(fx)
        assert r["ari"] >= 0.9999
        assert partitions_equal(r["labels_py"], r["labels_ml"])   # EXACT, not just ARI

    def test_cluster_partition(fx):
        r = check_cluster_stage(fx, params_from_meta(fx["meta"]))
        assert r["ari"] >= 0.9999
        assert partitions_equal(r["labels_py"], r["labels_ml"])   # EXACT

    def test_clean_partition(fx):
        r = check_clean_stage(fx, params_from_meta(fx["meta"]))
        assert r["ari"] >= 0.9999
        assert partitions_equal(r["labels_py"], r["labels_ml"])   # EXACT (incl. the unlabelled set)

    def test_ellipsoid_geometry(fx):
        r = check_ellipsoids(fx)
        assert r["R_frobenius_med"] < 1e-4      # rotation matches MATLAB (F2a bug was 0.3-0.5)
        assert r["baxis_relerr_med"] < 1e-3     # b-axis radius matches
        assert r["ml_fit_missed"] == 0          # every grain MATLAB fit, the port also fits
        assert r["aqualityok_agree"] >= 0.9     # Aqualityok decision agrees (stochastic ~0.92-1.0)

    def test_grains_on_matlab_partition(fx):
        """Tight deterministic gate: the port's GSD on MATLAB's exact clean partition must
        reproduce MATLAB's granulo (isolates the grain/GSD code from the denoise divergence)."""
        from g3point import compute_grains, grain_size_distribution
        xyz = fx["xyz_denoised"].astype(float)
        lc = np.nan_to_num(fx["labels_clean"], nan=0).astype(int)
        nlab = int(fx["meta"].nlabels_clean)
        stacks = [np.where(lc == (k + 1))[0] for k in range(nlab)]
        gsd = grain_size_distribution(
            compute_grains(xyz, stacks, np.arange(len(xyz)), run_seed=PARITY_SEED))
        bml = np.atleast_2d(fx["granulo"].diameter)[1, :]
        # 5 mm tolerance: the geometry is deterministic to <1 mm, but the fitok&aqualityok
        # membership depends on the STOCHASTIC Acover test (port's seeded RNG vs MATLAB's), so a
        # threshold-straddling grain can flip and shift a percentile a few mm. Still far below
        # the tens-of-mm error the pre-fix pipeline produced.
        for q in (16, 50, 84):
            assert abs(np.percentile(gsd, q) - np.percentile(bml, q)) < 5e-3

    def test_run_is_repeatable(fx, tmp_path):
        """run() twice on the same object must give an identical GSD (SOR-idempotency guard)."""
        import os as _os
        from g3point import G3Point
        p = fx["meta"].param
        tile = _os.path.join(str(p.ptCloudpathname), str(p.ptCloudname))
        if not _os.path.exists(tile):
            if _ALLOW_MISSING_TILES:
                pytest.skip("tile PLY not available (G3_ALLOW_MISSING_TILES=1)")
            pytest.fail(f"tile PLY missing: {tile} (set G3_ALLOW_MISSING_TILES=1 to skip)")
        ini = str(tmp_path / "p.ini")
        _synth_ini(p, ini)
        g = G3Point(tile, ini, remove_mins=True)
        gsd1 = np.sort(g.run(version="matlab_dbscan") and g.grain_size_distribution())
        gsd2 = np.sort(g.run(version="matlab_dbscan") and g.grain_size_distribution())
        assert gsd1.shape == gsd2.shape and np.allclose(gsd1, gsd2)

    def test_degenerate_grain_does_not_abort():
        """A zero-extent grain becomes a failed result, not an exception."""
        from g3point import compute_grains
        xyz = np.zeros((10, 3))                       # all identical -> zero extent
        grains = compute_grains(xyz, [np.arange(10)], np.arange(10))
        assert len(grains) == 1 and grains[0].fitok is False
        assert grains[0].fail_reason == "degenerate_input"


if __name__ == "__main__":
    main()

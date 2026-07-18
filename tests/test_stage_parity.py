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
from g3point.cluster import merge_labels_dbscan, get_sink_indexes  # noqa: E402
from g3point.ellipsoid import fit_ellipsoid_to_grain  # noqa: E402
from g3point.quality import acover  # noqa: E402

FIXTURE_DIR = Path(os.environ.get(
    "G3_FIXTURE_DIR",
    Path(__file__).resolve().parent / "fixtures_matlab"))

PARITY_SEED = 42  # seed the stochastic Acover sampler so ablations are comparable


def load_fixture(path: Path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


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
# Stage 1 — initial segmentation (tests F5: sink test >= vs MATLAB >)
# --------------------------------------------------------------------------- #
def check_segmentation(fx) -> dict:
    xyz_detrended = fx["xyz_detrended"]
    neigh0 = fx["indNeighbors"].astype(int) - 1  # 1-based -> 0-based
    knn = int(fx["meta"].param.nnptCloud)
    labels_ml = fx["labels_seg"].astype(int) - 1  # every point labelled

    labels_py, stacks_py, ndon_py, locmax_py = segment(xyz_detrended, knn, neigh0)

    ari = adjusted_rand_score(labels_ml, labels_py)
    # ndon is base-agnostic (a count per point); compare exactly
    ndon_ml = fx["ndon"].astype(int)
    ndon_match = bool(np.array_equal(ndon_py.astype(int), ndon_ml))
    # sinks: MATLAB isink (per label) vs port local maxima seeds
    sink_ml = set((fx["isink"].astype(int) - 1).tolist())
    sink_py = set(np.asarray(locmax_py).astype(int).tolist())
    return {"stage": "segment", "ari": ari,
            "n_ml": int(fx["meta"].nlabels_seg), "n_py": len(stacks_py),
            "ndon_exact": ndon_match, "sink_jaccard": _jaccard(sink_ml, sink_py)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


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
    # The fixture's segmentation partition is bit-exact to the port's (Stage 1 ARI=1.0),
    # so feed the port's own mutually-consistent seg outputs (its label numbering, ndon,
    # and sink seeds) and test whether the CLUSTER stage reproduces MATLAB's cluster
    # partition. F2d: cluster/clean run on the original (denoised) frame, not detrended.
    xyz = fx["xyz_denoised"]
    neigh0 = fx["indNeighbors"].astype(int) - 1
    surface = fx["surface"].astype(float)
    normals = fx["normals"].astype(float)

    labels0, stacks0, ndon0, sink0 = segment(fx["xyz_detrended"], params.knn, neigh0)

    new_labels, _stk, _snk = cluster_stage(
        xyz, params, neigh0, labels0, stacks0, ndon0, sink0, surface, normals,
        version="matlab_dbscan")
    ari = adjusted_rand_score(fx["labels_cl"].astype(int) - 1, new_labels)
    return {"stage": "cluster", "ari": ari,
            "n_ml": int(fx["meta"].nlabels_cluster),
            "n_py": len(np.unique(new_labels))}


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
    ari = adjusted_rand_score(_labels_to_partition(fx["labels_clean"]),
                              _labels_to_partition(labels_out + 1))
    return {"stage": "clean", "ari": ari, "crash": "",
            "n_ml": int(fx["meta"].nlabels_clean),
            "n_py": len(np.unique(labels_out[labels_out != -1]))}


# --------------------------------------------------------------------------- #
# Stage 4 — ellipsoid geometry + Acover per clean grain (tests F2a/F2 rotation)
# --------------------------------------------------------------------------- #
def check_ellipsoids(fx) -> dict:
    xyz = fx["xyz_denoised"]                      # MATLAB fits on the original/denoised frame
    labels_clean = fx["labels_clean"]            # 1-based, NaN for removed
    nlab = int(fx["meta"].nlabels_clean)
    radii_ml = np.atleast_2d(fx["radii"])        # (3, nlab)
    R_ml = np.atleast_2d(fx["Rmats"])            # (9, nlab)
    fitok_ml = np.atleast_1d(fx["fitok"]).astype(bool)
    acover_ml = np.atleast_1d(fx["acover"]).astype(float)
    thresh = float(fx["meta"].param.Aquality_thresh)
    rng = np.random.default_rng(PARITY_SEED)

    radii_relerr, R_fro, acov_abs, aq_agree, n_fit = [], [], [], [], 0
    for k in range(nlab):
        stack = np.where(np.nan_to_num(labels_clean, nan=-1).astype(int) == (k + 1))[0]
        if stack.size < 4:
            continue
        center, radii, _q, Rm, _ep = fit_ellipsoid_to_grain(xyz[stack, :])
        if center is None:
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
    return {"stage": "ellipsoid", "n_fit": n_fit, "n_ml": nlab,
            "baxis_relerr_med": float(np.nanmedian(radii_relerr)) if radii_relerr else np.nan,
            "R_frobenius_med": float(np.nanmedian(R_fro)) if R_fro else np.nan,
            "acover_absdiff_med": float(np.nanmedian(acov_abs)) if acov_abs else np.nan,
            "aqualityok_agree": float(np.mean(aq_agree)) if aq_agree else np.nan}


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

    print("== Stage 1: initial segmentation (F5) ==")
    print(f"{'tile':<22} {'ARI':>7} {'n_ml':>5} {'n_py':>5} {'ndon=':>6} {'sinkJ':>6}")
    for f in fixtures:
        fx = load_fixture(f)
        r = check_segmentation(fx)
        print(f"{fx['meta'].tile:<22} {r['ari']:>7.4f} {r['n_ml']:>5} {r['n_py']:>5} "
              f"{str(r['ndon_exact']):>6} {r['sink_jaccard']:>6.3f}")

    print("\n== Stage 2A: DBSCAN merge on MATLAB Mmerge (direct vs transpose) ==")
    print(f"{'tile':<22} {'stage':<14} {'ARIdirect':>9} {'ARItrans':>9} {'n_ml':>5}")
    for f in fixtures:
        fx = load_fixture(f)
        for r in check_dbscan_matrix(fx):
            print(f"{fx['meta'].tile:<22} {r['stage']:<14} {r['ari_direct']:>9.4f} "
                  f"{r['ari_transpose']:>9.4f} {r['n_ml']:>5}")

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


if __name__ == "__main__":
    main()

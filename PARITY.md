# MATLAB parity — changes, equivalence, and known divergences

This fork is being made faithful to the reference MATLAB G3Point implementation, verified
stage-by-stage against a MATLAB fixture oracle (`tests/test_stage_parity.py`,
`tests/matlab_oracle/`). This file is the running record of:

1. **what changed** and why (with the oracle result that justifies it), and
2. **where Python legitimately gives a DIFFERENT answer than MATLAB** — by dependency or by
   design — so downstream results are interpreted correctly and never assumed identical.

Verification is on the fixture tiles (`station_040/041/043/047`); scores are the
adjusted Rand index (ARI) of the partition vs MATLAB, or a per-grain geometry residual.

---

## Verified equivalence (as of fork 483319c, 2026-07-17)

The deterministic core reproduces MATLAB **exactly** on every fixture tile:

| stage | metric vs MATLAB | result |
|-------|------------------|--------|
| initial segmentation | partition ARI | **1.0000** |
| cluster (merge)      | partition ARI | **1.0000** |
| clean (merge + small/flat filters) | partition ARI | **1.0000** |
| ellipsoid radii      | b-axis relative error | **1e-4** |
| ellipsoid rotation   | row-matched Frobenius residual | **0.0** |
| per-grain GSD (on MATLAB's partition) | D16/D50/D84 vs `granulo` | **sub-mm** |

These are the DETERMINISTIC core: fixture-level PARTITION equivalence (ARI=1.0) plus grain
geometry, with the denoise held identical. Stage internals that differ without changing the
partition (the `ndon` self-count convention; the F5 `>=`/`>` sink edge on quantized input) are
documented under "known divergences", not claimed identical.

The **full class end-to-end** (Stage 5) additionally runs the port's own SOR denoise, so it is
a TOLERANCE comparison, not bit-exact — see divergence #1 for the quantified result.

Run `python tests/test_stage_parity.py` (set `G3_FIXTURE_DIR`) to regenerate.

---

## Changes applied (each ablated against the oracle)

| id | file | change | why it was wrong | oracle delta |
|----|------|--------|------------------|--------------|
| F2a | `ellipsoid.implicit_to_explicit` | reorder rotation by **rows**, not columns, when sorting radii | rows are the axis directions (`R = evecs.T`, cf. `ellipsoid_im2ex.m`); a column reorder desynchronised R from the sorted radii and mis-oriented the fitted ellipsoid → poisoned the Acover fit-quality test → dropped good grains | R Frobenius 0.3–0.5 → **0** |
| transpose | `cluster.merge_labels_dbscan` | `DBSCAN.fit(Mmerge.T)` | MATLAB's `dbscan` traverses a precomputed matrix opposite to scikit-learn; on the **asymmetric** merge condition the two directions give different partitions (no-op on the symmetric clean matrix) | cluster ARI 0.86–0.98 → **1.0** |
| F2b | `cluster.clean_labels` | small-label filter `nstack >= n_min` (was strict `>`) | `clean_labels.m` uses `>=`; strict `>` dropped the finest grains sitting exactly at the threshold | clean ARI 0.958–0.986 → **1.0** |
| F1 | `tools.check_stacks` | drop the `min index == 0` invariant | not an invariant after small/flat grains are removed (point 0 can be dropped) → crashed a large fraction of real tiles with "stacks are not coherent" | clean stage ran instead of crashing |
| F3 | `G3Point.fit_ellipsoids`, `ellipsoid.implicit_to_explicit` | skip failed fits (row → NaN) instead of flattening a `None` rotation; copy `p` before halving it | unconditional `None.flatten()` crashed on any failed fit; in-place `p` halving corrupted the caller's array | no crash on failed fits |
| F6 | `cluster.compute_mean_angle` | `version == 'matlab' or version == 'matlab_dbscan'` | bare `'matlab_dbscan'` string is always truthy → the guard never keyed on the version | no regression (latent bug) |
| F7 | `detrend.vec2rot` | guard `s**2` division when vectors already parallel → identity | divide-by-zero when the fitted normal is already `[0,0,1]` | no regression |
| F4 | `tools.load_data` | read scaled `x/y/z` from `.laz`, not raw integer `X/Y/Z` | integer record values are pre-scale, not metres | (PLY path unaffected) |

---

## KNOWN DIVERGENCES — Python may give a different answer than MATLAB

These are the places to watch. Each is either an unavoidable dependency difference or a
deliberate, documented choice; none is silently hidden.

### 1. Denoising — `pcdenoise` (MATLAB) vs statistical outlier removal (Python)  ⚠ MATERIAL
MATLAB `pcdenoise` is **not** reproducible by standard statistical outlier removal. Sweeping
k and threshold (manual SOR and Open3D), the best achievable agreement with MATLAB's exact
inlier set is **min Jaccard ≈ 0.99** (k=4, t=2.5), and the removal *counts* stay inconsistent
across tiles (e.g. tile 041: SOR removes ~50 where MATLAB removes 11; tile 047: SOR removes
~910 where MATLAB removes ~1240). So MATLAB's denoiser is a different (likely iterative /
robust-spread) algorithm in the same family, not just different parameters.

- **Nature of the difference:** both flag points whose local neighbourhood is anomalously
  sparse. They **agree on ~99% of points** and disagree only on *borderline* points near the
  threshold — genuinely ambiguous "is this an outlier?" cases. Neither is objectively more
  correct; MATLAB's default is slightly more conservative on clean tiles.
- **DECISION (approved 2026-07-17):** the port uses a fixed, documented standard SOR
  (reproducible, not a proprietary black box) and the acceptance test is moved to the
  **final grain-size distribution** (Phase 4: D16/D50/D84 + KS within the cross-view
  tolerance), NOT bit-exact denoise. Rationale: denoise touches ~1–2% of points as outlier
  pre-filtering; most disputed points are absorbed into a grain or removed by the downstream
  small/flat filters anyway. This is an *empirical* claim to be verified, not assumed.
- **Retained- vs removed-set concordance.** Retained-set Jaccard (~0.99) is misleading — it
  is dominated by the many points both keep. The discriminating **removed-set Jaccard is
  0.21–0.84** across fixtures (e.g. 041: SOR removes 52, MATLAB 11 → 0.21). Both are reported
  by the harness (Stage 0a).
- **QUANTIFIED downstream effect (end-to-end vs MATLAB `granulo`, Stage 5).** With the refactor
  correct (feeding MATLAB's exact partition reproduces `granulo` to sub-mm), the SOR-vs-
  pcdenoise difference moves the GSD as follows: tiles 040/041/047 match to **≤9 mm on
  D16/D50/D84** (incl. the 218-grain tile 047: dD84 = +1 mm), BUT the smallest tile 043 (29
  grains) misses **dD84 = −93 mm** because the different denoise dropped one large grain.
  So the divergence is negligible at aggregate/reach scale but introduces **coarse-tail
  sensitivity on small samples** — cell-level D84 on sparse spatial cells can be affected.
  Aggregate D-tolerance alone is insufficient; grain counts, fit failures, and held-out tiles
  must be checked (the current k=4,t=2.5 was tuned on these fixtures).
- **How to report it:** METHODS must state the port uses SOR and is ~99% (retained) / 0.2–0.8
  (removed) concordant with MATLAB `pcdenoise` — it must **not** claim to replicate it.
- **Status:** implemented (`denoise.py`, wired into `G3Point.run`). Open: held-out validation
  and a decision on whether the small-sample coarse-tail sensitivity is acceptable for the
  spatial GSD mapping, or whether `pcdenoise` must be ported exactly.

### 2. Acover / Aqualityok fit-quality test — stochastic
`Acover` samples 200 random points on each fitted ellipsoid's surface, so the per-grain value
is **not deterministic** and will not match MATLAB's value exactly (MATLAB's is itself an
unseeded random draw). With the rotation fixed (F2a) the geometry is now exact, so the
*Aqualityok* pass/fail decision agrees with MATLAB on ~92–100% of grains; the residual
disagreement is grains sitting within RNG noise of the threshold. The port seeds the sampler
for run-to-run reproducibility. Impact on the GSD is limited to a few threshold-straddling
grains per tile.

### 3. Initial-segmentation sink test `>=` vs `>` (F5) — quantized input only
`segment_labels` uses `min_slope >= 0` where MATLAB uses `> 0`. On float coordinates this is
a **no-op** (verified: segmentation ARI = 1.0), because an exactly-zero slope is measure-zero.
It could differ on integer/quantized input. Left as-is (matches MATLAB on all real tiles);
revisit only if quantized clouds are processed.

### 4. Donor-count (`ndon`) convention — no observed effect
The port's `braun_willett` builder counts a self-donor at each local maximum; MATLAB's `ndon`
does not. This feeds the `ndon == 0` border test in the merge stages. It did **not** change
cluster/clean parity on the fixtures (ARI = 1.0), but the two conventions are not identical;
verify before relying on `ndon` in any new code path.

### 5. Normals (`pcnormals` MATLAB vs Open3D) — VERIFIED, effectively identical
The port's Open3D `estimate_normals` (KNN = `nnptCloud`) + `orient_normals` reproduces MATLAB
`pcnormals` + `adjustnormals3d` to **median 0.000°, p99 0.04°, max 0.06°** on three tiles;
on the largest tile a single point out of ~35k reached 12.7° (a degenerate/tie PCA
neighbourhood), with 0.0% of points above 5°. So the cluster/clean equivalence above is **not**
in fact conditional on fed-in normals — the port's own normals give the same partition.
Checked by `check_normals` in the stage-parity harness (Stage 0).

---

## G3Point class refactor (done 2026-07-18, post-codex review)
- **Frames (F2d):** the class no longer overwrites `self.xyz` with detrended coords. Neighbours,
  surface, normals, cluster, clean, and ellipsoid fit run on the analysis frame; a detrended
  COPY is used only for segmentation. Export is in the loaded/scan frame.
- **Canonical `run(version='matlab_dbscan')`** using the verified merge mode and honouring
  `params.clean` (the `cluster`/`clean` method defaults stay `cpp` for back-compat).
- **Typed per-grain table** (`grains.py`, `GrainResult`): separate `point_centroid` and fitted
  `ellipsoid_center` (MATLAB `centers` = the fitted centre, compare to that), source-point
  index mapping preserved through invalid-point removal + denoise, Optional fields for failed
  fits with verified `fail_reason` (`too_few_points` / `fit_not_positive_definite`).
- **GSD** filters on `fitok & aqualityok` ONLY — no `min_diam` cut (matches
  `grainsizedistribution.m`; `min_diam` belongs to the grid-by-number workflow).
- **Acover RNG** is a deterministic per-grain stream keyed on the grain's source-point indexes
  (stable under reordering), not one shared generator.
- **Config contract:** `fit_method` threaded through; enabled-but-unimplemented `decimate` /
  `minima` raise `NotImplementedError` instead of silently no-op'ing.

## Open items
- Held-out-tile validation of the denoise (params were tuned on the current fixtures), and the
  decision on the small-sample coarse-tail sensitivity (divergence #1) — port `pcdenoise`
  exactly, or accept SOR for reach-scale GSD.
- Packaging/tests/CI (Phase 2); scale work (Phase 3).

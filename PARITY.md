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
- **RESOLUTION (2026-07-18, operator).** The port uses standard statistical outlier removal
  (SOR) as a deliberate **open-source substitute** for the proprietary `pcdenoise`, keeping
  everything else identical. Neither denoiser is "more correct"; we do NOT try to reproduce
  `pcdenoise` bit-for-bit and we do NOT tune SOR to match MATLAB's output. `std_ratio` /
  `n_neighbors` are ordinary parameters a user sets in the `.ini` per run (defaults in
  `denoise.py`, single-sourced into `G3PointParameters`).
- **Default: `n_neighbors=4`, `std_ratio=3.0`.** Chosen on denoise merit, not to match MATLAB:
  at these values SOR removes ~1–1.5% of points — the tail whose mean neighbour distance sits
  farthest above the cloud-wide mean. `std_ratio=3.0`
  is deliberately conservative for the small `n_neighbors=4` statistic (a tighter 2.0–2.5 can
  clip legitimately-sparse grain edges). (An earlier note calling 2.5 "too aggressive / 6%
  removal" was wrong — that 6% was Open3D SOR at `std_ratio=1.0`, a different config.)
- **Resulting difference from MATLAB (informational, NOT a target).** Because SOR ≠ `pcdenoise`,
  the two remove different ~1–2% subsets (removed-set Jaccard 0.275/0.530/0.853/0.927 on the four
  fixtures) and the port's GSD runs
  slightly FINER. Held-out (40 random B4 tiles, excluding the calibration tiles; port end-to-end
  vs existing MATLAB `granulo`) at `std_ratio=3.0`: dD50 median **−0.8 mm** (IQR −2.6..0.0),
  dD84 median **−3.4 mm** (IQR −10.7..+1.6; 3/40 tiles >20 mm at D84 — small-sample coarse-tail
  cases). This is the expected difference between two legitimate denoisers, quantified and
  documented — not a defect. (Reproduce: `scripts/b4_denoise_holdout.py`,
  `scripts/b4_denoise_stdsweep.py`.)
- **Denoise is the only ALGORITHMIC substitution.** Every other G3Point parameter is used
  identically to MATLAB (same `.ini`), and on the deterministic stages (segment/cluster/clean/
  ellipsoid) the port matches MATLAB exactly (ARI = 1.0, verified). The other documented divergences
  below (stochastic Acover, the `ndon` self-count convention, the F5 quantized-input edge, the
  all-coincident-point policy) are separately noted — "identical everywhere except denoise" refers
  to the deterministic core, not a claim that no other difference of any kind exists.
- **How to report it:** METHODS states the port denoises with SOR (stating `n_neighbors` /
  `std_ratio`) as an open-source substitute for `pcdenoise` that does not reproduce it
  bit-for-bit; the two agree on ~99% of retained points and the port's GSD runs a few mm finer.
- **Status:** RESOLVED and implemented. Open (minor): document the SOR choice in METHODS.md when
  the port becomes the authoritative path.

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

### 6. Load order: invalid-point removal before min-shift — deliberate, safer divergence
MATLAB `loadptCloud` min-shifts then removes invalid points; the port removes non-finite points
FIRST, then min-shifts, so a stray `inf`/`NaN` coordinate cannot poison the per-axis minimum.
This only differs for clouds that contain non-finite coordinates (none in the fixtures). A
signed-normal orientation test across both `remove_mins` settings is still owed.

### 7. Coincident points in segmentation — deliberate, coherent-singleton policy
Float32-quantised tiles can contain coincident points (a zero-distance neighbour → an undefined
`0/0` slope). MATLAB's `min`/`max` ignore NaN, so a **mixed** neighbourhood (≥1 finite slope) is
handled the same way by the port, which replaces only the NaN slope with `+inf` (segmentation
ARI = 1.0 on the fixtures — none of which contain coincident points). For an **all-coincident**
neighbourhood the two differ: MATLAB's `min` returns NaN and `argmin` picks the first index (an
undefined artifact — no coherent sink), whereas the port yields a coherent singleton grain. This is
a deliberate robustness choice (a coherent partition on degenerate input), verified by a unit test;
it was the fix for a ~12% "stacks are not coherent" crash on the full reach run. It affects only
points whose entire k-NN is coincident with them — vanishingly rare in real data.

## G3Point class refactor (done 2026-07-18, post-codex review)
- **Frames (F2d):** the class no longer overwrites `self.xyz` with detrended coords. Neighbours,
  surface, normals, cluster, clean, and ellipsoid fit run on the analysis frame; a detrended
  COPY is used only for segmentation. Export is in the loaded/scan frame.
- **Canonical `run(version='matlab_dbscan')`** using the verified merge mode and honouring
  `params.clean` (the `cluster`/`clean` method defaults stay `cpp` for back-compat).
- **Typed per-grain table** (`grains.py`, `GrainResult`): separate `point_centroid` and fitted
  `ellipsoid_center` (MATLAB `centers` = the fitted centre, compare to that), source-point
  index mapping preserved through invalid-point removal + denoise, Optional fields for failed
  fits with verified `fail_reason` (`too_few_points` / `fit_not_positive_definite` / `fit_not_real`).
- **`fit_not_real` robustness guard.** A marginal quadric fit can return COMPLEX radii/rotation
  (sqrt of a slightly-negative/complex eigenvalue from the general eigen solve). Feeding a complex
  ellipsoid to `Acover` crashes the tile (`cKDTree` rejects complex input) — observed on ~40% of
  real field tiles. `compute_grains` now casts numerical-noise imaginary parts to real
  (`|imag| <= 1e-9 * scale`) and marks a genuinely-complex fit as a failed grain (`fit_not_real`),
  mirroring MATLAB's `fitok` filter. **Parity caveat:** if MATLAB instead keeps such grains (real
  part), the port drops a few degenerate fits MATLAB retains — these are near-degenerate ellipsoids
  that `Aqualityok` would usually reject anyway; the full-sweep `dpsi` quantifies whether it matters.
- **GSD** filters on `fitok & aqualityok` ONLY — no `min_diam` cut (matches
  `grainsizedistribution.m`; `min_diam` belongs to the grid-by-number workflow).
- **Acover RNG** is a deterministic per-grain stream keyed on the grain's source-point indexes
  (stable under reordering), not one shared generator.
- **Config contract:** `fit_method` threaded through; enabled-but-unimplemented `decimate` /
  `minima` raise `NotImplementedError` instead of silently no-op'ing.

## Packaging & CI (Phase 2, done 2026-07-18)
The port is now a pip-installable package (`pyproject.toml`) with a `g3point` command-line
entry point (`g3point/cli.py`) and a MATLAB-free unit suite (`tests/test_unit.py`, 19 tests)
that locks in every correctness fix on synthetic inputs + the committed Otira example cloud, so
CI has real regression coverage without the out-of-band MATLAB fixtures. `.github/workflows/
ci.yml` runs ruff + pytest on Python 3.10-3.12; the MATLAB stage-parity oracle auto-skips when
`G3_FIXTURE_DIR` is unset. None of this changes any numeric result.

## Scale / performance (Phase 3)
- **E2 — catchment stack builder made iterative (DONE, numerically identical).** `segment.py`
  `add_to_stack` / `add_to_stack_bw` were recursive (one Python frame per catchment point) and
  raised `RecursionError` on large tiles (a deep donor chain exceeds the ~1000 call-stack
  limit). Rewritten as an explicit-stack pre-order DFS that pushes donors in reverse, so the
  traversal order is **byte-identical** to the recursion (verified by a reference-recursion
  unit test) — the segmentation partition stays ARI = 1.0 on every fixture. A pure refactor, not
  a method change.
- **E4 — neighbour search reuse: already satisfied.** The KDTree neighbour query runs once in
  `initial_segmentation`; `cluster` / `clean` reuse `neighbors_indexes`. No change needed.
- **E1 — dense `nlabels²` merge matrices → sparse: NOT done (deferred, with reason).** `cluster`
  builds `D1`/`Dist`/`Nneigh`/`Aangle`/`Mmerge` as dense `nlabels²` arrays (peak ≈ several GB
  only on multi-million-point tiles; ~5 MB on the fixtures). A sparse candidate-pair rewrite is
  parity-critical (it must reproduce the asymmetric-DBSCAN transpose partition exactly) yet is
  verifiable only on the four small fixtures, so shipping it blind risks the silent divergence
  this whole effort guards against. Deferred until the port is on the critical path AND a
  large-tile stage fixture exists as an added guard.
- **E3 — batched border-angle SVD: NOT done (low priority).** `clean_labels` fits one small SVD
  per grain in a loop; a CPU micro-optimisation, not a memory wall.

## Open items
- Denoise: RESOLVED — open-source SOR (`std_ratio=3.0`), documented as a non-bit-reproducing
  substitute for `pcdenoise` (divergence #1). Remaining: state it in METHODS.md when the port
  becomes authoritative.
- Scale: E1 (sparse merge matrices) + E3 (batched SVD) remain — see "Scale / performance" above.

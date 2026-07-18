# Stage-parity tests against the MATLAB G3Point oracle

`test_stage_parity.py` feeds each MATLAB pipeline-stage intermediate into the corresponding
Python stage and scores the port output against MATLAB, stage by stage. This isolates each
stage so a fix can be validated where it acts, rather than only on the end grain-size
distribution (two divergent pipelines can still pass a downstream D50 gate).

## Running

The fixtures embed field point-cloud coordinates and are **not redistributed with this
repo**. Point the harness at a local fixture directory:

```bash
G3_FIXTURE_DIR=/path/to/fixtures_matlab python tests/test_stage_parity.py
```

Each `*_fixture.mat` holds every stage intermediate for one tile: loaded/denoised/detrended
coordinates, kNN indices + distances, normals, per-stage labels, the cluster/clean merge
matrices (`DBG_cluster` / `DBG_clean`), and per-grain ellipsoid center/radii/rotation/Acover.

## Regenerating fixtures (`matlab_oracle/`)

`matlab_oracle/g3_fixture_export.m` runs the byte-faithful MATLAB G3Point pipeline and dumps
the intermediates. It uses the two instrumented copies `cluster_labels_fix.m` /
`clean_labels_fix.m` (identical algorithm to the production `Utils/cluster_labels.m` /
`clean_labels.m`, plus a `DBG` struct exposing the internal merge matrices and dbscan
indices). Run from the MATLAB G3Point working dir with `Utils/` on the path:

```
G3_FIX_TILEDIR=<dir of tile .ply>  G3_FIX_LIST=<tile list>  G3_FIX_OUT=<out dir> \
    matlab -batch "addpath('<this>/matlab_oracle'); g3_fixture_export"
```

## What the harness checks

| stage | what it verifies |
|-------|------------------|
| segment  | initial-segmentation partition (ARI vs MATLAB `labels_seg`) |
| dbscan   | the merge primitive on the actual MATLAB `Mmerge` (direct vs transposed) |
| cluster  | full cluster stage: port output vs MATLAB `labels_cl` |
| clean    | full clean stage incl. small/flat filters: vs MATLAB `labels_clean` |
| ellipsoid| per-grain radii, rotation (row-matched), and Acover / Aqualityok vs MATLAB |

# G3Point

Granulometry from 3D Point clouds
> Based on Steer, Guerit et al. (2022)

![ForGitHub](https://user-images.githubusercontent.com/17555304/159018713-7272a95e-6400-4490-83f5-868248cffbcb.gif)

## Introduction

**G3Point** is a tool which aims at automatically measuring the size, shape, and orientation of a large 
number of individual grains as detected from any type of 3D point clouds describing the topography of surfaces covered by sediments.
The tool has been developped initially in *Matlab* https://github.com/philippesteer/G3Point

This repository aims at converting the tool in *Python* at first and also to try to improve it it a longer term.

This algorithm relies on 3 main phases:
1. Grain **segmentation** using a waterhsed algorithm
2. Grain **merging and cleaning**
3. Grain **fitting by geometrical models** including ellipsoids and cuboids

## Install

```
pip install -e .            # from a clone; add ".[test]" for the test/lint extras
```

This installs the `g3point` package and a `g3point` command-line entry point.

## Command line

Run the full pipeline (denoise → segment → cluster → clean → per-grain ellipsoid fit) on a
point cloud and print the b-axis grain-size distribution:

```
g3point CLOUD.ply CONFIG.ini                 # prints grain count + D16/D50/D84
g3point CLOUD.ply CONFIG.ini --json          # machine-readable result + run provenance
g3point CLOUD.ply CONFIG.ini --save          # also write labelled _G3POINT.laz outputs
```

The `matlab_dbscan` merge mode is the default and is the path verified against the MATLAB
reference (see `PARITY.md`). An example cloud + config lives in `data/`.

## Tests

```
pytest                                       # MATLAB-free unit suite (runs anywhere)
```

The `tests/test_unit.py` suite validates the correctness fixes and class contract on synthetic
inputs and the committed example cloud. `tests/test_stage_parity.py` compares each pipeline
stage against a MATLAB fixture oracle; those fixtures embed field data and are provided
out-of-band, so that module auto-skips unless `G3_FIXTURE_DIR` points at a local fixture set.

## HOWTO (library)

First you have to import the ```g3point``` module.  
**Note:** it's up to you to configure correctly the python path for your 
system to be able to find it.

```
import g3point
```

To instantiate a G3Point object, you will need a point cloud, in las or ply and an ini file (see the example in the 
```data``` 
section).

```python
import g3point
g = g3point.G3Point(cloud, ini)                 # cloud = .ply/.laz, ini = parameter file
grains = g.run(version="matlab_dbscan")          # denoise -> segment -> cluster -> clean -> fit
gsd = g.grain_size_distribution()                 # b-axis diameters (m), fitok & aqualityok grains
out, out_sinks = g.save()                         # optional: write labelled point clouds
```

`run()` is the verified MATLAB-parity path (`version="matlab_dbscan"`). It runs `denoise()`
(if the config enables it), `initial_segmentation()`, `cluster()`, `clean()`, and the per-grain
ellipsoid fit, in that order.

### Driving the stages manually

You can call the stages yourself, but note two defaults: `cluster()` and `clean()` default to
`version="cpp"` (a legacy mode that does **not** reproduce MATLAB) — pass `version="matlab_dbscan"`
for the verified path — and call `denoise()` first if your `.ini` sets `denoise = 1`.

```python
g.denoise()
g.initial_segmentation()
g.cluster(version="matlab_dbscan")
g.clean(version="matlab_dbscan")
grains = g.compute_grains()
```

### Ellipsoid fitting

Once you have a group of points, it is possible to fit an ellipsoid to this group.
This is not done for the g3point_data object as a whole at the current time.

```center, radii, quaternions, rotation_matrix, ellipsoid_parameters = g3point.fit_ellipsoid_to_grain(xyz)```
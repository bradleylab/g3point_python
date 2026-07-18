import os
import colorsys
import random

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from .cluster import clean_labels, cluster
from .denoise import statistical_outlier_removal
from .detrend import orient_normals, rotate_point_cloud_plane
from .G3PointParameters import G3PointParameters
from .grains import compute_grains, grain_size_distribution
from .segment import segment_labels
from .tools import load_data, save_data_with_colors

# Sensor height used to orient normals toward the scanner (matches MATLAB adjustnormals3d).
SENSOR_HEIGHT = 10000.0
UNLABELLED = -1


def generate_distinct_colors(n):
    hues = [i / n for i in range(n)]
    random.shuffle(hues)  # Optional: shuffle so colors aren't in rainbow order
    rgb = np.empty((n, 3))
    for k, hue in enumerate(hues):
        rgb[k, :] = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return rgb


class G3Point:
    """G3Point grain segmentation.

    Coordinate frames are kept separate (this matters for MATLAB parity): all geometry
    (neighbours, surface, normals, cluster, clean, ellipsoid fit) runs on the ANALYSIS frame
    ``self.xyz`` (the loaded cloud, invalid points removed, denoised, and optionally
    min-shifted). A detrended COPY ``self.xyz_detrended`` is used only as the coordinate
    argument to the initial segmentation. ``self.source_indexes`` maps analysis-cloud rows back
    to rows in the loaded file, surviving invalid-point removal and denoising.
    """

    def __init__(self, cloud, ini, remove_mins=True):
        if not os.path.exists(cloud):
            raise FileNotFoundError(cloud)
        if not os.path.exists(ini):
            raise FileNotFoundError(ini)

        self.cloud = cloud
        self.ini = ini
        self.params = G3PointParameters(self.ini)

        # Reject enabled-but-unimplemented settings explicitly rather than silently no-op'ing
        # them (a silent no-op would be a silent method change vs the MATLAB pipeline).
        if getattr(self.params, "decimate", 0):
            raise NotImplementedError(
                "decimate is enabled in the .ini but not implemented in the port; "
                "disable it or implement it in MATLAB order.")
        if getattr(self.params, "minima", 0):
            raise NotImplementedError(
                "minima resampling is enabled in the .ini but not implemented in the port; "
                "disable it.")

        # Load (scaled coordinates) and remove invalid (non-finite) points, tracking the map
        # back to rows in the loaded file.
        xyz = load_data(self.cloud)
        valid = np.all(np.isfinite(xyz), axis=1)
        self.source_indexes = np.where(valid)[0]
        xyz = xyz[valid]

        # Min-shift for numerical conditioning (recorded so the scan frame is recoverable).
        self.remove_mins = remove_mins
        if self.remove_mins:
            print("WARNING original data shifted (minimums removed)")
            self.mins = np.amin(xyz, axis=0)
            xyz = xyz - self.mins
        else:
            self.mins = np.zeros(3)
        self.xyz = xyz  # analysis frame; NEVER overwritten with detrended coords

        # Set during initial_segmentation
        self.xyz_detrended = None
        self.neighbors_indexes = None
        self.surface = None
        self.normals = None
        self.initial_labels = None
        self.initial_stacks = None
        self.ndon = None
        self.initial_sink_indexes = None

        # Modified during clustering / cleaning
        self.labels = None
        self.stacks = None
        self.sink_indexes = None

        # Grain fitting
        self.grains = None
        self.g3point_results = None

    # --- pipeline ---------------------------------------------------------------------------
    def denoise(self):
        """Statistical-outlier-removal denoise (gated by params.denoise), keeping the source map.

        NOTE: this is a documented approximation of MATLAB `pcdenoise`, not a bit-exact replica
        (see PARITY.md divergence #1).
        """
        if not getattr(self.params, "denoise", 0):
            return
        kept_xyz, kept = statistical_outlier_removal(self.xyz)
        self.xyz = kept_xyz
        self.source_indexes = self.source_indexes[kept]  # preserve the map to the loaded file

    def initial_segmentation(self):
        # Neighbours / surface / normals on the ANALYSIS frame (MATLAB uses ptCloud.Location).
        tree = KDTree(self.xyz)
        neighbors_distances, neighbors_indexes = tree.query(self.xyz, self.params.knn + 1)
        neighbors_distances = neighbors_distances[:, 1:]  # drop self
        neighbors_indexes = neighbors_indexes[:, 1:]
        self.neighbors_indexes = neighbors_indexes
        self.surface = np.pi * np.amin(neighbors_distances, axis=1) ** 2

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(self.params.knn))
        centroid = np.mean(self.xyz, axis=0)
        sensor_center = np.array([centroid[0], centroid[1], SENSOR_HEIGHT])
        self.normals = orient_normals(self.xyz, np.asarray(pcd.normals), sensor_center)

        # Detrended COPY, used only as the coordinate argument to segmentation.
        if self.params.rot_detrend:
            print("WARNING a detrended copy is used for segmentation only")
            xyz_rot = rotate_point_cloud_plane(self.xyz, np.array([0, 0, 1]))
            x, y, z = np.split(xyz_rot, 3, axis=1)
            coefficients, _residuals, _rank, _s = np.linalg.lstsq(
                np.c_[np.ones(x.shape), x, y, x ** 2, x * y, y ** 2], z, rcond=None)
            a0, a1, a2, a3, a4, a5 = coefficients
            z_detrended = z - (a0 + a1 * x + a2 * y + a3 * x ** 2 + a4 * x * y + a5 * y ** 2)
            self.xyz_detrended = np.c_[x, y, z_detrended]
        else:
            self.xyz_detrended = self.xyz.copy()

        # Segment on the detrended coordinates, indexing the original-frame neighbours.
        res = segment_labels(self.xyz_detrended, self.params.knn, self.neighbors_indexes)
        self.initial_labels, self.initial_stacks, self.ndon, self.initial_sink_indexes = res

        self.labels = np.copy(self.initial_labels)
        self.stacks = self.initial_stacks.copy()
        self.sink_indexes = np.copy(self.initial_sink_indexes)

    def cluster(self, version="cpp", condition_flag=None):
        """version: 'matlab_dbscan' (verified MATLAB-parity path) | 'matlab' | 'cpp' | 'custom'."""
        res = cluster(self.xyz, self.params, self.neighbors_indexes, self.initial_labels,
                      self.initial_stacks, self.ndon, self.initial_sink_indexes, self.surface,
                      self.normals, version=version, condition_flag=condition_flag)
        self.labels, self.stacks, self.sink_indexes = res

    def clean(self, version="cpp", condition_flag=None):
        res = clean_labels(self.xyz, self.params, self.neighbors_indexes, self.labels,
                           self.stacks, self.ndon, self.normals,
                           version=version, condition_flag=condition_flag)
        self.labels, self.stacks, self.sink_indexes = res

    def run(self, version="matlab_dbscan", run_seed=42):
        """Canonical MATLAB-compatible workflow: denoise -> segment -> cluster -> clean -> grains.

        Uses the verified `matlab_dbscan` merge mode (not the `cpp` method default) and honours
        `params.clean`. Returns the per-grain result table.
        """
        self.denoise()
        self.initial_segmentation()
        self.cluster(version=version)
        if self.params.clean:
            self.clean(version=version)
        return self.compute_grains(run_seed=run_seed)

    # --- grains -----------------------------------------------------------------------------
    def compute_grains(self, run_seed=42):
        """Fit ellipsoids + Acover to every grain -> typed per-grain table (self.grains)."""
        self.grains = compute_grains(
            self.xyz, self.stacks, self.source_indexes,
            fit_method=self.params.fit_method, a_quality_thresh=self.params.a_quality_thresh,
            run_seed=run_seed)
        return self.grains

    def grain_size_distribution(self):
        """b-axis diameters of grains passing fitok & aqualityok (mirrors MATLAB raw GSD)."""
        if self.grains is None:
            self.compute_grains()
        return grain_size_distribution(self.grains)

    def fit_ellipsoid(self, label):
        """Fit an ellipsoid to a single grain's points (analysis frame). Convenience wrapper."""
        from .ellipsoid import fit_ellipsoid_to_grain
        xyz_grain = self.xyz[self.stacks[label], :]
        return fit_ellipsoid_to_grain(xyz_grain, method=self.params.fit_method)

    def fit_ellipsoids(self):
        """DEPRECATED: legacy (n, 15) array [fitted-center | radii | R.flatten()].

        Kept for back-compat; prefer `compute_grains()` / `grain_size_distribution()`. Rows for
        failed fits stay NaN. The centre stored here is the fitted ellipsoid centre (as before),
        NOT the arithmetic point centroid.
        """
        if self.grains is None:
            self.compute_grains()
        self.g3point_results = np.full((len(self.stacks), 3 + 3 + 9), np.nan)
        for g in self.grains:
            if not g.fitok:
                continue
            self.g3point_results[g.label, 0:3] = g.ellipsoid_center
            self.g3point_results[g.label, 3:6] = g.radii
            self.g3point_results[g.label, 6:15] = g.rotation.flatten()
        return self.g3point_results

    # --- output -----------------------------------------------------------------------------
    def save(self):
        """Export the LABELLED points (grains) in the loaded/scan frame with per-grain colours.

        Unlabelled points (label == -1) are excluded rather than indexed into the colour array
        (which would wrap to a huge uint32 label).
        """
        labelled = self.labels != UNLABELLED
        xyz_out = self.xyz[labelled] + self.mins
        labels_out = self.labels[labelled]
        g3point = save_data_with_colors(self.cloud, xyz_out, self.stacks, labels_out, "_G3POINT")
        g3point_sinks = save_data_with_colors(
            self.cloud, self.xyz[self.sink_indexes, :] + self.mins,
            self.stacks, np.arange(len(self.stacks)), "_G3POINT_SINKS")
        return g3point, g3point_sinks

    def get_pcd_and_pcd_sinks(self, other_colors=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        rng = np.random.default_rng(42)
        if other_colors:
            colors = generate_distinct_colors(len(self.stacks))[self.labels, :]
        else:
            colors = rng.random((len(self.stacks), 3))[self.labels, :]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_sinks = o3d.geometry.PointCloud()
        pcd_sinks.points = o3d.utility.Vector3dVector(self.xyz[self.sink_indexes, :])
        pcd_sinks.paint_uniform_color(np.array([1., 0., 0.]))
        return pcd, pcd_sinks

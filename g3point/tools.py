import os
import random
import colorsys

import laspy
import numpy as np
import open3d as o3d


def get_random_colors(number_of_colors, version=None):

    if version == 'hsv':
        rgb = np.zeros((number_of_colors, 3))
        random.seed(42)
        for k in range(number_of_colors):
            hue, saturation, lightness = random.random(), 0.8 + random.random() / 5.0, 0.5 + random.random() / 5.0
            r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(hue, lightness, saturation)]
            rgb[k, :] = r, g, b

    elif version == 'cc':
        rng = np.random.default_rng(42)
        rg = rng.random((number_of_colors, 2)) * 255
        b = 255 - (rg[:, 0] + rg[:, 1]) / 2
        rgb = np.c_[rg, b]

    else:
        rng = np.random.default_rng(42)
        rgb = rng.random((number_of_colors, 3)) * (2**16 - 1)

    return rgb


def _output_header(cloud, xyz):
    """LAS header for the output cloud, with an explicit precision + CRS policy.

    laspy's default scales (0.01) quantise coordinates to 1 cm and carry no CRS, silently degrading
    georeferenced output. Inherit scales/offsets/CRS from a LAS/LAZ source; for a PLY source (no
    header) use a 1 mm scale with data-min offsets so coordinates survive the LAS integer encoding.
    """
    header = laspy.LasHeader(point_format=7, version="1.4")
    if os.path.splitext(cloud)[-1].lower() in ('.las', '.laz'):
        with laspy.open(cloud) as reader:
            src = reader.header
        header.scales = src.scales
        header.offsets = src.offsets
        try:  # preserve the coordinate reference system if the source carries one
            crs = src.parse_crs()
            if crs is not None:
                header.add_crs(crs)
        except Exception:  # pragma: no cover - CRS copy is best-effort, never fatal to a save
            pass
    else:
        header.scales = np.array([0.001, 0.001, 0.001])   # 1 mm
        header.offsets = np.min(xyz, axis=0)
    return header


def save_data_with_colors(cloud, xyz, stacks, labels, tag):
    head, tail = os.path.split(cloud)
    root, ext = os.path.splitext(tail)
    filename = os.path.join(head, root + tag + '.laz')

    x, y, z = np.split(xyz, 3, axis=1)

    # header carries an explicit precision + CRS policy (see _output_header)
    header = _output_header(cloud, xyz)
    las = laspy.LasData(header)

    las.x = np.squeeze(x)
    las.y = np.squeeze(y)
    las.z = np.squeeze(z)

    # set random colors
    # rng = np.random.default_rng(42)
    # rgb = rng.random((len(stacks), 3))[labels, :] * 255
    rgb = get_random_colors(len(stacks))[labels, :]
    las.red = rgb[:, 0]
    las.green = rgb[:, 1]
    las.blue = rgb[:, 2]

    las.add_extra_dim(laspy.ExtraBytesParams(
        name="g3point_label",
        type=np.uint32
    ))

    las.g3point_label = labels

    print(f"save {filename}")
    las.write(filename)

    return filename


def load_data(file, dtype=None):
    ext = os.path.splitext(file)[-1]
    if ext == '.ply':
        pcd_orig = o3d.io.read_point_cloud(file).points
        xyz = np.asarray(pcd_orig)
    elif ext == '.laz':
        las_data = laspy.read(file)
        # use the scaled/offset coordinates (lowercase x/y/z), not the raw integer
        # X/Y/Z record values -- the latter are pre-scale and are not real-world metres.
        xyz = np.c_[las_data.x, las_data.y, las_data.z]
    else:
        raise TypeError('unhandled extension ' + ext)

    if dtype is not None:
        xyz = xyz.astype(dtype)

    return xyz


def check_stacks(stacks, number_of_points):

    # Union and total membership count across all stacks.
    myset = set()
    total = 0
    for stack in stacks:
        total += len(stack)
        myset.update(int(i) for i in stack)

    # Coherency: the stacks must be DISJOINT (no point in two grains) and cover exactly
    # `number_of_points` labelled points. The old code also required min index == 0, but that
    # is NOT an invariant after clean_labels removes small/flat grains (point 0 can be dropped,
    # leaving min > 0) -- so that spurious check crashed ~40% of real tiles. It only checked
    # union cardinality, which silently accepts overlapping stacks; require disjointness too.
    if len(myset) != total:
        raise ValueError('stacks are not coherent: stacks overlap (a point appears in >1 grain)')
    if len(myset) != number_of_points:
        raise ValueError('stacks are not coherent: covered points != number_of_points')

    return True
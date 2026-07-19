from time import perf_counter

import numpy as np

from .tools import check_stacks


def add_to_stack(index, n_donors, donors, stack):
    # Add an outlet's donors to the stack, then those donors' donors, etc., until the entire
    # catchment is collected. Iterative pre-order DFS with an explicit work stack: a recursive
    # version overflows Python's call stack on large catchments (a deep donor chain on a big
    # tile has one frame per point). Donors are pushed in reverse so the first donor is popped
    # first -- byte-identical traversal order to the recursion this replaces.
    work = [index]
    while work:
        node = work.pop()
        stack.append(node)
        for k in range(n_donors[node] - 1, -1, -1):
            work.append(donors[node, k])


def add_to_stack_bw(index, delta, Di, stack, local_maximum):
    # Braun & Willett catchment collection, iterative for the same reason as add_to_stack (a
    # recursive version overflows the call stack on large catchments). Pushing donors in reverse
    # reproduces the recursion's exact pre-order sequence.
    work = [index]
    while work:
        node = work.pop()
        stack.append(node)
        for k in range(delta[node + 1] - 1, delta[node] - 1, -1):
            if Di[k] != local_maximum:  # avoid the local-maximum self-loop
                work.append(Di[k])


def philippe_steer_stack_building(receivers, local_maximum_indexes, knn):
    start = perf_counter()
    n_points = len(receivers)

    # identify the donors for each receiver
    ndon = np.zeros(n_points, dtype=int)  # number of donors
    donor = np.zeros((n_points, knn), dtype=int)  # donor list
    for k, receiver in enumerate(receivers):
        if receiver != k:
            ndon[receiver] = ndon[receiver] + 1
            donor[receiver, ndon[receiver] - 1] = k

    # build the stacks
    labels = np.zeros(n_points, dtype=int)
    stacks = []

    for k, ij in enumerate(local_maximum_indexes):
        stack = []
        add_to_stack(ij, ndon, donor, stack)  # recursive function
        stacks.append(stack)
        labels[stack] = k

    end = perf_counter()
    print(f'Philippe Steer {end - start}')

    return stacks, labels, ndon


def braun_willett_stack_building(receivers, local_maximum_indexes):
    start = perf_counter()
    n_points = len(receivers)

    # get the number of donors per receiver and build the list of donors per receiver
    di = np.zeros(n_points, dtype=int)  # number of donors
    Dij = [[] for i in range(n_points)]  # lists of donors
    for k, receiver in enumerate(receivers):
        di[receiver] = di[receiver] + 1
        Dij[receiver].append(k)

    # build Di, the list of donors
    Di = np.zeros(n_points, dtype=int)  # list of donors
    idx = 0
    for list_ in Dij:  # build the list of donors
        for point in list_:
            Di[idx] = point
            idx = idx + 1

    # build delta, the index array
    delta = np.zeros(n_points + 1, dtype=int)  # index of the first donor
    delta[n_points] = n_points
    for i in range(n_points - 1, -1, -1):
        delta[i] = delta[i + 1] - di[i]
    stacks = []
    labels = np.zeros(n_points, dtype=int)

    # build the stacks
    for k, ij in enumerate(local_maximum_indexes):
        stack = []
        add_to_stack_bw(ij, delta, Di, stack, ij)  # recursive function
        stacks.append(stack)
        labels[stack] = k

    end = perf_counter()
    print(f'[braun_willett_stack_building] {end - start: .2f} s')

    return stacks, labels, di


def segment_labels(xyz, knn, neighbors_indexes, braun_willett=True):
    print('[segment_labels]')

    # for each point, compute the slopes between the point and each one of its neighbors.
    # squeeze(axis=2) only removes the trailing length-1 axis from x[neighbors_indexes] (shape
    # (n, knn, 1)); a bare squeeze would also collapse the neighbour axis when knn == 1.
    x, y, z = np.split(xyz, 3, axis=1)
    n_points = len(xyz)
    dx = x - np.squeeze(x[neighbors_indexes], axis=2)
    dy = y - np.squeeze(y[neighbors_indexes], axis=2)
    dz = z - np.squeeze(z[neighbors_indexes], axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        slopes = dz / (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5  # slope between a point and each neighbour

    # A COINCIDENT neighbour (distance 0, from float32-quantised tiles collapsing near-identical
    # points) gives an undefined 0/0 = NaN slope. MATLAB's min/max ignore NaN; numpy's propagate it,
    # which corrupts the catchment graph and leaves some points in no stack -> "stacks are not
    # coherent". Replace ONLY NaN with +inf (leave any real +/-inf slope untouched): +inf is never
    # chosen as the downslope receiver and never a spurious downhill, matching MATLAB's NaN-ignoring
    # reduction. No-op on tiles without coincident points -> segmentation ARI stays 1.0 on the
    # fixtures. DELIBERATE DIVERGENCE (documented, PARITY.md): an ALL-coincident neighbourhood
    # becomes all-+inf -> min_slope = +inf >= 0 -> a singleton local maximum. MATLAB's `min` returns
    # NaN there and `argmin` picks the first index (an undefined artifact, no coherent sink); the
    # port instead yields a coherent singleton grain.
    slopes = np.where(np.isnan(slopes), np.inf, slopes)

    # for each point, find in the neighborhood the point with the minimum slope (the receiver)
    index_of_min_slope = np.argmin(slopes, axis=1)  # get the index of the point with the minimum slope
    min_slope = np.amin(slopes, axis=1)  # get the value of the minimum slope
    receivers = neighbors_indexes[np.arange(n_points), index_of_min_slope]

    # if the minimum slope is positive, we have a local maximum
    # look for the minimum slopes
    local_maximum_indexes = np.where(min_slope >= 0)[0]  # be careful to use >= and not >
    receivers[local_maximum_indexes] = local_maximum_indexes

    if braun_willett:
        stacks, labels, ndon = braun_willett_stack_building(receivers, local_maximum_indexes)
    else:
        stacks, labels, ndon = philippe_steer_stack_building(receivers, local_maximum_indexes, knn)

    if check_stacks(stacks, len(labels), n_cloud=len(labels)):
        print(f"[segment_labels] initial segmentation: {len(stacks)} labels")
    else:
        raise ValueError("[segment_labels] stacks are not valid")

    return labels, stacks, ndon, local_maximum_indexes

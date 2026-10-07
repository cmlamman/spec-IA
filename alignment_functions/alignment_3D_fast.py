"""
Optimized, parallel re-implementation of the 3D multiplet-alignment measurement in
`alignment_3D.get_3D_MIA_from3D_autocorr`, for plane-parallel (los_mode='z') simulation boxes.

Results are numerically identical to the original -- `verify_alignment_3D_fast.py` checks the
multiplet catalog, the per-batch 2D arrays and the final outputs against it. The speedups are
all bookkeeping, not approximations:

  1. Query radius.  The original searches a ball of radius sqrt(rp_max^2 + rpar_max^2) = sqrt(2)
     s_max and then cuts to a cylinder, but every pair it keeps with s > s_max falls outside the
     s bins and is silently dropped by the histogram. Searching a ball of radius s_max instead
     returns exactly the pairs that can land in a bin: 2.8x fewer pairs pre-cut, 1.5x fewer post.
  2. One tracer tree.  The original rebuilds the KD-tree of the full tracer catalog inside every
     batch (200 rebuilds of a ~10^6-point tree). It is built once here.
  3. One arctan2.  For a plane-parallel LOS the two position angles differ by exactly pi
     (pa1 = pa0 + pi), and every estimator uses them only as cos/sin of 2*angle, which that shift
     leaves invariant. The second arctan2 over every pair is redundant.
  4. All estimators in one pass.  x+, ++ and g+ share the same pair search, the same geometry and
     the same relative angles; only the final trig combination differs. The original repeats the
     entire measurement once per estimator.
  5. Binning.  s and mu are turned into a flat bin index once with searchsorted, then reused by
     np.bincount for the counts and for each estimator, instead of calling np.histogram2d
     (searchsorted + bincount internally) once per estimator plus once for the counts.
  6. Vectorized multiplet finding.  `find_components` unions *stringified* index pairs in a
     Python UnionFind and `make_group_catalog` then loops over every group in Python. Both are
     replaced by array operations (scipy connected_components + reduceat), reproducing the
     original group order, membership and per-group quantities exactly.
  7. Parallel batches.  The batches are independent, so they are farmed out to a process pool.
     Workers inherit the tracer tree through fork, so nothing large is pickled.

Everything that affects the *result* -- the multiplet catalog and its order, the RNG and the
batch partition, the ghost catalog, the bin edges -- is byte-for-byte the original logic.
"""

import os
import random

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from geometry_functions.coordinate_functions import get_proj_dist

ESTIMATORS = ('x+', '++', 'g+')

# scipy renamed the thread argument of query_ball_point in 1.6; support both
_QBP_WORKERS_KW = 'workers'
try:                                     # pragma: no cover - depends on the installed scipy
    cKDTree(np.zeros((2, 3))).query_ball_point(np.zeros((1, 3)), r=1.0, workers=1)
except TypeError:
    _QBP_WORKERS_KW = 'n_jobs'


# --------------------------------------------------------------------------- multiplet finding

def find_multiplets(points, transverse_max, los_max, max_n=100):
    """Group indices from mutual-nearest-neighbour linking, matching find_groups_UnionFind.

    Reproduces the original's ordering exactly: groups appear in order of first appearance of
    any member in the raveled pair list, members within a group in that same order, and any
    component larger than `max_n` is dropped (as the original's `root.size <= max_size` does).
    """
    tree = cKDTree(points)
    max_pair_sep = np.sqrt(transverse_max**2 + los_max**2)
    _, ii = tree.query(points, distance_upper_bound=max_pair_sep, k=2)
    ii = ii[ii[:, 1] < len(points)]

    # plane-parallel LOS along +z
    transverse_seps = np.abs(get_proj_dist(points[ii[:, 1]], points[ii[:, 0]], los_mode='z'))
    los_seps = np.abs(points[ii[:, 1], 2] - points[ii[:, 0], 2])
    ii = ii[~((transverse_seps > transverse_max) | (los_seps > los_max))]
    if len(ii) == 0:
        return []

    # Insertion order of the original UnionFind dict: keys appear as the pair list is scanned
    # row by row, first element then second -- i.e. first appearance in ii.ravel().
    flat = ii.ravel()
    nodes, first_at = np.unique(flat, return_index=True)
    order = np.argsort(first_at, kind='stable')
    nodes_in_order = nodes[order]                     # node ids, in dict-insertion order

    # connected components over the involved nodes only
    pos_of = np.full(flat.max() + 1, -1, dtype=np.int64)
    pos_of[nodes] = np.arange(len(nodes))
    a, b = pos_of[ii[:, 0]], pos_of[ii[:, 1]]
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(graph, directed=False)

    sizes = np.bincount(labels)
    lab_in_order = labels[pos_of[nodes_in_order]]
    keep = sizes[lab_in_order] <= max_n
    nodes_kept, lab_kept = nodes_in_order[keep], lab_in_order[keep]

    # group members by label, preserving insertion order both between and within groups
    sort = np.argsort(lab_kept, kind='stable')
    lab_sorted, nodes_sorted = lab_kept[sort], nodes_kept[sort]
    starts = np.flatnonzero(np.r_[True, lab_sorted[1:] != lab_sorted[:-1]])
    ends = np.r_[starts[1:], len(lab_sorted)]
    groups = [nodes_sorted[s:e] for s, e in zip(starts, ends)]

    # order groups by FIRST appearance of their label, as the original defaultdict does
    # (np.unique's return_index gives the first occurrence; enumerate would give the last)
    uniq_lab, first_pos = np.unique(lab_kept, return_index=True)
    first_of_label = dict(zip(uniq_lab.tolist(), first_pos.tolist()))
    group_labels = lab_sorted[starts].tolist()
    return [groups[i] for i in np.argsort([first_of_label[l] for l in group_labels],
                                          kind='stable')]


def multiplet_catalog(points_3D, transverse_max, los_max, max_n=100):
    """Vectorized equivalent of make_group_catalog(..., los_mode='z', use_sky_coords=False).

    Returns (centers, orientations, n_group, max_dist_to_center) as plain arrays.
    """
    groups = find_multiplets(points_3D, transverse_max, los_max, max_n=max_n)
    if len(groups) == 0:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int), np.zeros(0))

    n_group = np.array([len(g) for g in groups])
    members = np.concatenate(groups)                       # all members, group by group
    offsets = np.r_[0, np.cumsum(n_group)[:-1]]            # start of each group
    pts = points_3D[members]

    # per-group centroid
    sums = np.add.reduceat(pts, offsets, axis=0)
    centers = sums / n_group[:, None]

    # orientation: same arithmetic as calculate_2D_group_orientation, done on every member at once
    off = pts - np.repeat(centers, n_group, axis=0)
    x, y = off[:, 0], off[:, 1]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(x, y)
    z = r * np.exp(2j * theta)
    orientations = np.angle(np.add.reduceat(z, offsets) / n_group) / 2

    dist = np.sqrt(np.sum(off * off, axis=1))
    max_dist = np.maximum.reduceat(dist, offsets)
    return centers, orientations, n_group, max_dist


# --------------------------------------------------------------------------- the pair stage

def _bin_index(s, mu, s_bins, mu_bins):
    """Flat (s, mu) bin index, with np.histogram2d's edge conventions. -1 marks out of range."""
    ns, nm = len(s_bins) - 1, len(mu_bins) - 1
    si = np.searchsorted(s_bins, s, side='right') - 1
    mi = np.searchsorted(mu_bins, mu, side='right') - 1
    si[s == s_bins[-1]] = ns - 1          # right-most edge belongs to the last bin
    mi[mu == mu_bins[-1]] = nm - 1
    ok = (si >= 0) & (si < ns) & (mi >= 0) & (mi < nm)
    flat = np.where(ok, si * nm + mi, -1)
    return flat, ok, ns, nm


# module-level state shared with pool workers through fork (never pickled)
_W = {}


def _init_worker(state):
    _W.update(state)


def _run_batch(bounds):
    i_start, i_end = bounds
    s_bins, mu_bins = _W['s_bins'], _W['mu_bins']
    ns, nm = len(s_bins) - 1, len(mu_bins) - 1
    q_loc = _W['centers'][i_start:i_end]
    q_ang = _W['orients'][i_start:i_end]

    kw = {_QBP_WORKERS_KW: _W['query_threads']}
    # (1 + 1e-12) so a pair whose recomputed s lands a rounding step above s_max is still
    # returned, exactly as the original's larger search radius would have done
    ii = _W['tree'].query_ball_point(q_loc, r=s_bins[-1] * (1 + 1e-12), **kw)

    counts_per_q = [len(i) for i in ii]
    idx = np.concatenate(ii) if len(ii) else np.zeros(0, dtype=int)
    coords0 = np.repeat(q_loc, counts_per_q, axis=0)
    angles0 = np.repeat(q_ang, counts_per_q)
    coords1 = _W['tracer_locs'][idx]
    angles1 = _W['tracer_orients'][idx]

    # sign conventions kept exactly as the original: los_sep is (0 - 1) in z, while the
    # transverse difference inside get_proj_dist / the position angle is (1 - 0)
    los_sep = coords0[:, 2] - coords1[:, 2]
    proj_dist = np.abs(get_proj_dist(coords0, coords1, los_mode='z'))
    keep = (np.abs(los_sep) < s_bins[-1]) & (proj_dist < s_bins[-1])

    s_perp, s_par = proj_dist[keep], los_sep[keep]
    a0, a1 = angles0[keep], angles1[keep]
    dx = coords1[keep, 0] - coords0[keep, 0]
    dy = coords1[keep, 1] - coords0[keep, 1]
    pa0 = np.arctan2(dx, dy)
    # pa1 = pa0 + pi exactly, and cos/sin of twice an angle are invariant under that shift
    r0, r1 = a0 - pa0, a1 - pa0

    s = np.sqrt(s_perp**2 + s_par**2)
    mu = np.zeros_like(s)
    nz = s > 0
    mu[nz] = s_par[nz] / s[nz]

    flat, ok, ns, nm = _bin_index(s, mu, s_bins, mu_bins)
    flat_ok = flat[ok]
    counts = np.bincount(flat_ok, minlength=ns * nm).astype(float)

    two_r0, two_r1 = 2 * r0, 2 * r1
    cos0, sin0, cos1 = np.cos(two_r0), np.sin(two_r0), np.cos(two_r1)
    weights = {'x+': sin0 * cos1, '++': cos0 * cos1, 'g+': cos0}

    out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for est in _W['estimators']:
            tot = np.bincount(flat_ok, weights=weights[est][ok], minlength=ns * nm)
            out[est] = (tot / counts).reshape(ns, nm)
    return out


def measure_mia(points_3D, s_bins, mu_bins, transverse_max, los_max,
                estimators=ESTIMATORS, n_batches=200, max_n=100,
                periodic_boundary=True, n_workers=1, query_threads=1,
                catalog=None, print_info=False):
    """MIA(s, mu) for several estimators in one pass over the pairs.

    Returns {estimator: dict(mean, err, batches)} plus a 'catalog' entry describing the
    multiplets. `n_workers` processes handle the batches; `query_threads` is passed to
    query_ball_point inside each worker (leave at 1 when n_workers > 1).
    """
    points_3D = np.asarray(points_3D, dtype=float)
    points_3D = points_3D - np.min(points_3D, axis=0)
    s_bins, mu_bins = np.asarray(s_bins, float), np.asarray(mu_bins, float)

    if catalog is None:
        centers, orients, n_group, _ = multiplet_catalog(points_3D, transverse_max, los_max,
                                                          max_n=max_n)
    else:
        centers, orients, n_group = catalog
    n_mult = len(centers)
    if print_info:
        print(f'{n_mult} multiplets, <n_group> = {n_group.mean():.3f}', flush=True)

    if periodic_boundary:
        extend_by = np.max(s_bins)
        box_size = np.max(points_3D[:, 0])
        shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                            if not (i == 0 and j == 0 and k == 0)]) * box_size
        ghost_centers = np.concatenate([centers + s for s in shifts], axis=0)
        ghost_orients = np.tile(orients, len(shifts))
        lo, hi = np.min(points_3D, axis=0) - extend_by, np.max(points_3D, axis=0) + extend_by
        keep = np.all((ghost_centers > lo) & (ghost_centers < hi), axis=1)
        tracer_locs = np.concatenate([centers, ghost_centers[keep]], axis=0)
        tracer_orients = np.concatenate([orients, ghost_orients[keep]])
    else:
        tracer_locs, tracer_orients = centers, orients

    # identical shuffle and identical batch partition to the original (the trailing
    # len % n_batches multiplets are never queried, as in the original)
    random.seed(42)
    indices = np.asarray(range(n_mult))
    random.shuffle(indices)
    centers, orients = centers[indices], orients[indices]

    per_batch = int(n_mult / n_batches)
    bounds = [(i * per_batch, (i + 1) * per_batch) for i in range(int(n_batches))]

    state = dict(tree=cKDTree(tracer_locs), tracer_locs=tracer_locs,
                 tracer_orients=tracer_orients, centers=centers, orients=orients,
                 s_bins=s_bins, mu_bins=mu_bins, estimators=tuple(estimators),
                 query_threads=query_threads)

    if n_workers and n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('fork')          # fork so the tree is shared, not pickled
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(state,)) as pool:
            per_batch_out = pool.map(_run_batch, bounds, chunksize=1)
    else:
        _init_worker(state)
        per_batch_out = [_run_batch(b) for b in bounds]

    results = {}
    for est in estimators:
        stack = np.asarray([b[est] for b in per_batch_out])
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(stack, axis=0)
            err = np.nanstd(stack, axis=0) / np.sqrt(len(stack))
        results[est] = dict(mean=mean, err=err, batches=stack)
    results['catalog'] = dict(n_multiplets=n_mult, mean_n_group=float(n_group.mean()),
                              n_tracers=len(tracer_locs))
    return results

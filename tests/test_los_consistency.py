"""
Madey by Claude

Line-of-sight (LOS) consistency tests for get_MIA_from3D and the functions it calls.

Every LOS-dependent step (finding multiplets, projecting their shapes, pair separations and
position angles) must use the same LOS. These tests check that:
  * LOS arguments are validated (resolve_los, get_MIA_from3D)
  * each step uses the requested LOS axis / observer (planted geometries with known answers)
  * position angles are measured in the pair-midpoint sky plane, exactly as the original
    get_orientation_angle_cartesian did (multiplet shapes are measured in their own sky plane)
  * the measurement is unchanged by operations that must not change it:
      - translating the points and the observer together
      - rotating the points about the observer's z axis (radial LOS)
      - relabelling the coordinate axes together with los_axis
  * a radial LOS from a very distant observer reproduces the single-axis result
  * get_MIA_from3D agrees with an independent brute-force implementation of the estimator,
    for single-axis and radial LOS, with and without periodic boundaries
  * cached batch files from one LOS are never reused for another

HOW TO RUN
----------
    pytest test_los_consistency.py -v
"""
import contextlib, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from alignment_functions.gal_multiplets import (make_group_catalog, find_groups_UnionFind,
                                                calculate_2D_group_orientation, get_MIA_from3D)
from alignment_functions.basic_alignment import calculate_rel_ang_cartesian_binAverage
from geometry_functions.coordinate_functions import (resolve_los, los_unit_vectors, get_proj_dist,
                                                     get_orientation_angle_cartesian)

BOX = 200.
R_BINS = np.logspace(np.log10(5), np.log10(40), 5)
PIMAX = 8 + (2/3) * (R_BINS[1:] + R_BINS[:-1]) / 2
LINK = dict(transverse_max=1, los_max=1)


def filament_mock(seed, n_fil=120, n_per_fil=40, fil_length=30., n_background=3000):
    '''Points along randomly oriented line segments (giving aligned multiplets) plus a uniform background, in [0, BOX)^3.'''
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, BOX, size=(n_fil, 3))
    dirs = rng.normal(size=(n_fil, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    t = rng.uniform(0, fil_length, size=(n_fil, n_per_fil))
    pts = starts[:, None, :] + t[..., None] * dirs[:, None, :] + rng.normal(scale=0.2, size=(n_fil, n_per_fil, 3))
    pts = np.concatenate([pts.reshape(-1, 3), rng.uniform(0, BOX, size=(n_background, 3))])
    return np.mod(pts, BOX)


def measure(points, **los):
    '''Multiplets + projected alignment, using the same LOS for both steps.'''
    groups = make_group_catalog(None, comoving_points=points, use_sky_coords=False, **LINK, **los)
    signal, counts = calculate_rel_ang_cartesian_binAverage(
        np.asarray(groups['center_loc']), np.asarray(groups['orientation']), points, np.ones(len(points)),
        R_bins=R_BINS, pimax=PIMAX, E_ABS=np.ones(len(groups)), return_pair_counts=True, **los)
    return groups, np.asarray(signal), np.asarray(counts)


def group_sets(points, **los):
    return set(map(frozenset, find_groups_UnionFind(points, max_n=100, **LINK, **los)))


# ------------------------------------------------------------------ argument validation

def test_resolve_los():
    assert resolve_los('axis', los_axis='y') == {'los_mode': 'axis', 'los_location': None, 'los_axis': 1}
    assert resolve_los('axis', los_axis=np.int64(0))['los_axis'] == 0
    assert resolve_los('z') == {'los_mode': 'axis', 'los_location': None, 'los_axis': 2}      # legacy shorthand
    assert resolve_los('z', los_location=[0, 0, 0])['los_location'] is None                  # legacy callers pass this
    np.testing.assert_array_equal(resolve_los('radial')['los_location'], [0, 0, 0])
    loc = np.array([1., 2., 3.])
    los = resolve_los('radial', loc)
    loc[0] = 99.
    np.testing.assert_array_equal(los['los_location'], [1, 2, 3])                            # copied, not aliased
    for los in [resolve_los('x'), resolve_los('radial', [4, 5, 6])]:
        out = resolve_los(**los)                                                             # canonical form is a fixed point
        assert out['los_mode'] == los['los_mode'] and out['los_axis'] == los['los_axis']
        np.testing.assert_array_equal(out['los_location'], los['los_location'])

    bad = [dict(los_mode='axis'), dict(los_mode='radial', los_axis='z'), dict(los_mode='z', los_axis='x'),
           dict(los_mode='axis', los_axis='w'), dict(los_mode='axis', los_axis=3), dict(los_mode='axis', los_axis=True),
           dict(los_mode='radial', los_location=[1, 2]), dict(los_mode='radial', los_location=[0, np.nan, 0]),
           dict(los_mode='plane-parallel'), dict(los_mode=None)]
    for kwargs in bad:
        with pytest.raises(ValueError):
            resolve_los(**kwargs)


def test_get_MIA_from3D_requires_explicit_los(tmp_path):
    pts = filament_mock(0)
    for kwargs in [dict(), dict(los_mode='radial'), dict(los_mode='axis'),
                   dict(los_mode='axis', los_axis='z', los_location=[0, 0, 0]),
                   dict(los_mode='radial', los_location=[0, 0, 0], los_axis='z')]:
        with pytest.raises(ValueError):
            get_MIA_from3D(pts, str(tmp_path), print_info=False, **kwargs)


# ------------------------------------------------------------------ each step uses the requested LOS

def test_sky_axes_conventions():
    '''East/North conventions: legacy 'z' (east=+x, north=+y), the cyclic analogues for 'x' and 'y',
    and RA/DEC for a radial LOS from the origin (at RA=0, DEC=0: east=+y, north=+z).
    (Radial position angles are of points1 as seen from points2, as in the original code: pi from the others.)'''
    p = np.array([[100., 0., 0.]])
    for los, east, north in [(dict(los_mode='axis', los_axis='z'), [1, 0, 0], [0, 1, 0]),
                             (dict(los_mode='z'), [1, 0, 0], [0, 1, 0]),
                             (dict(los_mode='axis', los_axis='x'), [0, 1, 0], [0, 0, 1]),
                             (dict(los_mode='axis', los_axis='y'), [0, 0, 1], [1, 0, 0]),
                             (dict(los_mode='radial', los_location=[0, 0, 0]), [0, 1, 0], [0, 0, 1])]:
        flip = np.pi if los['los_mode'] == 'radial' else 0.
        assert np.isclose(np.cos(get_orientation_angle_cartesian(p, p + east, **los)[0] - np.pi / 2 - flip), 1)
        assert np.isclose(np.cos(get_orientation_angle_cartesian(p, p + north, **los)[0] - flip), 1)
        members = np.vstack([p - 0.3 * np.asarray(east), p + 0.3 * np.asarray(east)])
        assert np.isclose(np.cos(2 * calculate_2D_group_orientation(members, **los)), -1)   # E-W elongated: angle = pi/2


def test_group_finder_uses_los():
    '''Two points 2 apart along x form a pair (transverse_max=3, los_max=1) only if x is NOT the LOS.'''
    p = np.array([[50., 60., 70.]])
    pair = np.vstack([p, p + [2., 0, 0]])
    kw = dict(max_n=10, transverse_max=3, los_max=1)
    assert find_groups_UnionFind(pair, los_mode='axis', los_axis='x', **kw) == []
    assert len(find_groups_UnionFind(pair, los_mode='axis', los_axis='y', **kw)) == 1
    assert len(find_groups_UnionFind(pair, los_mode='axis', los_axis='z', **kw)) == 1
    assert find_groups_UnionFind(pair, los_mode='radial', los_location=p[0] - [1e6, 0, 0], **kw) == []
    assert len(find_groups_UnionFind(pair, los_mode='radial', los_location=p[0] - [0, 0, 1e6], **kw)) == 1


PLANTED_LOS = [dict(los_mode='axis', los_axis='x'), dict(los_mode='axis', los_axis='y'),
               dict(los_mode='axis', los_axis='z'), dict(los_mode='radial', los_location=[0., 0., 160.]),
               dict(los_mode='radial', los_location=[-300., 100., 160.]),
               dict(los_mode='radial', los_location=[100., 300., 160.])]   # multiplet in front of the observer

@pytest.mark.parametrize('los', PLANTED_LOS)
def test_planted_alignment(los):
    '''A multiplet elongated along a sky direction u: a tracer along u gives cos(2 phi) = +1, one along the
    perpendicular sky direction v gives -1. u, v are built from the LOS directly, not from the package's sky axes.
    Single axis: u is a random sky direction. Radial: the multiplet is level with the observer (on its 'equator')
    and u, v are local East and North, where the pair-midpoint sky plane gives exactly +/-1.'''
    P = np.array([100., 100., 160.])
    n_hat = los_unit_vectors(P, **los)[0]
    if los['los_mode'] == 'axis':
        u = np.random.default_rng(1).normal(size=3)
        u -= (u @ n_hat) * n_hat
        u /= np.linalg.norm(u)
    else:
        u = np.cross([0., 0., 1.], n_hat)
        u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    members = np.vstack([P - 0.4 * u, P + 0.4 * u])
    points = np.vstack([members, P + 10 * u, P + 10 * v])

    groups = make_group_catalog(None, comoving_points=points, use_sky_coords=False, **LINK, **los)
    assert len(groups) == 1 and groups['n_group'][0] == 2
    assert np.isclose(np.cos(2 * (groups['orientation'][0] - get_orientation_angle_cartesian(P[None], (P + u)[None], **los)[0])), 1)
    for tracer, expected in [(P + 10 * u, 1.), (P + 10 * v, -1.)]:
        signal = calculate_rel_ang_cartesian_binAverage(
            np.asarray(groups['center_loc']), np.asarray(groups['orientation']), tracer[None], [1.],
            R_bins=np.array([5., 15.]), pimax=5., E_ABS=np.ones(1), **los)
        assert np.isclose(signal[0], expected, atol=1e-9)


def original_midpoint_position_angle(points1, points2, los_location):
    '''The radial position angle of the original get_orientation_angle_cartesian (git HEAD), copied verbatim.'''
    dx = points2[:, 0] - points1[:, 0]
    dy = points2[:, 1] - points1[:, 1]
    dz = points2[:, 2] - points1[:, 2]
    mx = 0.5 * (points1[:, 0] + points2[:, 0]) - los_location[0]
    my = 0.5 * (points1[:, 1] + points2[:, 1]) - los_location[1]
    mz = 0.5 * (points1[:, 2] + points2[:, 2]) - los_location[2]
    mnorm = np.sqrt(mx * mx + my * my + mz * mz)
    nx = mx / mnorm
    ny = my / mnorm
    nz = mz / mnorm
    proj_alpha = dx * ny - dy * nx
    d_dot_n = dx * nx + dy * ny + dz * nz
    proj_delta = nz * d_dot_n - dz
    return np.arctan2(proj_alpha, proj_delta)


@pytest.mark.parametrize('los', [dict(los_mode='radial', los_location=[0., 0., 0.]),
                                 dict(los_mode='radial', los_location=[-150., 80., 120.]),
                                 dict(los_mode='radial', los_location=[100., 100., 100.]),
                                 dict(los_mode='axis', los_axis='x'), dict(los_mode='axis', los_axis='z')])
def test_position_angle_midpoint_convention(los):
    '''Position angles are measured in the pair-midpoint sky plane: both directions of a pair differ by exactly pi
    (as alignment_3D_fast assumes), and for a radial LOS they equal the original midpoint formula.'''
    pts = filament_mock(10)
    p1, p2 = pts[:-1], pts[1:]
    pa12 = get_orientation_angle_cartesian(p1, p2, **los)
    pa21 = get_orientation_angle_cartesian(p2, p1, **los)
    np.testing.assert_allclose(np.cos(pa12 - pa21), -1, atol=1e-9)
    if los['los_mode'] == 'radial':
        pa_orig = original_midpoint_position_angle(p1, p2, np.asarray(los['los_location']))
        np.testing.assert_allclose(np.cos(pa12 - pa_orig), 1, atol=1e-9)


def test_raw_outputs_match_original():
    '''Existing callers see the same raw values as before (formulas below copied from the original code):
    get_proj_dist, get_orientation_angle_cartesian and calculate_2D_group_orientation, radial (any observer) and 'z'.'''
    def original_proj_dist(pos1, pos2, pos_obs):
        dx, dy, dz = (pos2 - pos1).T
        ox, oy, oz = (0.5 * (pos2 + pos1) - pos_obs).T
        d2 = dx * dx + dy * dy + dz * dz
        dot = dx * ox + dy * oy + dz * oz
        perp2 = d2 - (dot * dot) / (ox * ox + oy * oy + oz * oz)
        return np.sqrt(np.maximum(perp2, 0.0))

    def original_group_orientation(points_3d, los_location, los_mode):
        offsets_3d = points_3d - np.mean(points_3d, axis=0)
        if los_mode == 'z':
            x, y = offsets_3d[:, 0], offsets_3d[:, 1]
        else:
            n_hat = np.mean(points_3d, axis=0) - los_location
            n_hat = n_hat / np.linalg.norm(n_hat)
            plane_y = np.array([0.0, 0.0, 1.0]) - n_hat[2] * n_hat
            plane_y /= np.linalg.norm(plane_y)
            plane_x = np.cross(plane_y, n_hat)
            plane_x /= np.linalg.norm(plane_x)
            x, y = offsets_3d @ plane_x, offsets_3d @ plane_y
        points_2d = np.column_stack([x, y])
        r = np.linalg.norm(points_2d, axis=1)
        theta = np.arctan2(points_2d[:, 0], points_2d[:, 1])
        return np.angle(np.mean(r * np.exp(2j * theta))) / 2

    pts = filament_mock(11)
    p1, p2 = pts[:-1], pts[1:]
    for obs in [np.zeros(3), np.array([0., -200., 4321.])]:
        np.testing.assert_allclose(get_proj_dist(p1, p2, pos_obs=obs), original_proj_dist(p1, p2, obs), rtol=1e-12, atol=1e-12)
        d = get_orientation_angle_cartesian(p1, p2, los_location=obs) - original_midpoint_position_angle(p1, p2, obs)
        np.testing.assert_allclose(np.cos(d), 1, atol=1e-12)
        for members in find_groups_UnionFind(pts, max_n=100, los_mode='radial', los_location=obs, **LINK)[:300]:
            np.testing.assert_allclose(calculate_2D_group_orientation(pts[members], los_location=obs),
                                       original_group_orientation(pts[members], obs, 'radial'), atol=1e-12)
    dx, dy = (p2 - p1)[:, :2].T
    np.testing.assert_array_equal(get_proj_dist(p1, p2, los_mode='z'), np.sqrt(dx * dx + dy * dy))
    np.testing.assert_array_equal(get_orientation_angle_cartesian(p1, p2, los_mode='z'), np.arctan2(*(p2 - p1)[:, :2].T))
    for members in find_groups_UnionFind(pts, max_n=100, los_mode='z', **LINK)[:300]:
        assert calculate_2D_group_orientation(pts[members], los_mode='z') == original_group_orientation(pts[members], None, 'z')


def test_inputs_not_modified():
    pts = filament_mock(0)
    pts_before = pts.copy()
    obs = np.array([-100., 50., 50.])
    make_group_catalog(None, comoving_points=pts, use_sky_coords=False, los_mode='radial', los_location=obs)
    find_groups_UnionFind(pts, los_mode='radial', los_location=obs)
    np.testing.assert_array_equal(pts, pts_before)
    np.testing.assert_array_equal(obs, [-100., 50., 50.])


# ------------------------------------------------------------------ invariances

@pytest.mark.parametrize('los', [dict(los_mode='radial', los_location=[-150., 80., 120.]),
                                 dict(los_mode='axis', los_axis='y')])
def test_translation_invariance(los):
    pts = filament_mock(1)
    shift = np.array([1234.5, -987.25, 333.125])
    los_shifted = dict(los)
    if los['los_mode'] == 'radial':
        los_shifted['los_location'] = np.asarray(los['los_location']) + shift
    assert group_sets(pts, **los) == group_sets(pts + shift, **los_shifted)
    _, s0, n0 = measure(pts, **los)
    _, s1, n1 = measure(pts + shift, **los_shifted)
    np.testing.assert_array_equal(n0, n1)
    np.testing.assert_allclose(s1, s0, atol=1e-9)


def test_rotation_invariance_radial():
    '''Rotating everything about the observer's z axis (the axis that defines North) must not change the result.'''
    pts = filament_mock(2)
    obs = np.array([-150., 80., 120.])
    rot = Rotation.from_rotvec([0, 0, 2.1]).as_matrix()
    pts_rot = (pts - obs) @ rot.T + obs
    los = dict(los_mode='radial', los_location=obs)
    assert group_sets(pts, **los) == group_sets(pts_rot, **los)
    g0, s0, n0 = measure(pts, **los)
    g1, s1, n1 = measure(pts_rot, **los)
    assert np.all(np.isfinite(s0)) and np.all(n0 > 100)
    np.testing.assert_array_equal(n0, n1)
    np.testing.assert_allclose(s1, s0, atol=1e-9)


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_axis_relabelling(axis):
    '''Relabelling the coordinates so that `axis` becomes z, and using los_axis='z', must give the same result.'''
    pts = filament_mock(3)
    a = 'xyz'.index(axis)
    perm = [(a + 1) % 3, (a + 2) % 3, a]
    g0, s0, n0 = measure(pts, los_mode='axis', los_axis=axis)
    g1, s1, n1 = measure(pts[:, perm], los_mode='axis', los_axis='z')
    np.testing.assert_array_equal(n0, n1)
    np.testing.assert_allclose(s1, s0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(g1['orientation']), np.asarray(g0['orientation']), atol=1e-12)


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_distant_observer_matches_axis(axis):
    '''A radial LOS from a very distant observer on the -axis side must reproduce los_mode='axis'.
    (Not z: with the observer far along -z every LOS is close to +z, the direction that defines North, and
    the shape and pair-midpoint sky planes then have very different North directions.)'''
    pts = filament_mock(4)
    obs = np.full(3, BOX / 2)
    obs['xyz'.index(axis)] = -1e7
    assert group_sets(pts, los_mode='axis', los_axis=axis) == group_sets(pts, los_mode='radial', los_location=obs)
    _, s0, n0 = measure(pts, los_mode='axis', los_axis=axis)
    _, s1, n1 = measure(pts, los_mode='radial', los_location=obs)
    np.testing.assert_allclose(n1, n0, rtol=1e-3)
    np.testing.assert_allclose(s1, s0, atol=1e-3)


# ------------------------------------------------------------------ end-to-end vs brute force

def east_north(n_hat):
    '''East, North unit vectors of the sky plane perpendicular to unit LOS vectors n_hat (n, 3): North = projection of +z.'''
    north = np.array([0., 0., 1.]) - n_hat[:, 2:3] * n_hat
    north /= np.linalg.norm(north, axis=1)[:, None]
    return np.cross(north, n_hat), north


def brute_force_mia(points, groups, R_bins, pimax, los, box=None):
    '''Independent, slow implementation of the estimator: average cos(2 * (multiplet axis - position angle of the
    tracer)) over pairs with r_p in an R bin, |r_par| < pimax of that bin, and 3D separation within
    sqrt(max(R)^2 + max(pimax)^2) (the neighbour search radius of the estimator).
    Single axis: every sky plane is the same, so both angles are measured in one arbitrary basis of it.
    Radial: the multiplet axis is measured East of North in the sky plane at its centre, the position angle East of
    North in the sky plane at the pair midpoint (North = projection of +z).
    r_p is transverse to the pair-midpoint LOS (radial) or the axis; r_par is the difference in distance from the
    observer (radial) or the separation along the axis.'''
    tracers = points
    if box is not None:
        shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) * box
        tracers = np.concatenate([points + s for s in shifts])
    tree = cKDTree(tracers)
    radius = np.sqrt(np.max(R_bins)**2 + np.max(pimax)**2)
    rng = np.random.default_rng(42)
    sums = np.zeros(len(R_bins) - 1)
    counts = np.zeros(len(R_bins) - 1)
    for members in groups:
        m = points[members]
        c = m.mean(axis=0)
        t = tracers[tree.query_ball_point(c, radius)]
        d = t - c
        if los['los_mode'] == 'axis':
            n_hat = np.eye(3)['xyz'.index(los['los_axis'])]
            r_par = d @ n_hat
            r_p = np.linalg.norm(d - r_par[:, None] * n_hat, axis=1)
            e1 = rng.normal(size=3)
            e1 -= (e1 @ n_hat) * n_hat
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n_hat, e1)
            off = (m - c) @ e1 + 1j * ((m - c) @ e2)
            sep_angle = np.angle(d @ e1 + 1j * (d @ e2))
        else:
            obs = np.asarray(los['los_location'], dtype=float)
            mid = (t + c) / 2 - obs
            mid /= np.linalg.norm(mid, axis=1)[:, None]
            along_mid = np.sum(d * mid, axis=1)
            r_p = np.sqrt(np.maximum(np.sum(d * d, axis=1) - along_mid**2, 0))
            r_par = np.linalg.norm(t - obs, axis=1) - np.linalg.norm(c - obs)
            east_c, north_c = east_north(((c - obs) / np.linalg.norm(c - obs))[None])
            off = (m - c) @ north_c[0] + 1j * ((m - c) @ east_c[0])            # angle from North towards East
            east_m, north_m = east_north(mid)
            sep_angle = np.angle(np.sum(d * north_m, axis=1) + 1j * np.sum(d * east_m, axis=1))
        axis_angle = np.angle(np.mean(np.abs(off) * np.exp(2j * np.angle(off)))) / 2
        cos2 = np.cos(2 * (sep_angle - axis_angle))
        b = np.digitize(r_p, R_bins) - 1
        ok = (r_p > R_bins[0]) & (r_p < R_bins[-1])
        ok[ok] &= np.abs(r_par[ok]) < np.broadcast_to(pimax, (len(R_bins) - 1,))[b[ok]]
        np.add.at(sums, b[ok], cos2[ok])
        np.add.at(counts, b[ok], 1)
    return sums / counts, counts


@pytest.mark.parametrize('los, periodic, pimax', [
    (dict(los_mode='axis', los_axis='z'), False, 'variable'),
    (dict(los_mode='axis', los_axis='x'), True, 'variable'),
    (dict(los_mode='axis', los_axis='y'), True, 60.),          # pimax > max(R_bins): periodic padding along the LOS axis matters
    (dict(los_mode='radial', los_location=[-250., 90., 130.]), False, 'variable'),
    (dict(los_mode='radial', los_location=[-250., 90., 130.]), True, PIMAX * 0.5),
    (dict(los_mode='radial', los_location=[100., 100., 100.]), False, 'variable'),   # observer inside the box
])
def test_get_MIA_from3D_matches_brute_force(los, periodic, pimax, tmp_path):
    pts = filament_mock(5)
    pts_before = pts.copy()
    radial_periodic = periodic and los['los_mode'] == 'radial'
    with pytest.warns(UserWarning, match='periodic') if radial_periodic else contextlib.nullcontext():
        res = get_MIA_from3D(pts, str(tmp_path), R_bins=R_BINS, pimax=pimax, print_info=False, sim_label='test',
                             periodic_boundary=periodic, box_size=BOX if periodic else None, n_batches=1,
                             return_pair_counts=True, **LINK, **los)
    np.testing.assert_array_equal(pts, pts_before)

    pimax_values = PIMAX if isinstance(pimax, str) else pimax
    groups = find_groups_UnionFind(pts, max_n=100, **LINK, **los)
    ref_signal, ref_counts = brute_force_mia(pts, groups, R_BINS, pimax_values, los, box=BOX if periodic else None)
    np.testing.assert_array_equal(np.asarray(res['pair_counts']), ref_counts)
    np.testing.assert_allclose(np.asarray(res['relAng_plot']), ref_signal, atol=1e-10)
    np.testing.assert_allclose(np.asarray(res['pimax']), np.broadcast_to(pimax_values, (len(R_BINS) - 1,)))

    # the saved table records the LOS, in its header and its file name
    tag = 'los' + los['los_axis'] if los['los_mode'] == 'axis' else 'losradial' + '_'.join('%g' % v for v in los['los_location'])
    saved = glob.glob(str(tmp_path / ('*_' + tag + '_*.fits')))
    assert len(saved) == 1
    meta = Table.read(saved[0]).meta
    assert meta['LOSMODE'] == los['los_mode']
    if los['los_mode'] == 'axis':
        assert meta['LOSAXIS'] == los['los_axis']
    else:
        np.testing.assert_allclose([meta['LOSOBSX'], meta['LOSOBSY'], meta['LOSOBSZ']], los['los_location'])


def test_get_MIA_from3D_batches_and_cache(tmp_path):
    '''Parallel batches match sequential ones, and saved batch files from one LOS are not reused for another.'''
    pts = filament_mock(6)
    kw = dict(R_bins=R_BINS, print_info=False, sim_label='cache', n_batches=4, save_intermediate=True, **LINK)
    r_z = get_MIA_from3D(pts, str(tmp_path), los_mode='axis', los_axis='z', n_jobs=1, **kw)
    r_x = get_MIA_from3D(pts, str(tmp_path), los_mode='axis', los_axis='x', n_jobs=2, **kw)
    os.mkdir(tmp_path / 'fresh')
    r_x_fresh = get_MIA_from3D(pts, str(tmp_path / 'fresh'), los_mode='axis', los_axis='x', n_jobs=1, **kw)
    assert not np.allclose(r_z['relAng_plot'], r_x['relAng_plot'])
    np.testing.assert_allclose(r_x['relAng_plot'], r_x_fresh['relAng_plot'], atol=1e-12)
    np.testing.assert_allclose(r_x['relAng_plot_e'], r_x_fresh['relAng_plot_e'], atol=1e-12)

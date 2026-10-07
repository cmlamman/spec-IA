# Useful functions for dealing with the coordinates of and making sky catalogs

import numpy as np
from astropy.table import Table, join, vstack
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.cosmology import LambdaCDM, z_at_value

cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)
def rw1_to_z(rw1):
    # rough photometric relation from DESI SV
    return rw1*0.2443683701202549+0.0037087929968548927
def comoving_to_z(d_comoving): # in units of Mpc/h
    return z_at_value(cosmo.comoving_distance, d_comoving * 0.7 * u.Mpc)

def v_to_cz(v, z):
    '''returns comoving Mpc / h of positions with RSD'''
    return 0.7 * (cosmo.comoving_distance(z) + ((1 + z) * (v * u.km / u.s) / cosmo.H(z))).to(u.Mpc).value

def wrap_180(angles):
    angles -= (angles >=180) * 360
    return angles

def wrap_pi(angles):
    angles -= (angles >=np.pi) * 2*np.pi
    return angles

# converting between deg and radians ( if not already in astropy angle)
def rad_to_deg(ang_rad):
    return ang_rad * 180 / np.pi
def deg_to_rad(ang_deg):
    return ang_deg * np.pi / 180

def remove_astropyu(values_list, unit=u.Mpc):
    if isinstance(values_list[0], u.Quantity):
        return [v.to(u.Mpc).value for v in values_list]
    else:
        return values_list
    
def get_mu(r_p, r_par):
    return np.cos(np.arctan2(r_p, r_par))

###########################################################################################################
# CALCULATING RELATIVE SEPARATIONS AND ANGLES
###########################################################################################################

def get_sep(ra1, dec1, ra2, dec2, u_coords='deg', u_result=u.rad):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - astropy quantity of separation 
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    return (c1.separation(c2)).to(u_result)
    
def get_pa(ra1, dec1, ra2, dec2, u_coords='deg', u_result=u.rad):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - separation [deg]
    - astropy quantity of position angle of second galaxy relative to first [deg], E of N
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    pa = c1.position_angle(c2).to(u_result)
    return pa


def get_sep_pa(ra1, dec1, ra2, dec2, u_coords='deg'):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - separation [deg]
    - position angle of second galaxy relative to first [deg], E of N
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    sep = c1.separation(c2).to(u.rad)
    pa = c1.position_angle(c2).to(u.rad)
    return sep, pa

def get_points(data):
    '''convert from astropy table of RA and DEC to cartesian coordinates on a unit sphere'''
    points = SkyCoord(data['RA'], data['DEC'], unit='deg', frame='icrs', equinox='J2000.0')
    points = points.cartesian   # old astropy: points.representation = 'cartesian'
    return np.dstack([points.x.value, points.y.value, points.z.value])[0]

def get_cosmo_points(data, cosmology=cosmo, truez=False):
    '''convert from astropy table of RA, DEC, and Z to 3D cartesian coordinates in Mpc/h'''
    if truez:
        comoving_dist = cosmo.comoving_distance(data['TRUEZ']).to(u.Mpc)
    else:
        comoving_dist = cosmo.comoving_distance(data['Z']).to(u.Mpc)
    points = coordinates.spherical_to_cartesian(np.abs(comoving_dist), np.asarray(data['DEC'])*u.deg, np.asarray(data['RA'])*u.deg)     # in Mpc
    cp_points = np.asarray(points).transpose() * cosmology.h                                                                            # in Mpc/h
    return np.float32(cp_points)


def get_pair_coords(obs_pos1, obs_pos2, use_center_origin=True, cosmology=cosmo):
    '''
    Takes in observed positions of galaxy pairs and returns comoving coordinates, in Mpc/h, with the orgin at the center of the pair. 
    The first coordinate (x-axis) is along the LOS
    The second coordinate (y-axis) is along 'RA'
    The third coordinate (z-axis) along 'DEC', i.e. aligned with North in origional coordinates.
    
    INPUT
    -------
    obs_pos1, obs_pos2: table with columns: 'RA', 'DEC', z_column
    use_center_origin: True for coordinate orgin at center of pair, othersise centers on first position
    cosmology: astropy.cosmology
    
    RETURNS
    -------
    numpy array of cartesian coordinates, in Mpc/h. Shape (2,3)

    '''
    cartesian_coords = get_cosmo_points(vstack([obs_pos1, obs_pos2]), cosmology=cosmology)  # in Mpc/h
    # find center position of coordinates
    origin = cartesian_coords[0]
    if use_center_origin==True:
        origin = np.mean(cartesian_coords, axis=0)
    cartesian_coords -= origin
    return cartesian_coords                 # in Mpc/h

def add_rsd(z, v_3d, pos_3d, poo_3d=np.asarray([-3700, 0, 0])*.7):
    '''
    z: redshift
    v_3d: 3d velocities, units of km/s. shape is (n positions, 3)
    pos_3d: 3d positions, units of Mpc/h. shape is (n positions, 3)
    poo_3d: 3d position of oberver. shape is (3,) in units of Mpc/h
    returns comoving distance with rsd, units of Mpc/h
    '''
    los_vector = pos_3d - poo_3d
    los_unit = los_vector / np.linalg.norm(los_vector, axis=1)[:, np.newaxis]
    v = np.sum(v_3d*los_unit, axis=1)
    return 0.7 * (cosmo.comoving_distance(z) + ((1 + z) * (v * u.km / u.s) / cosmo.H(z))).to(u.Mpc).value

def get_cosmo_psep_pa(ra1, dec1, ra2, dec2, z1, z2, u_coords='deg'):
    '''
    Input: ra and decs [deg] and redshifts for two objects. 
    Returns: 
    - physical projected separation [Mpc/h]
    '''
    angular_sep, pa = get_sep_pa(ra1, dec1, ra2, dec2, u_coords=u_coords)
    comoving_distances = cosmo.comoving_distance([z1, z2]).to(u.Mpc)
    psep = angular_sep * comoving_distances[0]
    return psep, pa


############
# LINE OF SIGHT (LOS)
############
# Every LOS-dependent step (finding multiplets, projecting their shapes, pair separations and
# position angles) goes through the functions below, so one set of LOS arguments
# (los_mode, los_location, los_axis) defines the projection everywhere:
#   los_mode='radial': the LOS of each object points from the observer at los_location to the object.
#   los_mode='axis':   every LOS is parallel to the coordinate axis los_axis ('x', 'y' or 'z'),
#                      with the observer at -infinity along that axis.

_LOS_AXES = {'x': 0, 'y': 1, 'z': 2}

def _axis_index(los_axis):
    if isinstance(los_axis, str) and los_axis.lower() in _LOS_AXES:
        return _LOS_AXES[los_axis.lower()]
    if isinstance(los_axis, (int, np.integer)) and not isinstance(los_axis, bool) and 0 <= los_axis <= 2:
        return int(los_axis)
    raise ValueError("los_axis must be 'x', 'y' or 'z' (or 0, 1, 2), got %r" % (los_axis,))

def resolve_los(los_mode='radial', los_location=None, los_axis=None):
    '''
    Validate line-of-sight (LOS) arguments and return them in canonical form, as a dict that can be
    passed on to any LOS-aware function with **los.

    los_mode: 'radial' (LOS from the observer at los_location to each object) or
              'axis' (LOS parallel to the coordinate axis los_axis).
              'x', 'y' or 'z' are shorthand for los_mode='axis' with that los_axis.
    los_location: observer position, shape (3,). Used for 'radial' (default: the origin); ignored for 'axis'.
    los_axis: 'x', 'y' or 'z' (or 0, 1, 2). Required for 'axis'; an error for 'radial'.

    returns: {'los_mode': 'radial', 'los_location': float array (3,), 'los_axis': None} or
             {'los_mode': 'axis', 'los_location': None, 'los_axis': 0, 1 or 2}
    '''
    if isinstance(los_mode, str) and los_mode in _LOS_AXES:
        if los_axis is not None and _axis_index(los_axis) != _LOS_AXES[los_mode]:
            raise ValueError("los_mode=%r conflicts with los_axis=%r" % (los_mode, los_axis))
        los_mode, los_axis = 'axis', los_mode

    if isinstance(los_mode, str) and los_mode == 'radial':
        if los_axis is not None:
            raise ValueError("los_axis is only used with los_mode='axis'")
        location = np.zeros(3) if los_location is None else np.array(los_location, dtype=float).reshape(-1)
        if location.shape != (3,) or not np.all(np.isfinite(location)):
            raise ValueError('los_location must be a finite 3D position, got %r' % (los_location,))
        return {'los_mode': 'radial', 'los_location': location, 'los_axis': None}

    if isinstance(los_mode, str) and los_mode == 'axis':
        if los_axis is None:
            raise ValueError("los_mode='axis' needs los_axis ('x', 'y' or 'z')")
        return {'los_mode': 'axis', 'los_location': None, 'los_axis': _axis_index(los_axis)}

    raise ValueError("los_mode must be 'radial' or 'axis' (or 'x', 'y', 'z'), got %r" % (los_mode,))

def los_unit_vectors(positions, los_mode='radial', los_location=None, los_axis=None):
    '''Unit LOS vector (pointing away from the observer) at each position. positions: shape (n, 3) or (3,). returns: shape (n, 3)'''
    los = resolve_los(los_mode, los_location, los_axis)
    positions = np.atleast_2d(positions)
    if los['los_mode'] == 'axis':
        n_hat = np.zeros(positions.shape)
        n_hat[:, los['los_axis']] = 1.0
        return n_hat
    v = positions - los['los_location']
    return v / np.linalg.norm(v, axis=1)[:, np.newaxis]

def sky_components(vectors, positions, los_mode='radial', los_location=None, los_axis=None):
    '''
    Project 3D vectors onto the sky plane (the plane perpendicular to the LOS at `positions`) and
    return their (east, north) components. This is the single definition of sky-plane axes used for
    every projected angle (multiplet shapes and position angles), so anything measured at the same
    position is measured in the same plane, relative to the same North.

    vectors: shape (n, 3)
    positions: shape (n, 3), or (3,) to use one position for all vectors
    los_mode='axis':   east, north = the two axes after los_axis, cyclically
                       ('z': east=+x, north=+y; 'x': east=+y, north=+z; 'y': east=+z, north=+x).
    los_mode='radial': north = projection of +z onto the sky plane (+x if the LOS is along z), east = north x LOS.
                       With the observer at the origin this is the RA/DEC convention (east = increasing RA).
    returns: east, north, each shape (n,)
    '''
    los = resolve_los(los_mode, los_location, los_axis)
    vectors = np.atleast_2d(vectors)
    if los['los_mode'] == 'axis':
        a = los['los_axis']
        return vectors[:, (a + 1) % 3], vectors[:, (a + 2) % 3]

    n_hat = los_unit_vectors(positions, **los)
    ref = np.zeros_like(n_hat)
    ref[:, 2] = 1.0
    ref[1.0 - n_hat[:, 2]**2 < 1e-12] = [1.0, 0.0, 0.0]    # +z has no sky-plane projection if the LOS is along z
    north = ref - np.sum(ref * n_hat, axis=1)[:, np.newaxis] * n_hat
    north /= np.linalg.norm(north, axis=1)[:, np.newaxis]
    east = np.cross(north, n_hat)
    return np.sum(vectors * east, axis=1), np.sum(vectors * north, axis=1)

def los_pair_separations(pos1, pos2, los_mode='radial', los_location=None, los_axis=None):
    '''
    Split the separation of pos2 from pos1 (each shape (n, 3)) into transverse and LOS parts.
    los_mode='axis':   r_par = component along los_axis; r_p = the other two components in quadrature.
    los_mode='radial': r_par = difference in distance from the observer;
                       r_p = separation transverse to the LOS through the pair midpoint (see get_proj_dist).
    returns: r_p (>= 0), r_par (signed; > 0 when pos2 is farther from the observer than pos1)
    '''
    los = resolve_los(los_mode, los_location, los_axis)
    if los['los_mode'] == 'axis':
        r_par = pos2[:, los['los_axis']] - pos1[:, los['los_axis']]
    else:
        r_par = np.sqrt(np.sum((pos2 - los['los_location'])**2, axis=1)) - np.sqrt(np.sum((pos1 - los['los_location'])**2, axis=1))
    r_p = get_proj_dist(pos1, pos2, pos_obs=los['los_location'], los_mode=los['los_mode'], los_axis=los['los_axis'])
    return r_p, r_par

def get_proj_dist(pos1, pos2, pos_obs=np.asarray([0, 0, 0]) * .7, use_cat=False, los_mode='radial', los_axis=None):
    '''
    Return transverse projected distance of two positions. Returns in same units as given.
    los_mode='radial': transverse to the LOS from pos_obs (the observer) to the pair midpoint.
    los_mode='axis' (with los_axis), or 'x'/'y'/'z': transverse to that axis. See resolve_los().
    '''
    los = resolve_los(los_mode, pos_obs, los_axis)
    if use_cat:
        pos1 = pos1['x_L2com']
        pos2 = pos2['x_L2com']

    dx = pos2[:, 0] - pos1[:, 0]
    dy = pos2[:, 1] - pos1[:, 1]
    dz = pos2[:, 2] - pos1[:, 2]

    if los['los_mode'] == 'axis':
        # plane-parallel LOS: transverse separation is just the two components perpendicular to the axis
        d = (dx, dy, dz)
        a = los['los_axis']
        return np.sqrt(d[(a + 1) % 3] * d[(a + 1) % 3] + d[(a + 2) % 3] * d[(a + 2) % 3])
    else:
        pos_obs = los['los_location']
        d2 = dx * dx + dy * dy + dz * dz
        ox = 0.5 * (pos2[:, 0] + pos1[:, 0]) - pos_obs[0]
        oy = 0.5 * (pos2[:, 1] + pos1[:, 1]) - pos_obs[1]
        oz = 0.5 * (pos2[:, 2] + pos1[:, 2]) - pos_obs[2]
        onorm2 = ox * ox + oy * oy + oz * oz
        dot = dx * ox + dy * oy + dz * oz
        parallel2 = (dot * dot) / onorm2
        perp2 = d2 - parallel2
        np.maximum(perp2, 0.0, out=perp2)
        return np.sqrt(perp2)

############
# CARTESIAN FUNCTIONS
############

def project_points_onto_plane(points, plane_normal):
    '''
    project a set of 3d points onto a plane
    return a set of 2d vectors, where the orgin is the intersection of the plane and the LOS
    '''
    proj = np.sum(points*plane_normal) / np.linalg.norm(plane_normal, axis=1)
    proj_v = (proj[:, np.newaxis] * plane_normal) / np.linalg.norm(plane_normal, axis=1)[:, np.newaxis]
    
    # find the 2D vector in the plane perpendicular to the los
    group_points_in_plane = points - proj_v
    
    return group_points_in_plane

def get_points_in_plane(group_points, los_location=np.asarray([0,0,0]), n_groups=1):
    '''
    get the orientation of a group of points projected onto a plane perpendicular to the LOS
    group_points: array of shape (n_points, 3)
    los_location: array of shape (3,)
    return: array of shape (n_points, 2)
    return the normalized 2D vector representing the group's orientation relative to "North"
    "North" (or y-axis) is assumed to be the projection of the z-axis onto the plane of the sky
    '''
    # get LOS vector
    group_center = np.mean(group_points, axis=0)
    los_vec = (group_center - los_location).reshape(n_groups,3)
    
    group_points -= los_location  # just in case los_location is not the origin
    
    # project points onto plane perpendicular to los
    plane_y = project_points_onto_plane(np.asarray([[0, 0, 1]]), los_vec) # project original z-axis onto plane of the sky
    plane_x = np.cross(plane_y, los_vec)
    
    # normalize
    plane_y /= np.linalg.norm(plane_y)
    plane_x /= np.linalg.norm(plane_x)
    
    # find the 2D coordinates of the points in the plane
    group_points_in_plane_x = np.sum(group_points*plane_x, axis=1)
    group_points_in_plane_y = np.sum(group_points*plane_y, axis=1)
    group_points_in_plane = np.asarray([group_points_in_plane_x, group_points_in_plane_y]).T
    
    return group_points_in_plane

def get_orientation_angle_cartesian(points1, points2, los_location=None, los_mode='radial', los_axis=None):
    '''
    Position angle of points1 relative to points2, measured East of North in the sky plane at the pair
    midpoint.
    Both directions of a pair are measured in the same plane, so swapping points1 and points2 adds exactly pi.
    (Shapes, from calculate_2D_group_orientation, are measured E of N in their own sky plane.)

    Known limitation (radial LOS): a shape angle (North of the shape's plane) minus this angle (North of the
    midpoint plane) mixes two different Norths. Where the LOS is close to +/-z (North = projection of +z) they
    can differ a lot, e.g. an observer far along -z of a box does not reproduce los_axis='z' (outer bins off by
    ~30% in tests). An observer along x or y is unaffected. Fix if needed: rotate the shape axis into the
    midpoint plane (rotation taking the shape's LOS to the midpoint LOS) before comparing.

    points1, points2: arrays of shape (n_points, 3)
    los_mode, los_location, los_axis: line of sight, see resolve_los(). Default: radial, observer at the origin.
    returns: array of shape (n_points,)
    '''
    los = resolve_los(los_mode, los_location, los_axis)
    east, north = sky_components(points2 - points1, 0.5 * (points1 + points2), **los)
    if los['los_mode'] == 'radial':
        return np.arctan2(-east, -north)
    return np.arctan2(east, north)


def projected_separation_ra_dec(ra1, dec1, x1, y1, z1, ra2, dec2, x2, y2, z2): 
    '''
    Calculate the projected physical separation between two points on the sky, given cartesian positions and their RA / DEC.
    Returns physical separation in same units as input cartesian positions.
    '''
    
    # Convert RA and DEC from degrees to radians
    ra1_rad = np.deg2rad(ra1)
    dec1_rad = np.deg2rad(dec1)
    ra2_rad = np.deg2rad(ra2)
    dec2_rad = np.deg2rad(dec2)

    # Calculate the unit vectors for the two points
    unit_vector1 = np.array([np.cos(ra1_rad) * np.cos(dec1_rad), np.sin(ra1_rad) * np.cos(dec1_rad), np.sin(dec1_rad)])
    unit_vector2 = np.array([np.cos(ra2_rad) * np.cos(dec2_rad), np.sin(ra2_rad) * np.cos(dec2_rad), np.sin(dec2_rad)])

    # Calculate the Cartesian separation along the line of sight (LOS)
    delta_x = x2 - x1
    delta_y = y2 - y1
    delta_z = z2 - z1

    # Calculate the projected separation in the plane of the sky
    projected_separation = np.sqrt((delta_x - (np.dot([delta_x, delta_y, delta_z], unit_vector1) * unit_vector1[0]))**2 + \
                                   (delta_y - (np.dot([delta_x, delta_y, delta_z], unit_vector1) * unit_vector1[1]))**2)

    return projected_separation

def get_pair_distances(catalog, indices, pos_obs=np.asarray([-3700, 0, 0])*.7, cartesian=False):
    '''pos_obs in Mpc/h'''
    # indices in catalog of centers and neighbors, arranges so each array is same shape
    ci = np.repeat(indices[:,0], (len(indices[0])-1)).ravel() # indices of centers
    ni = indices[:,1:].ravel()   # indices of neighbors
    
    # removing places where no neighbor was found in the tree
    neighbor_exists = (ni!=len(catalog))
    ci = ci[neighbor_exists]; ni = ni[neighbor_exists]
    
    centers_m = catalog[ci]
    neighbors_m = catalog[ni]   # excluding the centers
    
    #r_projected = projected_separation_ra_dec(centers_m['RA'], centers_m['DEC'], centers_m['x_L2com'][::,0], centers_m['x_L2com'][::,1], centers_m['x_L2com'][::,2], 
    #                                          neighbors_m['RA'], neighbors_m['DEC'], neighbors_m['x_L2com'][::,0], neighbors_m['x_L2com'][::,1], neighbors_m['x_L2com'][::,2])
    if cartesian==False:
        
        r_parallel = (np.abs(cosmo.comoving_distance(centers_m['Z_noRSD']) - cosmo.comoving_distance(neighbors_m['Z_noRSD'])) * 0.7 / u.Mpc).value
        s_parallel = (np.abs(cosmo.comoving_distance(centers_m['Z_withRSD']) - cosmo.comoving_distance(neighbors_m['Z_withRSD'])) * 0.7 / u.Mpc).value
        
        r_projected = get_proj_dist(centers_m, neighbors_m, pos_obs, use_cat=True)

        return r_projected, r_parallel, s_parallel
    
    elif cartesian==True:
        deltax = np.abs(centers_m['x_L2com'][::,0] - neighbors_m['x_L2com'][::,0])
        deltayz = np.sqrt((centers_m['x_L2com'][::,1] - neighbors_m['x_L2com'][::,1])**2 + (centers_m['x_L2com'][::,2] - neighbors_m['x_L2com'][::,2])**2)
        return deltax, deltayz
    
    
    # function to estimate the 3D volume of a set of points
    from scipy.spatial import ConvexHull
    def est_volume(points_3D, return_units=(u.Mpc**3 / u.h**3)):
        '''input: 3D points in Mpc/h. Returns astropy quantity of volume'''
        hull = ConvexHull(points_3D)
        volume = hull.volume * (u.Mpc/u.h)**3
        return volume.to(return_units)
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

def get_cosmo_points(data, cosmology=cosmo):
    '''convert from astropy table of RA, DEC, and redshift to 3D cartesian coordinates in Mpc/h'''
    comoving_dist = cosmo.comoving_distance(data['Z']).to(u.Mpc)
    points = coordinates.spherical_to_cartesian(np.abs(comoving_dist), np.asarray(data['DEC'])*u.deg, np.asarray(data['RA'])*u.deg)
    return np.asarray(points).transpose() * .7

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


def get_proj_dist(pos1, pos2, pos_obs=np.asarray([0, 0, 0])*.7, use_cat=False):
    '''return transverse projected distance of two positions given observer position. returns in same units as given. default is Mpc/h'''
    
    if use_cat==False:
        pos_diff = pos2 - pos1
        pos_mid = .5 * (pos2 + pos1)
    elif use_cat==True:
        pos_diff = pos2['x_L2com'] - pos1['x_L2com']
        pos_mid = .5 * (pos2['x_L2com'] + pos1['x_L2com'])
    obs_vec = pos_mid - pos_obs
    
    # project separation vector between objects onto LOS vector
    proj = np.sum(pos_diff*obs_vec, axis=1) / np.linalg.norm(obs_vec, axis=1)
    proj_v = (proj[:, np.newaxis] * obs_vec) / np.linalg.norm(obs_vec, axis=1)[:, np.newaxis]

    # subtract this vector from the separation vector
    # magnitude is projected transverse distance
    transverse_v = pos_diff - proj_v
    return np.linalg.norm(transverse_v, axis=1)

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

def get_orientation_angle_cartesian(points1, points2, los_location=np.asarray([0,0,0])):
    '''
    get the orientation of a points1 relative to points2 projected onto a plane perpendicular to the LOS
    points1: array of shape (n_points, 3)
    points2: array of shape (n_points, 3)
    return: array of shape (n_points,)
    return the orientation relative to "North"
    "North" (or y-axis) is assumed to be the projection of the z-axis onto the plane of the sky
    '''
    # find the LOS vector
    los_vector = (points1 + points2)/2 - los_location
    los_vector = los_vector / np.linalg.norm(los_vector, axis=1)[:, None]
    # find the vector perpendicular to the LOS vector and the z-axis
    perp_vector = np.cross(los_vector, np.asarray([0,0,1]))
    perp_vector = perp_vector / np.linalg.norm(perp_vector, axis=1)[:, None]
    # find the vector perpendicular to the LOS vector and the perp_vector
    perp_vector2 = np.cross(los_vector, perp_vector)
    perp_vector2 = perp_vector2 / np.linalg.norm(perp_vector2, axis=1)[:, None]
    
    # find the 2d projection of points onto the plane perpendicular to the LOS
    points1_proj = np.asarray([np.sum(points1*perp_vector, axis=1), np.sum(points1*perp_vector2, axis=1)]).T
    points2_proj = np.asarray([np.sum(points2*perp_vector, axis=1), np.sum(points2*perp_vector2, axis=1)]).T
    
    return np.arctan2((points2_proj[:,0]-points1_proj[:,0]), (points2_proj[:,1]-points1_proj[:,1]))


def projected_separation_ra_dec(ra1, dec1, x1, y1, z1, ra2, dec2, x2, y2, z2):  # from chatgpt...
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
import numpy as np
from astropy.cosmology import LambdaCDM, z_at_value
cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from geometry_functions.coordinate_functions import *

import time, glob

from scipy.spatial import cKDTree
from scipy import stats
# useful info at http://legacysurvey.org/dr8/catalogs/

#################################################################
# FUNCTIONS FOR MEASURING SEPARATIONS

def projected_separation_ra_dec(ra1, dec1, x1, y1, z1, ra2, dec2, x2, y2, z2):  
    '''
    Returns the projected separation between two cartesian points, projected along the LOS (which is assumed to be the x-axis)
    Input:
        ra1, dec1: RA and DEC of first point, in degrees
        x1, y1, z1: cartesian coordinates of first point
        ra2, dec2: RA and DEC of second point, in degrees
        x2, y2, z2: cartesian coordinates of second point
    Returns:
        projected_separation: projected separation between the two points, in input coordinates (default Mpc/h)
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

def get_proj_dist_abacus(cat1, cat2, pos_obs=np.asarray([-3700, 0, 0])*.7):
    '''
    Returns transverse projected distance of two cartesian positions given observer position. 
    Input:
        cat1: astropy table with keys 'x_L2com', 'y_L2com', 'z_L2com'
        cat2: astropy table with keys 'x_L2com', 'y_L2com', 'z_L2com'
        pos_obs: cartesian position of observer. default is [-3700, 0, 0] Mpc/h
    returns in same units as given. default is Mpc/h
    '''
    pos_diff = cat2['x_L2com'] - cat1['x_L2com']
    pos_mid = .5 * (cat2['x_L2com'] + cat1['x_L2com'])
    obs_vec = pos_mid - pos_obs
    
    # project separation vector between objects onto LOS vector
    proj = np.sum(pos_diff*obs_vec, axis=1) / np.linalg.norm(obs_vec, axis=1)
    proj_v = (proj[:, np.newaxis] * obs_vec) / np.linalg.norm(obs_vec, axis=1)[:, np.newaxis]

    # subtract this vector from the separation vector
    # magnitude is projected transverse distance
    transverse_v = pos_diff - proj_v
    return np.linalg.norm(transverse_v, axis=1)


def get_pair_distances(catalog, indices, pos_obs=np.asarray([-3700, 0, 0])*.7, cartesian=False):
    '''
    Calculate distances between input pairs, coordinates in Mpc/h
    -------------------------------------------------------------
    Input:
        catalog: astropy table with keys 'x_L2com', 'y_L2com', 'z_L2com'
        indices: array of indices for n centers and maximum m neighbors each- shape(n,m)
        corresponding to place in catalog
    Returns (for cartesian=False):        
        r_projected: projected distance between centers and neighbors
        r_parallel: LOS distance between centers and neighbors
        s_parallel: LOS distance between centers and neighbors, with RSD
    Returns (for cartesian=True):
        deltax: x distance between centers and neighbors    
        deltayz: yz distance between centers and neighbors
    '''
    # indices in catalog of centers and neighbors, arranges so each array is same shape
    ci = np.repeat(indices[:,0], (len(indices[0])-1)).ravel() # indices of centers
    ni = indices[:,1:].ravel()   # indices of neighbors
    
    # removing places where no neighbor was found in the tree
    neighbor_exists = (ni!=len(catalog))
    ci = ci[neighbor_exists]; ni = ni[neighbor_exists]
    
    centers_m = catalog[ci]
    neighbors_m = catalog[ni]   # excluding the centers
    
    if cartesian==False:
        
        r_parallel = (np.abs(cosmo.comoving_distance(centers_m['Z_noRSD']) - cosmo.comoving_distance(neighbors_m['Z_noRSD'])) * 0.7 / u.Mpc).value
        s_parallel = (np.abs(cosmo.comoving_distance(centers_m['Z_withRSD']) - cosmo.comoving_distance(neighbors_m['Z_withRSD'])) * 0.7 / u.Mpc).value
        
        r_projected = get_proj_dist(centers_m, neighbors_m, pos_obs)

        return r_projected, r_parallel, s_parallel
    
    elif cartesian==True:
        deltax = np.abs(centers_m['x_L2com'][::,0] - neighbors_m['x_L2com'][::,0])
        deltayz = np.sqrt((centers_m['x_L2com'][::,1] - neighbors_m['x_L2com'][::,1])**2 + (centers_m['x_L2com'][::,2] - neighbors_m['x_L2com'][::,2])**2)
        return deltax, deltayz
    
    
def cylindrical_cut(pair_table, rp_max=20, rpar_max=30):
    '''limits pairs to those within rp_max and rpar_max'''
    return pair_table[((pair_table['r_p']<20)&(pair_table['r_par']<30))] #pair_table[((pair_table['r_p']<20)&(pair_table['r_par']<30))]

#################################################################
# SHAPE ALIGNMENT FUNCTIONS

def get_galaxy_orientation_angle(e1, e2):
    '''return orientation angle of galaxy (range of 0-pi)'''
    return 0.5 * np.arctan2(e2, e1)

def abs_e(e1, e2):
    '''absolute value of complex ellipticity'''
    return np.sqrt(e1*e1 + e2*e2)

def e_complex(a, b, theta):
    '''complex ellipticity, theta must be in rad'''
    abs_e = (1 - (b/a)) / (1 + (b/a))
    e1 = abs_e * np.cos(2*theta)
    e2 = abs_e * np.sin(2*theta)
    return e1, e2

def a_b(e1, e2):
    '''return a and b of ellipse'''
    e = abs_e(e1, e2)
    return 1+e, 1-e  

def get_rel_es(catalog, indices, data_weights=None, weights=None, rcolor='rw1', return_sep=False, j=0):
    '''
    input: 
        array of indices for n centers and maximum m neighbors each- shape(n,m)
        corresponding to place in catalog
        first element of each row is indic of center
        shape can be 'ser', 'dev', or 'exp' for fit used to get ellipticity components
        data_weights must be same length as catalog, with each indice corresponding to right row
    returns: 
        array of same shape, containing ellipticities relative to separation
        vector between given neighbor and it's central galaxy
    '''
    
    # indices in catalog of centers and neighbors, arranges so each array is same shape
    ci = np.repeat(indices[:,0], (len(indices[0])-1)).ravel() # indices of centers
    ni = indices[:,1:].ravel()   # indices of neighbors
    
    # removing places where no neighbor was found in the tree
    neighbor_exists = (ni!=len(catalog))
    ci = ci[neighbor_exists]; ni = ni[neighbor_exists]
    
    centers_m = catalog[ci]
    neighbors_m = catalog[ni]   # excluding the centers
    
    # get position angle
    pa = get_pa(centers_m['RA'], centers_m['DEC'], neighbors_m['RA'], neighbors_m['DEC'])
    
    
    # calculate rotation angle of neighbor relative to the separation vector
    theta_neighbor = get_galaxy_orientation_angle(neighbors_m['E1'], neighbors_m['E2'])
    a, b = a_b(neighbors_m['E1'], neighbors_m['E2'])
    
    pa_rel = theta_neighbor - pa.value  # in rad
    e1_re, e2_rel = e_complex(a, b, pa_rel)
    
    # TEMP - TO GET Z DISTRIBUTION
    ##z_binned = plt.hist(centers_m['Z'], bins=np.linspace(0, 1.4, 100))
    ##np.savetxt('/pscratch/sd/c/clamman/ia_measurements/LRG_ELG_cut_z/'+str(j)+'.csv', z_binned[0], delimiter=',')
    
    if (weights is None) and (data_weights is None):
        return e1_re, e2_rel, None
    
    elif data_weights is not None:
        # combining weights of centers and neighbors
        all_ws = centers_m['WEIGHT_SYS'] * centers_m['WEIGHT_ZFAIL'] * neighbors_m['WEIGHT_SYS'] * neighbors_m['WEIGHT_ZFAIL']
        return e1_re, e2_rel, all_ws
    
    
    
def calculate_rel_ang_cartesian(ang_tracers, ang_values, loc_tracers, pimax = 20, max_proj_sep = 30, max_neighbors=100):
    '''ang and loc tracers are 3d points. ang_values are orientation angles of ang_tracers'''
    # make tree
    tree = cKDTree(loc_tracers)
    # find neighbors
    dd, ii = tree.query(ang_tracers, k=max_neighbors, distance_upper_bound=np.sqrt(max_proj_sep**2 + pimax**2))
    
    # add placeholder row to loc_tracers
    loc_tracers = np.vstack((loc_tracers, np.full(len(loc_tracers[0]), np.inf)))
    
    center_coords = ang_tracers[np.repeat(range(len(ang_tracers)), max_neighbors).ravel()]
    center_angles = ang_values[np.repeat(range(len(ang_tracers)), max_neighbors).ravel()]
    neighbor_coords = loc_tracers[ii.ravel()]
    
    dist_to_orgin_loc = np.sqrt(np.sum(neighbor_coords**2, axis=1))
    dist_to_orgin_ang = np.sqrt(np.sum(center_coords**2, axis=1))
    los_sep = np.abs(dist_to_orgin_loc - dist_to_orgin_ang)
    
    proj_dist = np.abs(get_proj_dist(center_coords, neighbor_coords))
    
    pairs_to_keep = (los_sep < pimax) & (proj_dist < max_proj_sep)
    center_coords = center_coords[pairs_to_keep]
    neighbor_coords = neighbor_coords[pairs_to_keep]
    
    # calculate projected position angle
    position_angle = get_orientation_angle_cartesian(center_coords, neighbor_coords)
    
    pa_rel = center_angles[pairs_to_keep] - position_angle
    
    
    return proj_dist[pairs_to_keep], pa_rel

################################################################
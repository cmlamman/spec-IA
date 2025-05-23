import numpy as np
from astropy.cosmology import LambdaCDM, z_at_value
cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from geometry_functions.coordinate_functions import *
from alignment_functions.general_stats import bin_sum_not_scipy, bin_results

import time, glob, os

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


def get_pair_distances(catalog, indices, pos_obs=np.asarray([-3700, 0, 0])*.7, cartesian=False, use_RSD=True):
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
        r_projected = get_proj_dist_abacus(centers_m, neighbors_m, pos_obs)

        if use_RSD == False:
            r_parallel = (np.abs(cosmo.comoving_distance(centers_m['Z']) - cosmo.comoving_distance(neighbors_m['Z'])) * 0.7 / u.Mpc).value
            return r_projected, r_parallel
        else:
            r_parallel = (np.abs(cosmo.comoving_distance(centers_m['Z_noRSD']) - cosmo.comoving_distance(neighbors_m['Z_noRSD'])) * 0.7 / u.Mpc).value
            s_parallel = (np.abs(cosmo.comoving_distance(centers_m['Z_withRSD']) - cosmo.comoving_distance(neighbors_m['Z_withRSD'])) * 0.7 / u.Mpc).value
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

def esq_to_e(e1_sq, e2_sq):
    '''convert from definition of ellipticity which used (b/a)^2 to (b/a)'''
    abs_sq = abs_e(e1_sq, e2_sq)
    b_a = np.sqrt((1 - abs_sq) / (1 + abs_sq))
    mult_factor = (b_a**2 + 1) / (b_a+1)**2
    return e1_sq * mult_factor, e2_sq * mult_factor

def e_to_esq(e1, e2):
    '''convert from definition of ellipticity which used (b/a) to (b/a)^2'''
    abs_ensq = abs_e(e1, e2)
    b_a = (1 - abs_ensq) / (1 + abs_ensq)
    mult_factor =  (b_a+1)**2 / (b_a**2 + 1)
    return e1 * mult_factor, e2 * mult_factor 

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
        if return_sep==True:
            psep = get_psep(centers_m['RA'], centers_m['DEC'], neighbors_m['RA'], neighbors_m['DEC'], u_coords='deg', u_result=u.deg)
            psep_mpc = get_Mpc_h(psep, centers_m['Z'])  # in units of Mpc / h
            return e1_re, e2_rel, all_ws, psep_mpc
        elif return_sep==False:
            return e1_re, e2_rel, all_ws

def calculate_rel_ang_cartesian(ang_tracers, ang_values, loc_tracers, abs_e=None, loc_weights=None, pimax = 20, max_proj_sep = 30, max_neighbors=100, 
                                return_los=False, print_progress=False, tracer_behind=False, keep_as_angle=True):
    '''ang and loc tracers are 3d points. ang_values are orientation angles of ang_tracers 
    - pimax (float, optional): Maximum line-of-sight separation for pairs of galaxies in Mpc/h. Default is 30.
    - abs_e (array, optional): can supply absolute value of ellipticity to get relative full ellipticity. If supplied, will automatically return (abs_e)*cos(2*angle)
    - keep_as_angle : if True, returns angle only. If False, returns the relevant IA signal (cos(2*angle))
    '''
    # make tree
    tree = cKDTree(loc_tracers)
    # find neighbors
    ii = tree.query_ball_point(ang_tracers, r=np.sqrt(max_proj_sep**2 + pimax**2))
    
    # add placeholder row to loc_tracers
    loc_tracers = np.vstack((loc_tracers, np.full(len(loc_tracers[0]), np.inf)))
    if loc_weights is not None:
        loc_weights = np.append(loc_weights, 0)
    
    center_coords = np.repeat(ang_tracers, [len(i) for i in ii], axis=0)
    center_angles = np.repeat(ang_values, [len(i) for i in ii])
    neighbor_coords = loc_tracers[np.concatenate(ii)]
    if abs_e is not None:
        center_abs_e = np.repeat(abs_e, [len(i) for i in ii])
    
    # breaking it up to save memory
    nc = neighbor_coords**2
    nc = np.sum(nc, axis=1)
    dist_to_orgin_loc = np.sqrt(nc)
    del nc
    cc = center_coords**2
    cc = np.sum(cc, axis=1)
    dist_to_orgin_ang = np.sqrt(cc)
    del cc
    
    if print_progress: print('calculating separations')
    dist_to_orgin_loc -= dist_to_orgin_ang
    los_sep = np.abs(dist_to_orgin_loc)
    proj_dist = np.abs(get_proj_dist(center_coords, neighbor_coords))
    proj_dist = np.abs(proj_dist)
    
    pairs_to_keep = (los_sep < pimax) & (proj_dist < max_proj_sep)
    center_coords = center_coords[pairs_to_keep]
    neighbor_coords = neighbor_coords[pairs_to_keep]
    
    # calculate projected position angle
    position_angle = get_orientation_angle_cartesian(center_coords, neighbor_coords)
    
    pa_rel = center_angles[pairs_to_keep] - position_angle
    if abs_e is not None:
        pa_rel = center_abs_e[pairs_to_keep] * np.cos(2 * np.asarray((pa_rel)))
    elif keep_as_angle == False:
        pa_rel = np.cos(2* np.asarray((pa_rel)))
    
    if loc_weights is None and return_los==False:
        return proj_dist[pairs_to_keep], pa_rel
    elif loc_weights is None and return_los==True:
        return proj_dist[pairs_to_keep], pa_rel, los_sep[pairs_to_keep]
    
    elif loc_weights is not None and return_los==False:
        return proj_dist[pairs_to_keep], pa_rel, loc_weights[np.concatenate(ii)][pairs_to_keep]
    elif loc_weights is not None and return_los==True:
        return proj_dist[pairs_to_keep], pa_rel, loc_weights[np.concatenate(ii)][pairs_to_keep], los_sep[pairs_to_keep]


# calculate relative angles in seprate regions and returned binned results

def rel_angle_regions(group_info, loc_tracers, tracer_weights=None, n_regions = 100, pimax = 20, max_proj_sep = 30, max_neighbors=100, return_los=False):
    '''
    divide the angle catalog into n_regions by RA and DEC, calculate cos(2*theta) the angles relative to the tracers, and return the results from each region
    group_info: must contain central carteisan postion, angle, and RA/DEC of groups
    loc_tracers: array of 3d points corresponding to the tracers 
    - pimax (float, optional) or (list, length of n_Rbins): Maximum line-of-sight separation for pairs of galaxies in Mpc/h. Default is 20.
    '''
    
    # sort ang_tracers and ang_values by DEC
    dec_sorter = np.argsort(group_info['DEC'])
    
    # divide catalogs into regions
    n_slices = int(np.sqrt(n_regions))
    n_in_dec_slice = int((len(group_info) / n_slices)+1)
    j = 0 
    k = n_in_dec_slice
    
    all_proj_dists = []
    all_pa_rels = []
    all_weights = []
    all_los_seps = []
    for _ in range(n_slices):   # loop over DEC slices
        
        # for handling the last region
        if k > len(group_info):     
            k = len(group_info)
        if len(group_info[j:k]) == 0:
            continue
        
        # slice in DEC
        groups_dec_slice = group_info[dec_sorter[j:k]]
        
        # sort this slice by RA
        ra_sorter = np.argsort(groups_dec_slice['RA'])
        n_in_ra_slice = int((len(groups_dec_slice) / n_slices)+1)
        n = 0
        m = n_in_ra_slice
        
        for _ in range(n_slices):       # loop over RA regions in this DEC slice
            
            # for handling the last region
            if m > len(groups_dec_slice):     
                m = len(groups_dec_slice)
            if len(groups_dec_slice[n:m]) == 0:
                continue
            
            group_square = groups_dec_slice[ra_sorter[n:m]]
            
            if return_los==False:
                proj_dist, pa_rel, weights = calculate_rel_ang_cartesian(group_square['center_loc'], group_square['orientation'], loc_tracers, loc_weights=tracer_weights, 
                                                                         pimax=pimax, max_proj_sep = max_proj_sep, max_neighbors=max_neighbors, return_los=False)
            elif return_los==True:
                proj_dist, pa_rel, weights, los_seps = calculate_rel_ang_cartesian(group_square['center_loc'], group_square['orientation'], loc_tracers, loc_weights=tracer_weights, 
                                                                         pimax=pimax, max_proj_sep = max_proj_sep, max_neighbors=max_neighbors, return_los=return_los)
                all_los_seps.append(los_seps)
                
            all_proj_dists.append(proj_dist)
            all_pa_rels.append(np.cos(2*pa_rel))
            all_weights.append(weights)
            
            n += n_in_ra_slice
            m += n_in_ra_slice
        
        
        j += n_in_dec_slice
        k += n_in_dec_slice
    if return_los==False:
        return all_proj_dists, all_pa_rels, all_weights
    elif return_los==True:
        return all_proj_dists, all_pa_rels, all_weights, all_los_seps

def sliding_pimax(r_sep):
    return 6 + (2/3)*r_sep

def bin_region_results(all_proj_dists, all_pa_rels, all_weights=None, R_bins=np.logspace(0, 2, 11), use_sliding_pimax=False, los_sep=None):
    '''
    bin the results from rel_angle_regions
    R_bins: bin edges for the projected separation, in Mpc/h
    if use_sliding_pimax is True, pairs are limited to a pimax of 10 + (2/3)*proj_dists, required los_sep
    '''
    
    sep_bins = R_bins
    
    all_binned_pa_rels = []
    for i in range(len(all_proj_dists)):
        
        dist_to_bin = all_proj_dists[i]
        pa_rel_to_bin = all_pa_rels[i]
        if all_weights is not None:
            weights_to_bin = all_weights[i]
        else:
            weights_to_bin = None
            
        if use_sliding_pimax==True:
            # pairs to keep. Start by removing ones that fall outside the lowest bin. The following is messed up if you include these.
            i_keep = dist_to_bin > np.min(sep_bins)
            i_keep &= dist_to_bin < np.max(sep_bins)                                                 
            sep_bins_middle = (sep_bins[1:] + sep_bins[:-1]) / 2                                        # middle of bins  
            sep_bins_middle = np.append(sep_bins_middle, np.nan)                                        # add placeholder for pairs that fall outside the bins (won't be used)                                                              
            binned_dist = sep_bins_middle[(np.digitize(dist_to_bin, bins=sep_bins, right=True)-1)]      # replace each distance with the middle of the bin it falls in
            i_keep &= los_sep[i] < sliding_pimax(binned_dist)                                              # keep only pairs within the pimax for that bin
            
            # limit all pairs to these requirements
            dist_to_bin = dist_to_bin[i_keep]
            pa_rel_to_bin = pa_rel_to_bin[i_keep]
            if all_weights is not None:
                weights_to_bin = weights_to_bin[i_keep]
        
        binned_seps, binned_pa_rels = bin_results(dist_to_bin, pa_rel_to_bin, sep_bins, weights=weights_to_bin)
        all_binned_pa_rels.append(binned_pa_rels)

    pa_rel_av = np.nanmean(all_binned_pa_rels, axis=0)
    pa_rel_e = np.nanstd(all_binned_pa_rels, axis=0) / np.sqrt(len(all_binned_pa_rels))
    return sep_bins, pa_rel_av, pa_rel_e
    

################################################################
# ALTERNATIVE HIGH-LEVEL FUNCTIONS WHICH BIN EARLIER TO SAVE MEMORY
# (necessary for very dense catalogs, like DESI BGS)
################################################################

def process_in_batches(func, array, batch_size):
    result = []
    for i in range(0, len(array), batch_size):
        batch = array[i:i+batch_size]
        result.append(func(batch))
    return np.concatenate(result)

def square_sum(coords):
    coords = coords**2
    coords = np.sum(coords, axis=1)
    return np.sqrt(coords)

def calculate_rel_ang_cartesian_binAverage(ang_tracers, ang_values, loc_tracers, loc_weights, R_bins, pimax, E_ABS, print_progress=False, tracer_behind=False):
    '''
    With especially dense regions, memory becomes an issue as the number of group-tracer matches drastically increse. 
    This function is a memory-sensitive version of calculate_rel_ang_cartesian, which bins the projected distances earlier and runs several functions in batches.
    It returns the average relative angles in each R bin, array of size (len(R_bins)-1)
    
    Parameters
    ----------
    ang_tracers : array, shape (n, 3)
        3d points of angle tracers
    ang_values : array, shape (n)
        orientation angles of angle tracers. Must be same length as ang_tracers.
    E_ABS : array, shape (n)
        Absolute value of ellipticity for measuring full-shape alignment. For the "stick" model, just use 1.
    loc_tracers : array, shape (m, 3)
        3d points of locations
    loc_weights : array, shape (m)
        weights of locations. Must be same length as loc_tracers.
    R_bins : array
        bin edges of projected separation, in Mpc/h
    pimax : float or array
        Maximum line-of-sight separation for pairs of galaxies in Mpc/h. Default is 30.
        Can also be array of size (R_bins-1) to use a different pimax for each R bin

    '''
    if print_progress: print('making tree')
    # make tree
    tree = cKDTree(loc_tracers)
    # find neighbors
    if print_progress: print('finding neighbors')
    ii = tree.query_ball_point(ang_tracers, r=np.sqrt(R_bins[-1]**2 + np.max(pimax)**2))
    if print_progress: print('found neighbors')
    # add placeholder row to loc_tracers
    loc_tracers = np.vstack((loc_tracers, np.full(len(loc_tracers[0]), np.inf)))
    loc_weights = np.append(loc_weights, 0)
    
    center_coords = np.repeat(ang_tracers, [len(i) for i in ii], axis=0)
    center_angles = np.repeat(ang_values, [len(i) for i in ii])
    center_E = np.repeat(E_ABS, [len(i) for i in ii])
    neighbor_coords = loc_tracers[np.concatenate(ii).astype(int)]
    neighbor_weights = loc_weights[np.concatenate(ii).astype(int)]
    
    if print_progress: print('calculating distances')
    
    # breaking it up to save memory
    dist_to_orgin_loc = process_in_batches(lambda x: square_sum(neighbor_coords[x]), np.arange(len(center_coords)), 100000)
    dist_to_orgin_ang = process_in_batches(lambda x: square_sum(center_coords[x]), np.arange(len(center_coords)), 100000)
    
    if print_progress: print('calculating separations')
    dist_to_orgin_loc -= dist_to_orgin_ang
    los_sep = np.abs(dist_to_orgin_loc)
    if print_progress: print('calculating proj dist in batches')
    proj_dist = process_in_batches(lambda x: get_proj_dist(center_coords[x], neighbor_coords[x]), np.arange(len(center_coords)), 100000)
    proj_dist = np.abs(proj_dist)
    
    if print_progress: print('binning')
    # bin the projected distances in the provided R bins.
    # pairs to keep.
    R_min = np.min(R_bins); 
    R_max = np.max(R_bins)
    
    if print_progress: print('digitizing')
    # get the index of the bin each distance falls in
    R_bin_i = np.digitize(proj_dist, bins=R_bins) - 1                       # length of proj_dist, values are the bin index for each proj_dist
    # if pimax is a float or integer, keep pairs within that pimax
    if isinstance(pimax, (int, float)):
        i_keep = (proj_dist > R_min) & (proj_dist < R_max) & (los_sep < pimax)
    # if pimax is an array, keep pairs within the pimax for each bin
    elif len(pimax)==(len(R_bins)-1):
        if print_progress: print('getting i_keep')
        pimax = np.append(pimax, np.nan)                                    # add placeholder value to pimax for pairs that fall outside the bins (won't be used)
        i_keep = (proj_dist > R_min) & (proj_dist < R_max) & (los_sep < pimax[R_bin_i])
    else:
        raise ValueError('pimax must be a float or array of size (len(R_bins)-1)')
    
    if tracer_behind==True:
        # only keeping pairs where the tracer is behind the shape, to avoid lensing effects
        i_keep &= (dist_to_orgin_loc>0)
    
    if print_progress: print('calculating relative angles in each R_bin')
    rel_angles = []
    # return average in each bin
    for i in range(len(R_bins)-1):
        if print_progress: print('working on bin', i)
        # get the indices of pairs that fall in this bin and are included in i_keep
        i_bin_keep = i_keep & (R_bin_i == i)
        
        # if the number of "True" in i_bin_keep is more than 100 million, randomly set the excess to False
        n_in_bin = np.count_nonzero(i_bin_keep)
        if print_progress: print(n_in_bin, ' pairs in bin')
        if n_in_bin > 100000000:
            if print_progress: print(n_in_bin, ' pairs in bin, randomly selecting 100 million')
            n_to_remove = n_in_bin - 100000000
            # randomly set n_to_remove indeces in i_bin_keep to False
            to_sample = np.where(i_bin_keep)[0]
            i_random_remove = np.random.choice(to_sample, n_to_remove, replace=False)
            i_bin_keep[i_random_remove] = False
            
            del i_random_remove, to_sample, n_to_remove
            
        if print_progress: print('indexing bin arrays')
        c_bin = center_coords[i_bin_keep]
        n_bin = neighbor_coords[i_bin_keep]
        ca_bin = center_angles[i_bin_keep]
        cE_bin = center_E[i_bin_keep]
        weight_to_use = neighbor_weights[i_bin_keep]
        
        if print_progress: print('calculating position angle')
        # calculate position angle in batches
        n_batches = int(len(c_bin)/100)
        if n_batches==0: n_batches=1
        position_angle = process_in_batches(lambda x: get_orientation_angle_cartesian(c_bin[x], n_bin[x]), np.arange(len(c_bin)), n_batches) 
        
        pa_rel = ca_bin - position_angle
        # get weighted average from provided weights
        pa_rel *= 2
        pa_rel = np.cos(pa_rel)
        pa_rel *= cE_bin                        # multiplying by absolute value of ellipticity
        pa_rel *= weight_to_use    
        weighted_sum = np.nansum(pa_rel)
        weight_sum = np.nansum(weight_to_use)
        weighted_av = weighted_sum / weight_sum
        rel_angles.append(weighted_av)
    
    # remove all uneccesary variables from memory
    del center_coords, center_angles, neighbor_coords, neighbor_weights, dist_to_orgin_loc, dist_to_orgin_ang, los_sep, proj_dist
    del i_keep, R_bin_i, c_bin, n_bin, ca_bin, weight_to_use, position_angle, pa_rel, weighted_sum, weight_sum, weighted_av
    
    return rel_angles


# New function that calculates realtive ellipticities but bins in sep earlier to save memory
def rel_angle_regions_binned(orientation_catalog, loc_tracers, tracer_weights, R_bins, use_E_ABS=False, n_regions=100, pimax=30, 
                             keep_as_regions=False, print_progress=False, intermediate_save_paths=None, tracer_behind=False):
    '''
    Divides the orientation catalog into n_regions by RA and DEC, calculate cos(2*theta) the orientations relative to the tracers
    in bins of projected separation on the sky, R_bins. The measurement in each bin is measured relative to the full tracer sample.
    
    Parameters
    ----------
    orientation_catalog : astropy table
        table of groups with keys 'center_loc', 'orientation', 'RA', 'DEC'. 
        Optional: 'WEIGHT' - which could be absolute value of ellipticity to get full-shape alignment. Will need to set shape_weights=True.
    loc_tracers : array
        array of 3d points corresponding to the tracers
    tracer_weights : array
        weights of the tracers
    R_bins : array
        array of bin edges for the projected separaton, in Mpc/h
    use_E_ABS : bool, optional
        if True, will use the 'E_ABS' key in the orientation catalog. Default is False. Can be either definition.
    n_regions : int, optional
        number of sky regions to divide the data into. The default is 100. This is used to calculate standard error
    pimax : float or array, optional
        Maximum line-of-sight separation for pairs of galaxies in Mpc/h. Default is 30.
        Can also be list of size (R_bins-1) to use a different pimax for each R bin
        
    Returns
    -------
    if keep_as_regions is True:
        relAng_regions : array, size (n_regions, len(R_bins)-1)
            array of relative angles in each region
    if keep_as_regions is False:
        relAng : array, size (len(R_bins)-1)
            array of relative angles in each R bin
        relAng_e : array, size (len(R_bins)-1)
            array of standard error of relative angles from the independent regions
    '''
    
    ## START BY LOOPING OVER THE SKY REGIONS
    
    # sort ang_tracers and ang_values by DEC
    dec_sorter = np.argsort(orientation_catalog['DEC'])
    
    # divide catalogs into regions
    n_slices = int(np.sqrt(n_regions))
    n_in_dec_slice = int((len(orientation_catalog) / n_slices)+1)
    j = 0 
    k = n_in_dec_slice
    
    all_pa_rels = []
    all_region_files = []
    for _ in range(n_slices):   # loop over DEC slices
        
        # for handling the last region
        if k > len(orientation_catalog):     
            k = len(orientation_catalog)
        if len(orientation_catalog[j:k]) == 0:
            continue
        
        # slice in DEC
        groups_dec_slice = orientation_catalog[dec_sorter[j:k]]
        
        # sort this slice by RA
        ra_sorter = np.argsort(groups_dec_slice['RA'])
        n_in_ra_slice = int((len(groups_dec_slice) / n_slices)+1)
        n = 0
        m = n_in_ra_slice
        
        for _ in range(n_slices):       # loop over RA regions in this DEC slice
            
            if intermediate_save_paths is not None:
                # check to see if the file already exists
                if os.path.exists(intermediate_save_paths+'_'+str(j)+'_'+str(n)+'.npy'):
                    n += n_in_ra_slice
                    m += n_in_ra_slice
                    all_region_files.append(intermediate_save_paths+'_'+str(j)+'_'+str(n)+'.npy')
                    continue
                
            if print_progress: print('Working on region', j, n)
            
            # for handling the last region
            if m > len(groups_dec_slice):     
                m = len(groups_dec_slice)
            if len(groups_dec_slice[n:m]) == 0:
                continue
            
            group_square = groups_dec_slice[ra_sorter[n:m]]
            
            E_ABS = np.asarray([1]*len(group_square))
            if use_E_ABS==True:
                E_ABS = group_square['E_ABS']

            # returns the relative angles in each R bin for this region, array of size (len(R_bins)-1)
            pa_rel_binned = calculate_rel_ang_cartesian_binAverage(ang_tracers = group_square['center_loc'], ang_values = group_square['orientation'], 
                                                                   loc_tracers = loc_tracers, loc_weights=tracer_weights, E_ABS=E_ABS, 
                                                                   R_bins=R_bins, pimax=pimax, print_progress=print_progress, tracer_behind=tracer_behind)
            
            if intermediate_save_paths is not None:
                np.save(intermediate_save_paths+'_'+str(j)+'_'+str(n)+'.npy', pa_rel_binned)
            elif intermediate_save_paths is None:    
                all_pa_rels.append(pa_rel_binned)
            
            n += n_in_ra_slice
            m += n_in_ra_slice
        
        j += n_in_dec_slice
        k += n_in_dec_slice
        
    if intermediate_save_paths is not None:
        print('Complete. Region results saved to '+intermediate_save_paths)
        return None
    all_pa_rels = np.array(all_pa_rels)
    if keep_as_regions==True:
        return all_pa_rels
    elif keep_as_regions==False:
        pa_rel_av = np.nanmean(all_pa_rels, axis=0)
        pa_rel_e = np.nanstd(all_pa_rels, axis=0) / np.sqrt(len(all_pa_rels))
        return pa_rel_av, pa_rel_e
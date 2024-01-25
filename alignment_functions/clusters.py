import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from astropy.cosmology import LambdaCDM, z_at_value
H0 = 69.6
cosmo = LambdaCDM(H0=H0, Om0=0.286, Ode0=0.714)
h = H0/100
from geometry_functions.coordinate_functions import *
    
    
def find_group_2D_tree(catalog, sky_sep_max=1e-4, los_max=1, max_n=5):
    # query nearest neighbors
    '''
    Input:
    - catalog: astropy table of objects with columns RA, DEC, and DIST_COMOVING
    - sky_sep_max: maximum separation on the sky in radians
    - los_max: maximum separation along the line of sight in Mpc/h
    - max_n: maximum number of galaxies in cluster
    
    Returns: array of size (n_clusters, max_n) of indices of cluster members corresponding to points input
    '''
    sky_points = get_points(catalog) # put RA / DEC into cartesian points on a unit sphere
    tree = cKDTree(sky_points)

    # dd is distances, ii is indices
    dd, ii = tree.query(sky_points, distance_upper_bound=sky_sep_max, k=max_n) 
    
    # limit to pairs within los_max
    place_holder_row = [0]*(len(catalog[0])-1)
    place_holder_row.append(-3)
    catalog.add_row(place_holder_row)
    los_seps = np.abs(catalog['DIST_COMOVING'][ii[:,:1]] - catalog['DIST_COMOVING'][ii]) # index of neighbor - index of center
    remove = (los_seps > los_max)
    catalog.remove_row(-1)
    remove[:,:1] = False # don't want to remove indices of centers
    ii[remove] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
    dd[remove] = float('inf')
    
    # remove rows where there are no neighbors
    no_neighbors = (ii[:,1] == len(catalog))
    ii = ii[~no_neighbors]
    dd = dd[~no_neighbors]
    
    # only unique clusters
    # start by sorting each row of ii
    ii.sort(axis=1)
    ii = np.unique(ii, axis=0)
    
    return ii


def find_groups_3D_tree(positions, transverse_max=0.5, los_max=5, max_n=4):
    # query nearest neighbors
    '''
    Identifies groups in a cartesian grid using a cKDTree
    Input:
    - positions: x, y, z positions in Mpc/h. 
        Assumes the observer is at the orgin and x || LOS
    - sep_max: maximum separation in Mpc/h
    - los_max: maximum separation in Mpc/h
    - max_n: maximum number of galaxies in cluster
    
    Returns: array of size (n_clusters, max_n) of indices of cluster members corresponding to points input
    '''
    tree = cKDTree(positions)

    sep_max = np.sqrt(transverse_max**2 + los_max**2)
    # dd is distances, ii is indices
    dd, ii = tree.query(positions, distance_upper_bound=sep_max, k=max_n) 
    
    # limit to pairs within transverse_max and los_max
    
    # add a row of 0s to the positions array
    positions = np.vstack((positions, [0,0,0]))
    
    # find the difference between the neighbor and the center's distance to the orgin
    dist_los = np.sqrt(positions[:,0]**2 + positions[:,1]**2 + positions[:,2]**2)
    los_seps = dist_los[ii[:,:1]] - dist_los[ii]  # neighbor - center
    # transverse separation between the galaxies relative to the orgin
    ci = np.repeat(ii[:,0], (len(ii[0])-1)).ravel() # indices of centers
    ni = ii[:,1:].ravel()   # indices of neighbors
    proj_sep = get_proj_dist(positions[ci], positions[ni], pos_obs=np.asarray([0, 0, 0]))
    
    remove = (los_seps > los_max)
    remove |= (proj_sep > transverse_max)
    positions = positions[:-1]  # remove the placeholer row
    remove[:,:1] = False # don't want to remove indices of centers
    ii[remove] = len(positions)  # where the pairs are too far, functionally remove them from the list of pairs
    dd[remove] = float('inf')
    
    # remove rows where there are no neighbors
    no_neighbors = (ii[:,1] == len(positions))
    ii = ii[~no_neighbors]
    dd = dd[~no_neighbors]
    
    # only unique clusters
    # start by sorting each row of ii
    ii.sort(axis=1)
    ii = np.unique(ii, axis=0)
    
    return ii


def find_neighbors(query_catalog, catalog, sep_max, sep_min=False, k=2, double=False, return_sep=False):
    # query nearest neighbors
    # dd is distances, ii is indices
    catalog_points = get_points(catalog)
    tree = cKDTree(catalog_points)
    
    query_points = get_points(query_catalog)
    dd, ii = tree.query(query_points, distance_upper_bound=sep_max, k=k) 
    
    # remove where separation below minimum
    if sep_min != False:
        place_holder_row = [0]*(len(catalog[0])-1)
        place_holder_row.append(-3)
        catalog.add_row(place_holder_row)
        remove = dd < sep_min
        catalog.remove_row(-1)
        remove[:,:1] = False # don't want to remove indices of centers
        ii[remove] = len(catalog)  # where the pairs are too close, functionally remove them from the list of pairs
        dd[remove] = float('inf')
    
    # indices in catalog of neighbors, arranges so is the same shape as the 
    #ci = np.repeat(ii[:,0], (len(ii[0])-1)).ravel() # indices of centers
    #ni = ii[:,1:].ravel()   # indices of neighbors
    # removing places where no neighbor was found in the tree
    #neighbor_exists = (ni!=len(catalog))
    #ci = ci[neighbor_exists]; ni = ni[neighbor_exists]

    if return_sep==True:
        seps = dd[:,1:].ravel()
        seps = seps[seps!= float('inf')]
        return ii, seps
    
    elif return_sep==False:
        return ii
    
    
def find_pairs(query_points, catalog, sep_max, k=2, double=False, return_sep=False):
    # query nearest neighbors
    # dd is distances, ii is indices
    tree = cKDTree(catalog)
    dd, ii = tree.query(query_points, distance_upper_bound=sep_max, k=k) 
    
    # limit to one occurence of each pair
    #if double==False:
    #    unique_pair_indices = np.where((ii[:,0] != ii[:,1]))[0]
    #    # functionally remove them from the list of pairs
    #    ii[unique_pair_indices] = len(points) 
    #    dd[unique_pair_indices] = float('inf')
    
    # indices in catalog of centers and neighbors, arranges so each array is same shape
    ci = np.repeat(ii[:,0], (len(ii[0])-1)).ravel() # indices of centers
    ni = ii[:,1:].ravel()   # indices of neighbors
    # removing places where no neighbor was found in the tree
    neighbor_exists = (ni!=len(query_points))
    ci = ci[neighbor_exists]; ni = ni[neighbor_exists]

    if return_sep==True:
        seps = dd[:,1:].ravel()
        seps = seps[seps!= float('inf')]
        return ci, ni, seps
    
    elif return_sep==False:
        return ci, ni


def calculate_pair_pa(catalog, cluster_indices, u_coords='deg', u_result=u.rad):
    centers_m = catalog[cluster_indices[:,0]]
    neighbors_m = catalog[cluster_indices[:,1]]
    
    # find center and orientation of pairs, projected along the LOS
    ### CHECK
    c1 = SkyCoord(centers_m['RA'], centers_m['DEC'], unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(neighbors_m['RA'], neighbors_m['DEC'], unit=u_coords, frame='icrs', equinox='J2000.0')
    c_middle = SkyCoord((centers_m['RA'] + neighbors_m['RA']) / 2, (centers_m['DEC'] + neighbors_m['DEC']) / 2, unit=u_coords, frame='icrs', equinox='J2000.0')
    
    position_angles = c1.position_angle(c2).to(u_result)
    
    pair_table = Table()
    pair_table['RA'] = c_middle.ra.to(u.deg) # in degree
    pair_table['DEC'] = c_middle.dec.to(u.deg) 
     # use astropy to convert from z to comoving distance and find average
    pair_table['DIST_COMOVING'] = ((cosmo.comoving_distance(centers_m['Z']) + cosmo.comoving_distance(neighbors_m['Z'])) / 2).to(u.Mpc) * 0.7
    pair_table['THETA'] = position_angles # in rad
    pair_table['Z'] = (centers_m['Z'] + neighbors_m['Z']) / 2
    
    return pair_table
    
def find_projected_pairs_cartesian(catalog, tree, sep_max=2, n_in_cluster=2):
    '''Assumes LOS along x
    sep_max in units of Mpc/h
    '''
    ci, ni = find_pairs(catalog['x_L2com'], tree, sep_max=sep_max, k=(n_in_cluster))
    centers_m = catalog[ci]; neighbors_m = catalog[ni] 
    
    # find center and orientation of pairs, projected along the LOS]
    center_points = (centers_m['x_L2com'] + neighbors_m['x_L2com']) / 2
    sep_vectors = (centers_m['x_L2com'] - neighbors_m['x_L2com'])
    
    return center_points, sep_vectors    
    
def find_projected_pairs_ra_dec(catalog, tree, sep_max=deg_to_rad(0.1), n_in_cluster=2, u_coords='deg', u_result=u.rad):
    '''Assumes LOS along x'''
    
    center_points = get_points(catalog)
    
    ci, ni = find_pairs(center_points, tree, sep_max=sep_max, k=n_in_cluster)
    
    centers_m = catalog[ci]
    neighbors_m = catalog[ni]   # excluding the centers
    
    # find center and orientation of pairs, projected along the LOS
    ### CHECK
    c1 = SkyCoord(centers_m['RA'], centers_m['DEC'], unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(neighbors_m['RA'], neighbors_m['DEC'], unit=u_coords, frame='icrs', equinox='J2000.0')
    c_middle = SkyCoord((centers_m['RA'] + neighbors_m['RA']) / 2, (centers_m['DEC'] + neighbors_m['DEC']) / 2, unit=u_coords, frame='icrs', equinox='J2000.0')
    
    position_angles = c1.position_angle(c2).to(u_result)
    
    pair_table = Table()
    pair_table['RA'] = c_middle.ra.to(u.deg) # in degree
    pair_table['DEC'] = c_middle.dec.to(u.deg) 
     # use astropy to convert from z to comoving distance and find average
    pair_table['DIST_COMOVING'] = ((cosmo.comoving_distance(centers_m['Z']) + cosmo.comoving_distance(neighbors_m['Z'])) / 2).to(u.Mpc) * 0.7
    pair_table['THETA'] = position_angles # in rad
    pair_table['Z'] = (centers_m['Z'] + neighbors_m['Z']) / 2
    
    return pair_table

def calculate_rel_ang_cartesian(center_points, tree, sep_vectors, sep_max = 30, max_neighbors=100, pimax=30):
    '''no sky projection. sep_max and pimax in units of Mpc/h'''
    
    ci, ni, dd = find_pairs(center_points, tree, sep_max, k=max_neighbors, double=False, return_sep=True)
    
    # limit to pimax and ones with some separation
    to_keep = np.abs(center_points[ci,0] - center_points[ni,0]) < pimax
    to_keep &= (dd > 2)
    ci = ci[to_keep]; ni = ni[to_keep]
    
    # get projected sep and position angle
    proj_seps = np.sqrt((center_points[ci,1]-center_points[ni,1])**2 + (center_points[ci,2] - center_points[ni,2])**2)
    pa = np.arctan2((center_points[ci,2] - center_points[ni,2]), (center_points[ci,1]-center_points[ni,1]))
    orientation_angle = np.arctan2(sep_vectors[ci,2], sep_vectors[ci,1])
    pa_rel = orientation_angle - pa
    
    return proj_seps, np.abs(pa_rel)

def calculate_rel_ang_ra_dec(query_catalog, catalog, sep_max = deg_to_rad(.5), 
                             sep_min = False, max_neighbors=100, pimax=30, sep_type='angular'):
    
    matches_ii = find_neighbors(query_catalog, catalog, sep_max, sep_min=sep_min, k=max_neighbors)
    # make arrays of centers and neighbors, with identical shapes
    cluster_centers_i = np.repeat(np.arange(len(query_catalog)), max_neighbors-1).ravel()
    cluster_neighbors_i = matches_ii[:,1:].ravel()
    
    # removing places where no neighbor was found in the tree
    neighbor_exists = (cluster_neighbors_i!=len(catalog))
    cluster_centers_i = cluster_centers_i[neighbor_exists]; cluster_neighbors_i = cluster_neighbors_i[neighbor_exists]
    
    centers_m = query_catalog[cluster_centers_i]
    neighbors_m = catalog[cluster_neighbors_i]
    
    # limit to pimax
    i_keep = np.abs(centers_m['DIST_COMOVING'] - neighbors_m['DIST_COMOVING']) < pimax
    centers_m = centers_m[i_keep];  neighbors_m = neighbors_m[i_keep]
    
    # get sep and relative position angle
    if sep_type=='angular':
        sep, pa = get_sep_pa(centers_m['RA'], centers_m['DEC'], neighbors_m['RA'], neighbors_m['DEC'])
        pa_rel = centers_m['THETA'] - pa
        return sep.to(u.deg).value, pa_rel
        
    elif sep_type=='comoving':
        sep, pa = get_cosmo_psep_pa(centers_m['RA'], centers_m['DEC'], 
                                    neighbors_m['RA'], neighbors_m['DEC'], centers_m['Z'], neighbors_m['Z']) 
        pa_rel = centers_m['THETA'] - pa
        return sep.to(u.Mpc*u.rad).value*.7, pa_rel
    
    
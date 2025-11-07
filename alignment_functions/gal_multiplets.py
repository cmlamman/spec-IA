import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from astropy.cosmology import LambdaCDM, z_at_value
from alignment_functions.basic_alignment import *
from geometry_functions.coordinate_functions import get_proj_dist
import random


H0 = 69.6
cosmo = LambdaCDM(H0=H0, Om0=0.286, Ode0=0.714)
h = H0/100
from geometry_functions.coordinate_functions import *

from collections import defaultdict


############################################
# FINDING MULTIPLETS WITH UNION-FIND
############################################
class Node:
    def __init__(self, key):
        self.key = key
        self.parent = self
        self.size = 1

class UnionFind(dict):
    def find(self, key):
        node = self.get(key, None)
        if node is None:
            node = self[key] = Node(key)
        else:
            while node.parent != node: 
                # walk up & perform path compression
                node.parent, node = node.parent.parent, node.parent
        return node

    def union(self, key_a, key_b):
        node_a = self.find(key_a)
        node_b = self.find(key_b)
        if node_a != node_b:  # disjoint? -> join!
            if node_a.size < node_b.size:
                node_a.parent = node_b
                node_b.size += node_a.size
            else:
                node_b.parent = node_a
                node_a.size += node_b.size
                
def find_components(line_iterator, max_size):
    forest = UnionFind()

    for line in line_iterator:
        forest.union(*line.split())

    result = defaultdict(list)
    for key in forest.keys():
        root = forest.find(key)
        if root.size <= max_size:
            result[root.key].append(int(float(key)))  # Convert key to float and then to int

    # return list of integers
    return list(result.values())

def find_groups_UnionFind(points, max_n=10, transverse_max = 0.5, los_max = 6, transverse_min=None, poo=np.asarray([0, 0, 0])):
    '''assumes LOS is along z-axis and observer at poo (default origin)'''
    points -= poo  # shift points to observer at origin
    # find nearest neighbor for each galaxy
    tree = cKDTree(points)
    max_pair_sep = np.sqrt(transverse_max**2 + los_max**2)
    dd, ii = tree.query(points, distance_upper_bound=max_pair_sep, k=2) 

    # only keep indices where neighbor exists
    ii = ii[ii[:,1] < len(points)]
    
    # remove pairs that are too far away
    transverse_seps = np.abs(get_proj_dist(points[ii[:,1]], points[ii[:,0]]))
    los_dist = np.sqrt(np.sum(points**2, axis=1))
    los_seps = np.abs(los_dist[ii[:,1]] - los_dist[ii[:,0]])
    to_remove = (transverse_seps > transverse_max) | (los_seps > los_max)
    if transverse_min is not None:
        to_remove |= (transverse_seps < transverse_min)
 
    #to_remove[:,:1]=False # don't want to remove indices of centers
    ii = ii[~to_remove]  # where the pairs are too close, functionally remove them from the list of pairs
    
    # find groups
    group_results = find_components([' '.join(map(str, row)) for row in ii], max_n)
    
    return group_results
    
# remove groups that have an outside object within group_sep_min of any of their members
def get_isolated_groups(points, group_result, group_transverseSep_min=10, group_losSep_min=12):
    
    # build the tree of all galaxies
    point_tree = cKDTree(points)
     # add row of infinite values to points for later
    points = np.vstack((points, np.full(len(points[0]), np.inf)))
    
    groups_to_keep = []
    
    for g in group_result:
        
        # find nearest non-group neighbor of each group member
        dd, ii = point_tree.query(points[g], k=len(g)+1)
        # remove the group members from the list of neighbors
    
        # find the neighbor that's not in the group
        i_outside = np.array([np.where(~np.isin(ii[i], g))[0][0] for i in range(len(g))])
    
        # find the transverse separation between each group member and its nearest non-group neighbor
        transverse_seps = np.array(np.abs(get_proj_dist(points[i_outside], points[g])))
        los_seps = np.array(np.abs(np.sqrt(np.sum(points[i_outside]**2, axis=1)) - np.sqrt(np.sum(points[g]**2, axis=1))))
        
        if (np.nanmin(transverse_seps) > group_transverseSep_min) & (np.nanmin(los_seps) > group_losSep_min):
            groups_to_keep.append(g)
    
    return groups_to_keep
    
    
###############################################
# CALCULATE GROUP PROJECTED SHAPE
###############################################


def calculate_2D_group_orientation(points_3d):
    '''
    Calculate the orientation of a group of points, measured "E of N", aka clockwise from the y-axis
    points_2d: array of shape (n_points, 2)
    returns: array of angles in radians, shape (n_points,)
    '''
    points_2d = get_points_in_plane(points_3d)
    # convert to polar coordinates
    r = np.linalg.norm(points_2d, axis=1)
    theta = np.arctan2(points_2d[:,0], points_2d[:,1])
    
    # compute complex number representation of the points
    points_complex = r * np.exp(2j*theta)  # 2 so invariant under rotation
    
    average_points_complex = np.mean(points_complex)
    
    return np.angle(average_points_complex) / 2

#-----------------------
# backup functions

# cutting list down to requirements
def trim_groups(points, group_indices, transverse_max, los_max):
    # assumes observer is as orgin
    
    # find groupenters
    # find the center coordinates of each group
    group_centers = np.array([np.mean(points[cl], axis=0) for cl in group_indices])
    
    # los-distance for groupenters
    los_dist = np.sqrt(np.sum(group_centers**2, axis=1))
    
    max_transverse_dist_to_center = []
    max_los_dist_to_center = []
    
    for i in range(len(group_indices)):
        cl = group_indices[i]
        max_transverse_dist_to_center.append(np.max(get_proj_dist(points[cl], group_centers[i])))
        max_los_dist_to_center.append(np.max(los_dist[i] - np.sqrt(np.sum(points[cl]**2, axis=1))))
    
    groupto_keep = (np.asarray(max_transverse_dist_to_center) < transverse_max)
    groupto_keep &= (np.asarray(max_los_dist_to_center) < los_max)
    
    return [group_indices[i] for i in range(len(group_indices)) if groupto_keep[i]]



###############################
# HIGH - LEVEL FUNCTIONS
###############################


def make_group_catalog(data_catalog, comoving_points=None, transverse_max = 1, los_max = 6, max_n = 100, cosmology=cosmo, transverse_min=None, truez=False, use_sky_coords=True):
    '''
    This function finds pairs of galaxies in a catalog, creats groups from the pairs, and returns a catalog of group properties.
    
    Input
    -----
    data_catalog: astropy table with columns ['RA', 'DEC', 'Z', 'WEIGHT']
    transverse_max: maximum distance in Mpc/h between galaxies in the plane of the sky for making pairs
    los_max: maximum distance in Mpc/h between galaxies along the line of sight for making pairs
    max_n: maximum number of group members. relistically this doesn't go above 10
    cosmology: astropy cosmology object
    truez: whether to use the true redshifts, for mocks. data_catalog must have a 'TRUEZ' column.
    
    Returns
    -------
    group_table: astropy table with columns:
    'center_loc': center of the group in comoving coordinates, Mpc/h
    'orientation': orientation of the group in radians
    'n_group': number of members in the group
    'max_dist_to_center': maximum distance of a group member from the center, Mpc/h
    'RA': RA of the group center
    'DEC': DEC of the group center
    'Z': redshift of the group center
    '''
    
    # convert points to comoving grid, in units of Mpc/h
    if comoving_points is None:
        comoving_points = get_cosmo_points(data_catalog, cosmology=cosmology) # this delibarately does not use truez
     
    # find groups with Union find
    group_indices = find_groups_UnionFind(comoving_points, max_n = max_n, transverse_max = transverse_max, los_max = los_max, transverse_min=transverse_min)
    
    if truez:
        comoving_points = get_cosmo_points(data_catalog, cosmology=cosmology, truez=truez)
    
    group_table = Table()
    group_table['center_loc'] = [np.mean(comoving_points[cl], axis=0) for cl in group_indices]
    group_table['orientation'] = [calculate_2D_group_orientation(comoving_points[cl]) for cl in group_indices]
    group_table['n_group'] = [len(cl) for cl in group_indices]
    group_table['max_dist_to_center'] = [np.max(np.linalg.norm(comoving_points[cl] - group_table['center_loc'][i], axis=1)) for i, cl in enumerate(group_indices)]
    if use_sky_coords:
        group_table['RA'] = [data_catalog['RA'][gi[0]] for gi in group_indices]    # just using one point in the group to get RA / DEC for sorting
        group_table['DEC'] = [data_catalog['DEC'][gi[0]] for gi in group_indices]
        group_table['Z'] = [np.mean(data_catalog['Z'][gi]) for gi in group_indices]
        if truez:
            group_table['TRUEZ'] = [np.mean(data_catalog['TRUEZ'][gi]) for gi in group_indices]
    
    return group_table


def get_multiplet_alignment(catalog_for_groups, catalog_for_tracers=None, R_bins=np.logspace(0, 2, 10), pimax=30, cosmology=cosmo, print_progress=False, 
                        n_sky_regions=100, save_path=None, pair_max_los=6, pair_max_transverse=1, pair_min_transverse=None, early_binning=False, keep_intermediate=False,
                        truez=False, intermediate_save_paths=None):
    '''
    Calculate the alignment of galaxy multiplets (sometimes called 'groups' here) within the given catalog, relative to tracers from the same catalog or other, if provided. 
    Saves results to save_path, if provided.
    
    Parameters:
    - catalog_for_groups (dict): Catalog used to find groups.
    - catalog_for_tracers (dict, optional): Catalog of tracers. If not provided, the same catalog as the groups will be used.
    - cosmology (Cosmology, optional): Cosmology object defining the cosmological parameters. Default is LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714).
    - print_progress (bool, optional): Whether to print progress messages. Default is False.
    - n_sky_regions (int, optional): Number of sky regions to measure alignment (relative to full catalog). 
        Regions divided so equal number of groups in each. Error on measurement is standard error of these regions. Default is 100. 
    - pimax (float, optional) or (list, length of n_Rbins): Maximum line-of-sight separation for pairs of galaxies in Mpc/h. Default is 30.
    - max_proj_sep (float, optional): Maximum projected separation for pairs of galaxies in Mpc/h. Default is 150.
    - max_neighbors (int, optional): Maximum number of neighbors to consider for each galaxy. Default is 1000.
    - n_Rbins (int, optional): Number of transverse bins for binning the results. Default is 10.
    - save_path (str, optional): Path to save the results. If not provided, the results will not be saved.
    - early binning: This option will bin the galaxies early on, saving memory. Helpful if running on dense regions (like BGS), but will generally be a bit noisier measurement.
    
    Returns:
    results (Table): Table with columns 'R_bin_edges', 'relAang_plot', 'relAng_plot_e'.
    - R_bin_min, R_bin_max: Edges of the transverse separation bins, Mpc/h.
    - relAang_plot: cos(2*theta), where theta is the mean relative angle between group orientation and tracer location in each bin.
    - relAng_plot_e: Error on relAang_plot, from standard error of measurements in each sky region.
    '''
    
    # put the catalogs for groups and tracers in comoving coordinates
    comoving_points_groups = get_cosmo_points(catalog_for_groups, cosmology=cosmology)  # this deliberately doesn't use truez
    if catalog_for_tracers is None:
        catalog_for_tracers = catalog_for_groups
        comoving_points_tracers = comoving_points_groups
    else:
        comoving_points_tracers = get_cosmo_points(catalog_for_tracers, cosmology=cosmology, truez=truez)
    try:
        catalog_for_tracers['WEIGHT']
    except KeyError:
        catalog_for_tracers['WEIGHT'] = np.ones(len(catalog_for_tracers))
        
    if print_progress:
        print('Making group catalog')
    group_catalog = make_group_catalog(catalog_for_groups, comoving_points = comoving_points_groups, cosmology=cosmology, 
                                       los_max=pair_max_los, transverse_max=pair_max_transverse, transverse_min=pair_min_transverse, truez=truez)
    #if print_progress:
    print('Number of multipelts found:', len(group_catalog))
    if print_progress:
        print('Measuring alignment')
    
    if early_binning:
        if intermediate_save_paths is None:
            try:
                intermediate_save_paths = save_path.split('.fits')[0]
            except:
                print('Save path must be provided to use early binning')
                return None
        
        rel_angle_regions_binned(group_catalog, loc_tracers = comoving_points_tracers,  tracer_weights = catalog_for_tracers['WEIGHT'],
                                                    R_bins=R_bins, n_regions=n_sky_regions, pimax=pimax, keep_as_regions=False, print_progress=print_progress, 
                                                    intermediate_save_paths=intermediate_save_paths)
        # reading in the calculated results
        if print_progress:
            print('Reading in region results')
        region_paths = glob.glob(intermediate_save_paths + '*.npy')
        all_pa_rels = np.asarray([np.load(region_path) for region_path in region_paths])
        relAng = np.nanmean(all_pa_rels, axis=0)
        relAng_e = np.nanstd(all_pa_rels, axis=0) / np.sqrt(len(all_pa_rels))
        # remove intermediate files
        if not keep_intermediate:
            for region_path in region_paths:
                os.remove(region_path)
        
    else:
        max_proj_sep = np.max(R_bins)
        n_Rbins = len(R_bins) - 1
        
        # if pimax is not a single value...
        if isinstance(pimax, (int, float)):
            group_seps, group_paRel, weights = rel_angle_regions(group_catalog, loc_tracers = comoving_points_tracers, tracer_weights = catalog_for_tracers['WEIGHT'],
                                                            n_regions=n_sky_regions, pimax=pimax, max_proj_sep=max_proj_sep, return_los=False)
            group_los = None
            use_sliding_pimax = False
        else:
            group_seps, group_paRel, weights, group_los = rel_angle_regions(group_catalog, loc_tracers = comoving_points_tracers, tracer_weights = catalog_for_tracers['WEIGHT'],
                                                            n_regions=n_sky_regions, pimax=np.max(pimax), max_proj_sep=max_proj_sep, return_los=True)
            use_sliding_pimax = True
        
        sep_bins, relAng, relAng_e = bin_region_results(group_seps, group_paRel, all_weights = weights, R_bins=R_bins, use_sliding_pimax=use_sliding_pimax, los_sep=group_los)
    
    results = Table()
    
    results['R_bin_min'] = R_bins[:-1]
    results['R_bin_max'] = R_bins[1:]
    results['relAng_plot'] = relAng
    results['relAng_plot_e'] = relAng_e
    if isinstance(pimax, (int, float)):
        results['pimax'] = [pimax] * len(R_bins[:-1])
    else:
        results['pimax'] = pimax
    
    if save_path is not None:
        results.write(save_path, overwrite=True)
        print('Results saved to ', save_path) 
        
    return results


def get_multiplet_alignment_randoms(catalog_for_groups, random_catalog_paths, R_bins, pimax=30, cosmology=cosmo, print_progress=False, 
                        n_sky_regions=100, save_path=None, pair_max_los=6, pair_max_transverse=1, pair_min_transverse=None, early_binning=False, keep_intermediate=False, intermediate_save_paths=None):
    '''
    Simillar to get_multiplet_alignment, but calculates the alignment of galaxy multiplets within the given catalog relative to multiple random catalogs.
    random_catalog_paths: list of paths to random catalogs. 
    If input for random_catalog_paths is an integer, code will automatically generate that many random catalogs from the data by shuffling Z.
    '''
    if save_path is None:
        raise ValueError('save_path must be provided')
    
    group_catalog = make_group_catalog(catalog_for_groups, cosmology=cosmology, los_max=pair_max_los, transverse_max=pair_max_transverse, transverse_min=pair_min_transverse)

    rand_signal = []
    # check if random_catalog_paths is an integer
    if isinstance(random_catalog_paths, int):
        n_random_catalogs = random_catalog_paths
    else:
        n_random_catalogs = len(random_catalog_paths)
    
    for rand_batch in range(n_random_catalogs):
        #rand_file_number = random_catalog_paths[rand_batch].split('_')[-2]
        #print('Working on random batch', rand_batch, 'of', n_random_catalogs, '. Random catalog', rand_file_number)
        #intermediate_save_paths_rand += '-' + rand_file_number  ## should come up with a more general way to do this! I actually think I should keep random batches seperate
        
        if isinstance(random_catalog_paths, int):
            random_catalog = Table()
            random_catalog['RA'] = catalog_for_groups['RA']
            random_catalog['DEC'] = catalog_for_groups['DEC']
            random_catalog['Z'] = np.random.permutation(catalog_for_groups['Z'])
        elif isinstance(random_catalog_paths, list):
            random_catalog = Table.read(random_catalog_paths[rand_batch])
            random_catalog.keep_columns(['RA', 'DEC'])
            random_catalog = random_catalog[(np.random.choice(len(random_catalog), len(catalog_for_groups), replace=False))]
            random_catalog['Z'] = catalog_for_groups['Z']
        
        random_catalog['WEIGHT'] = np.ones(len(random_catalog))
        comoving_points_tracers = get_cosmo_points(random_catalog)  # convert to comoving cartesian points in Mpc/h, assumes observer is at orgin
            
            
        if print_progress:
            print('Measuring alignment')
        
        if early_binning:
            if intermediate_save_paths is None:
                try:
                    intermediate_save_paths = save_path.split('.fits')[0]
                except:
                    print('Save path must be provided to use early binning')
                    return None
            
            rel_angle_regions_binned(group_catalog, loc_tracers = comoving_points_tracers,  tracer_weights = random_catalog['WEIGHT'],
                                                        R_bins=R_bins, n_regions=n_sky_regions, pimax=pimax, keep_as_regions=False, print_progress=False, 
                                                        intermediate_save_paths=intermediate_save_paths)
            # reading in the calculated results
            if print_progress:
                print('Reading in region results')
            region_paths = glob.glob(intermediate_save_paths + '*.npy')
            all_pa_rels = np.asarray([np.load(region_path) for region_path in region_paths])
            relAng = np.nanmean(all_pa_rels, axis=0)
            #relAng_e = np.nanstd(all_pa_rels, axis=0) / np.sqrt(len(all_pa_rels))
            # remove intermediate files
            if not keep_intermediate:
                for region_path in region_paths:
                    os.remove(region_path)
            rand_signal.append(relAng)
            
        else:
            max_proj_sep = np.max(R_bins)
            n_Rbins = len(R_bins) - 1
            
            # if pimax is not a single value...
            if isinstance(pimax, (int, float)):
                group_seps, group_paRel, weights = rel_angle_regions(group_catalog, loc_tracers = comoving_points_tracers, tracer_weights = random_catalog['WEIGHT'],
                                                                n_regions=n_sky_regions, pimax=pimax, max_proj_sep=max_proj_sep, return_los=False)
                group_los = None
                use_sliding_pimax = False
            else:
                group_seps, group_paRel, weights, group_los = rel_angle_regions(group_catalog, loc_tracers = comoving_points_tracers, tracer_weights = random_catalog['WEIGHT'],
                                                                n_regions=n_sky_regions, pimax=np.max(pimax), max_proj_sep=max_proj_sep, return_los=True)
                use_sliding_pimax = True
            
            sep_bins, relAng, relAng_e = bin_region_results(group_seps, group_paRel, all_weights = weights, R_bins=R_bins, use_sliding_pimax=use_sliding_pimax, los_sep=group_los)
            rand_signal.append(relAng)


    # saving randoms
    print('Saving signal from randoms to', save_path)
    results = Table()
    results['R_bin_min'] = R_bins[:-1]
    results['R_bin_max'] = R_bins[1:]
    results['relAng_plot'] = np.mean(np.asarray(rand_signal), axis=0)
    results['relAng_plot_e'] = np.std(np.asarray(rand_signal), axis=0) / np.sqrt(len(rand_signal))
    if isinstance(pimax, (int, float)):
        results['pimax'] = [pimax] * len(R_bins[:-1])
    else:
        results['pimax'] = pimax
    
    results.write(save_path, overwrite=True)
    print('Results saved to ', save_path) 
    
    
def get_group_2pt_projected_corr(catalog, random_paths, catalog2=None, tracer_catalog=None, rp_bins=np.logspace(0, np.log10(150), 11), rpar_bins=np.linspace(0, 80, 101), 
                                 use_sliding_pimax=False, print_progress=False, save_path=None):    
    '''
    Calculate projected 2-point correlation functions between galaxy groups in catalog and the catalog (or a tracer catalog).
    bins are given in bin edges.
    Will use variable pimax if use_sliding_pimax is True, else just the maximum of the rpar_bins.
    Returns the correlation function and saves it to save_path if provided.
    '''
    
    from pycorr import TwoPointCorrelationFunction  # needs to be run in environment with pycorr!
    
    if catalog2 is None:
        pos = format_pos_for_cf(catalog, z_column='Z')
    else:
        pos = format_pos_for_cf(catalog2, z_column='Z')
    
    catalog2 =  make_group_catalog(catalog)
    pos2 = format_pos_for_cf(catalog2, z_column='Z')
    pos_r2 = generate_randoms_zshuffle(catalog2)
    
    
    corr_results = []
    n=0
    for random_path in random_paths:
        if print_progress:
            print('working on ',n, ' of ', len(random_paths))
        
        desi_randoms = Table.read(random_path)
        desi_randoms.keep_columns(['RA', 'DEC'])
        desi_randoms = desi_randoms[(np.random.choice(len(desi_randoms), len(catalog), replace=False))]
        desi_randoms['Z'] = catalog['Z']
        pos_r = format_pos_for_cf(desi_randoms, z_column='Z')
        
        corr_result1 = TwoPointCorrelationFunction('rppi', edges=(rp_bins, rpar_bins), position_type='rdd', data_positions1=pos, randoms_positions1=pos_r,
                                              data_positions2=pos2, randoms_positions2=pos_r2, engine='corrfunc', nthreads=4)
           
        if use_sliding_pimax:
            bin_centers = (rp_bins[1:] + rp_bins[:-1])/2
            pi_max_values = sliding_pimax(bin_centers)
            wp_values = []
            for i in range(len(bin_centers)):
                wp1 = corr_result1(pimax=pi_max_values[i])
                wp_values.append(wp1[i])
            corr_results.append(wp_values)
        else:
            wp1 = corr_result1(pimax=None)
            corr_results.append(corr_result1)
        n+=1
    # averaging over all randoms
    corr_results = np.array(corr_results)
    wp_result = np.nanmean(corr_results, axis=0)
    wp_result_e = np.nanstd(corr_results, axis=0) / np.sqrt(len(corr_results))
    
    if save_path is not None:
        corr_table = Table()
        corr_table['R_bin_min'] = rp_bins[:-1]
        corr_table['R_bin_max'] = rp_bins[1:]
        corr_table['pimax'] = pi_max_values
        corr_table['wp'] = wp_result
        corr_table['wp_e'] = wp_result_e
        corr_table.write(save_path, overwrite=True)
        
    return np.mean(corr_results, axis=0)



######################
# HIGH_LEVEL FUNCTION FOR SIMULATION DATA
#######################

def get_MIA_from3D(points_3D, save_directory, R_bins = np.logspace(np.log10(5), np.log10(100), 16), print_info=True, sim_label='example',
                   periodic_boundary=False, transverse_max = 1, los_max=6, max_rp=100, n_batches = 10, save_intermediate=False):
    '''
    A high-level function to calculate projected multiplet alignment for a set of points in 3D comoving space.
    Input points and parameters can be in any units as long as they are consistent.
    -----------
    points_3D: x, y, z positions of points. 
    R_bins: bin edges of the transverse separation for the final measurement
    sim_label (optional): number to keep track of running multiple sims
    periodic_boundary (optional): whether to use periodic boundary conditions. If True, will extend the box by adding copies of the points from each side.
    '''
    
    if print_info:
        print('Calculating MIA for %d points' % len(points_3D))
        
    if print_info:
        print('Finding multiplets')
    
    points_3D -= np.min(points_3D, axis=0) # ensure that the simulation corner is at 0,0,0
    multiplet_table = make_group_catalog(None, comoving_points=points_3D, transverse_max=transverse_max, los_max=los_max, max_n=100, use_sky_coords=False)
    if print_info:
        print('Found %d multiplets' % len(multiplet_table), 'averange number of members: %f' % np.mean(multiplet_table['n_group']))  
    # print binned counts of multiplet sizes
    if print_info:
        print('Multiplet size counts:') 
        unique, counts = np.unique(multiplet_table['n_group'], return_counts=True)
        print('size:', [u for u in np.asarray(unique)])
        print('counts:', [c for c in np.asarray(counts)])

    if periodic_boundary:
        # make an array with every comination of adding or subtracting the box size to each dimmension
            
        extend_by = max_rp
        new_orgins = np.array([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]) * np.max(points_3D[:,0])  # assumes box is a cube with one corner at 0,0,0
        extended_points = np.array([points_3D + new_orgins[i] for i in range(len(new_orgins))])
        extended_points = np.concatenate(extended_points, axis=0)

        # trim to points within some distance of origional box
        i_keep = (extended_points[:,0] > np.min(points_3D[:,0])-extend_by) & (extended_points[:,0] < np.max(points_3D[:,0])+extend_by) 
        i_keep &= (extended_points[:,1] > np.min(points_3D[:,1])-extend_by) & (extended_points[:,1] < np.max(points_3D[:,1])+extend_by)
        i_keep &= (extended_points[:,2] > np.min(points_3D[:,2])-extend_by) & (extended_points[:,2] < np.max(points_3D[:,2])+extend_by)
        tracer_points = extended_points[i_keep]
    else:
        tracer_points = points_3D
    
    ## default - add as argument later
    R_bin_middles = (R_bins[1:] + R_bins[:-1])/2
    pimax_values = 8 + (2/3)*R_bin_middles   # add as argument later


    # order group table randomly (but reproducibly)
    random.seed(42)
    indices = np.asarray(range(len(multiplet_table)))
    random.shuffle(indices)
    multiplet_table = multiplet_table[indices]

    # run in n_batches
    i_end = int(len(multiplet_table)/n_batches)
    i_start = 0
    
    pa_rel_binned_all = []
    for i in range(int(n_batches)):
        # print progress every 12
        if i % 24 == 0 and print_info:
            print('working on batch', i, 'sim:', sim_label)
        batch_save_path = save_directory + '/MIA_16bins_5_100_'+str(len(points_3D))+'_sim'+sim_label+'_'+str(n_batches)+'batches_'+str(i)+'.npy'
        # check if file exists
        if len(glob.glob(batch_save_path)) > 0:
            pa_rel_binned = np.load(batch_save_path)
            pa_rel_binned_all.append(pa_rel_binned)
            continue
        
        group_batch = multiplet_table[i_start:i_end]
        i_start = i_end
        i_end += int(len(multiplet_table)/n_batches)

        pa_rel_binned = calculate_rel_ang_cartesian_binAverage(ang_tracers = group_batch['center_loc'], ang_values = group_batch['orientation'], 
                                                                    loc_tracers = tracer_points, loc_weights=[1]*len(tracer_points), E_ABS = np.asarray([1]*len(group_batch)),
                                                                    R_bins=R_bins, pimax=pimax_values) 
        pa_rel_binned_all.append(pa_rel_binned)
        if save_intermediate:
            np.save(batch_save_path, pa_rel_binned)
            
    pa_rel_binned_all = np.asarray(pa_rel_binned_all)
    relAng = np.nanmean(pa_rel_binned_all, axis=0)
    relAng_e = np.nanstd(pa_rel_binned_all, axis=0) / np.sqrt(len(pa_rel_binned_all))
    
    results = Table()
    
    results['R_bin_min'] = R_bins[:-1]
    results['R_bin_max'] = R_bins[1:]
    results['relAng_plot'] = relAng
    results['relAng_plot_e'] = relAng_e
    if isinstance(pimax_values, (int, float)):
        results['pimax'] = [pimax_values] * len(R_bins[:-1])
    else:
        results['pimax'] = pimax_values

    save_path = save_directory + '/MIA_16bins_'+str(round(np.min(R_bins)))+'-'+str(round(np.max(R_bins)))+'_'+str(n_batches)+'batches_'+str(sim_label)+'.fits'
    if save_path is not None:
        results.write(save_path, overwrite=True)
        print('Results saved to ', save_path) 
        
    return results
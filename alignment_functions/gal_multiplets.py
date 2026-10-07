import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from astropy.cosmology import LambdaCDM, z_at_value
from alignment_functions.basic_alignment import *
from geometry_functions.coordinate_functions import get_proj_dist
import random
import warnings


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

def find_groups_UnionFind(points, max_n=10, transverse_max=0.5, los_max=6, transverse_min=None,
                          los_location=None, los_mode='radial', los_axis=None):
    '''
    Find groups of nearby points using a Union-Find on pairs within a transverse/LOS distance cut.
    Pair separations are split into transverse and LOS parts by los_pair_separations().

    los_mode, los_location, los_axis: line of sight, see resolve_los() in coordinate_functions.
        'radial' (default; LOS from the observer at los_location, default the origin) or
        'axis' (LOS along los_axis = 'x', 'y' or 'z'; los_mode='z' is shorthand for los_axis='z').
    points is not modified.
    '''
    los = resolve_los(los_mode, los_location, los_axis)
    points = np.asarray(points)

    tree = cKDTree(points)
    max_pair_sep = np.sqrt(transverse_max**2 + los_max**2)
    dd, ii = tree.query(points, distance_upper_bound=max_pair_sep, k=2)

    ii = ii[ii[:, 1] < len(points)]

    transverse_seps, los_seps = los_pair_separations(points[ii[:, 1]], points[ii[:, 0]], **los)
    transverse_seps = np.abs(transverse_seps)
    los_seps = np.abs(los_seps)

    to_remove = (transverse_seps > transverse_max) | (los_seps > los_max)
    if transverse_min is not None:
        to_remove |= (transverse_seps < transverse_min)

    ii = ii[~to_remove]

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


def calculate_2D_group_orientation(points_3d, los_location=None, los_mode='radial', los_axis=None):
    '''
    Calculate the orientation of a group of points in the sky plane perpendicular to the LOS at the
    group centroid. Member positions are measured relative to the multiplet centroid before projection,
    so the result reflects the multiplet's intrinsic shape rather than its bulk position on the sky.

    points_3d: array of shape (n_points, 3)
    los_mode, los_location, los_axis: line of sight, see resolve_los() in coordinate_functions.
        'radial' (default; LOS from the observer at los_location, default the origin, to the centroid) or
        'axis' (LOS along los_axis = 'x', 'y' or 'z'; los_mode='z' is shorthand for los_axis='z').
    returns: orientation angle in radians, measured E of N (sky-plane axes from sky_components()).
    '''
    center = np.mean(points_3d, axis=0)
    offsets_3d = points_3d - center

    points_2d_x, points_2d_y = sky_components(offsets_3d, center, los_mode, los_location, los_axis)

    points_2d = np.column_stack([points_2d_x, points_2d_y])
    r = np.linalg.norm(points_2d, axis=1)
    theta = np.arctan2(points_2d[:, 0], points_2d[:, 1])
    points_complex = r * np.exp(2j * theta)
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


def make_group_catalog(data_catalog, comoving_points=None, transverse_max=1, los_max=6,
                       max_n=100, cosmology=cosmo, transverse_min=None, truez=False,
                       use_sky_coords=True, los_location=None, los_mode='radial', los_axis=None):
    '''
    Find pairs of galaxies, group them, and return a catalog of group properties.

    los_mode, los_location, los_axis: line of sight, used both to find the multiplets and to project
        their shapes; see resolve_los() in coordinate_functions.
        'radial' (default; LOS from the observer at los_location, default the origin, to each multiplet) or
        'axis' (LOS along los_axis = 'x', 'y' or 'z'; los_mode='z' is shorthand for los_axis='z').

    Returns
    -------
    group_table: astropy table with columns 'center_loc', 'orientation', 'n_group',
    'max_dist_to_center', and (if use_sky_coords) 'RA', 'DEC', 'Z'.
    '''
    los = resolve_los(los_mode, los_location, los_axis)
    if comoving_points is None:
        comoving_points = get_cosmo_points(data_catalog, cosmology=cosmology)

    group_indices = find_groups_UnionFind(comoving_points, max_n=max_n,
                                          transverse_max=transverse_max, los_max=los_max,
                                          transverse_min=transverse_min, **los)

    if truez:
        comoving_points = get_cosmo_points(data_catalog, cosmology=cosmology, truez=truez)

    group_table = Table()
    group_table['center_loc'] = [np.mean(comoving_points[cl], axis=0) for cl in group_indices]
    group_table['orientation'] = [
        calculate_2D_group_orientation(comoving_points[cl], **los)
        for cl in group_indices
    ]
    group_table['n_group'] = [len(cl) for cl in group_indices]
    group_table['max_dist_to_center'] = [
        np.max(np.linalg.norm(comoving_points[cl] - group_table['center_loc'][i], axis=1))
        for i, cl in enumerate(group_indices)
    ]
    if use_sky_coords:
        group_table['RA'] = [data_catalog['RA'][gi[0]] for gi in group_indices]
        group_table['DEC'] = [data_catalog['DEC'][gi[0]] for gi in group_indices]
        group_table['Z'] = [np.mean(data_catalog['Z'][gi]) for gi in group_indices]
        if truez:
            group_table['TRUEZ'] = [np.mean(data_catalog['TRUEZ'][gi]) for gi in group_indices]

    return group_table


def get_multiplet_alignment(catalog_for_groups, catalog_for_tracers=None, R_bins=np.logspace(0, 2, 10), pimax=30, cosmology=cosmo, print_progress=False, 
                        n_sky_regions=100, save_path=None, pair_max_los=6, pair_max_transverse=1, pair_min_transverse=None, early_binning=False, keep_intermediate=False,
                        truez=False, intermediate_save_paths=None, return_pair_counts=False, already_multiplets=False, n_jobs=1):
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
    if already_multiplets:
        group_catalog = catalog_for_groups
    else:
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
                                                    intermediate_save_paths=intermediate_save_paths, return_pair_counts=return_pair_counts, n_jobs=n_jobs)
        # reading in the calculated results
        if print_progress:
            print('Reading in region results')
        region_paths = glob.glob(intermediate_save_paths + '*.npy')
        all_pa_rels = np.asarray([np.load(region_path) for region_path in region_paths])
        relAng = np.nanmean(all_pa_rels, axis=0)
        relAng_e = np.nanstd(all_pa_rels, axis=0) / np.sqrt(len(all_pa_rels))
        if return_pair_counts:
            pair_count_paths = glob.glob(intermediate_save_paths + '*_paircounts.npy')
            all_pair_counts = np.asarray([np.load(region_path) for region_path in pair_count_paths])
            n_pairs = np.nansum(all_pair_counts, axis=0)
        # remove intermediate files
        if not keep_intermediate:
            for region_path in region_paths:
                os.remove(region_path)
            if return_pair_counts:
                for region_path in pair_count_paths:
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
        
        sep_bins, relAng, relAng_e, pair_counts_binned = bin_region_results(group_seps, group_paRel, all_weights = weights, R_bins=R_bins, use_sliding_pimax=use_sliding_pimax, 
                                                                            los_sep=group_los, return_pair_counts=return_pair_counts)
    
    results = Table()
    
    results['R_bin_min'] = R_bins[:-1]
    results['R_bin_max'] = R_bins[1:]
    results['relAng_plot'] = relAng
    results['relAng_plot_e'] = relAng_e
    if return_pair_counts:
        results['pair_counts'] = pair_counts_binned
    if isinstance(pimax, (int, float)):
        results['pimax'] = [pimax] * len(R_bins[:-1])
    else:
        results['pimax'] = pimax
    
    if save_path is not None:
        results.write(save_path, overwrite=True)
        print('Results saved to ', save_path) 
        
    return results


def get_multiplet_autocorr(catalog_for_groups, R_bins=np.logspace(0, 2, 10), pimax=30, cosmology=cosmo, print_progress=False,
                        n_sky_regions=100, save_path=None, pair_max_los=6, pair_max_transverse=1, pair_min_transverse=None, early_binning=False, keep_intermediate=False,
                        truez=False, intermediate_save_paths=None, return_pair_counts=False, already_multiplets=False, n_jobs=1):
    '''
    Calculate the autocorrelation of galaxy multiplet orientations (the '++' estimator).
    Measures cos(2*theta_A)*cos(2*theta_B) where theta_A and theta_B are the orientation angles
    of two multiplets relative to the separation vector between them.
    Follows the same pipeline as get_multiplet_alignment().

    Parameters:
    - catalog_for_groups (dict): Catalog used to find groups.
    - cosmology (Cosmology, optional): Cosmology object defining the cosmological parameters. Default is LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714).
    - print_progress (bool, optional): Whether to print progress messages. Default is False.
    - n_sky_regions (int, optional): Number of sky regions to measure autocorrelation (relative to full catalog).
        Regions divided so equal number of groups in each. Error on measurement is standard error of these regions. Default is 100.
    - pimax (float, optional) or (list, length of n_Rbins): Maximum line-of-sight separation for pairs of galaxies in Mpc/h. Default is 30.
    - R_bins (array): Bin edges for projected separation in Mpc/h.
    - save_path (str, optional): Path to save the results. If not provided, the results will not be saved.
    - early_binning: This option will bin the galaxies early on, saving memory. Helpful if running on dense regions (like BGS), but will generally be a bit noisier measurement.

    Returns:
    results (Table): Table with columns 'R_bin_min', 'R_bin_max', 'relAng_plot', 'relAng_plot_e'.
    - R_bin_min, R_bin_max: Edges of the transverse separation bins, Mpc/h.
    - relAang_plot: cos(2*theta_A)*cos(2*theta_B), where theta is the relative angle between multiplet orientation and separation vector, in each bin.
    - relAng_plot_e: Error on relAang_plot, from standard error of measurements in each sky region.
    '''

    comoving_points_groups = get_cosmo_points(catalog_for_groups, cosmology=cosmology)

    if print_progress:
        print('Making group catalog')
    if already_multiplets:
        group_catalog = catalog_for_groups
    else:
        group_catalog = make_group_catalog(catalog_for_groups, comoving_points = comoving_points_groups, cosmology=cosmology,
                                           los_max=pair_max_los, transverse_max=pair_max_transverse, transverse_min=pair_min_transverse, truez=truez)
    print('Number of multiplets found:', len(group_catalog))
    if print_progress:
        print('Measuring autocorrelation')

    loc_tracers = np.asarray(group_catalog['center_loc'])
    tracer_angles = np.asarray(group_catalog['orientation'])
    tracer_weights = np.ones(len(group_catalog))

    if early_binning:
        if intermediate_save_paths is None:
            try:
                intermediate_save_paths = save_path.split('.fits')[0]
            except:
                print('Save path must be provided to use early binning')
                return None

        rel_angle_regions_binned(group_catalog, loc_tracers = loc_tracers,  tracer_weights = tracer_weights,
                                                    R_bins=R_bins, n_regions=n_sky_regions, pimax=pimax, keep_as_regions=False, print_progress=print_progress,
                                                    intermediate_save_paths=intermediate_save_paths, return_pair_counts=return_pair_counts, n_jobs=n_jobs,
                                                    tracer_angles=tracer_angles)
        # reading in the calculated results
        if print_progress:
            print('Reading in region results')
        region_paths = glob.glob(intermediate_save_paths + '*.npy')
        all_pa_rels = np.asarray([np.load(region_path) for region_path in region_paths])
        relAng = np.nanmean(all_pa_rels, axis=0)
        relAng_e = np.nanstd(all_pa_rels, axis=0) / np.sqrt(len(all_pa_rels))
        if return_pair_counts:
            pair_count_paths = glob.glob(intermediate_save_paths + '*_paircounts.npy')
            all_pair_counts = np.asarray([np.load(region_path) for region_path in pair_count_paths])
            n_pairs = np.nansum(all_pair_counts, axis=0)
        # remove intermediate files
        if not keep_intermediate:
            for region_path in region_paths:
                os.remove(region_path)
            if return_pair_counts:
                for region_path in pair_count_paths:
                    os.remove(region_path)

    else:
        max_proj_sep = np.max(R_bins)
        n_Rbins = len(R_bins) - 1

        # if pimax is not a single value...
        if isinstance(pimax, (int, float)):
            group_seps, group_paRel, weights = rel_angle_regions(group_catalog, loc_tracers = loc_tracers, tracer_weights = tracer_weights,
                                                            n_regions=n_sky_regions, pimax=pimax, max_proj_sep=max_proj_sep, return_los=False,
                                                            tracer_angles=tracer_angles)
            group_los = None
            use_sliding_pimax = False
        else:
            group_seps, group_paRel, weights, group_los = rel_angle_regions(group_catalog, loc_tracers = loc_tracers, tracer_weights = tracer_weights,
                                                            n_regions=n_sky_regions, pimax=np.max(pimax), max_proj_sep=max_proj_sep, return_los=True,
                                                            tracer_angles=tracer_angles)
            use_sliding_pimax = True

        sep_bins, relAng, relAng_e, pair_counts_binned = bin_region_results(group_seps, group_paRel, all_weights = weights, R_bins=R_bins, use_sliding_pimax=use_sliding_pimax,
                                                                            los_sep=group_los, return_pair_counts=return_pair_counts)

    results = Table()

    results['R_bin_min'] = R_bins[:-1]
    results['R_bin_max'] = R_bins[1:]
    results['relAng_autocorr_plot'] = relAng
    results['relAng_autocorr_plot_e'] = relAng_e
    if return_pair_counts:
        results['pair_counts'] = pair_counts_binned
    if isinstance(pimax, (int, float)):
        results['pimax'] = [pimax] * len(R_bins[:-1])
    else:
        results['pimax'] = pimax

    if save_path is not None:
        results.write(save_path, overwrite=True)
        print('Results saved to ', save_path)

    return results


def get_multiplet_alignment_randoms(catalog_for_groups, random_catalog_paths, R_bins, pimax=30, cosmology=cosmo, print_progress=False,
                        n_sky_regions=100, save_path=None, pair_max_los=6, pair_max_transverse=1, pair_min_transverse=None, early_binning=False,
                        keep_intermediate=False, intermediate_save_paths=None, return_pair_counts=False):
    '''
    Simillar to get_multiplet_alignment, but calculates the alignment of galaxy multiplets within the given catalog relative to multiple random catalogs.
    random_catalog_paths: list of paths to random catalogs. 
    If input for random_catalog_paths is an integer, code will automatically generate that many random catalogs from the data by shuffling Z.
    '''
    if save_path is None:
        raise ValueError('save_path must be provided')
    
    group_catalog = make_group_catalog(catalog_for_groups, cosmology=cosmology, los_max=pair_max_los, transverse_max=pair_max_transverse, transverse_min=pair_min_transverse)

    rand_signal = []
    rand_pair_counts = []
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
                                                        intermediate_save_paths=intermediate_save_paths, return_pair_counts=return_pair_counts)
            # reading in the calculated results
            if print_progress:
                print('Reading in region results')
            region_paths = glob.glob(intermediate_save_paths + '*.npy')
            all_pa_rels = np.asarray([np.load(region_path) for region_path in region_paths])
            relAng = np.nanmean(all_pa_rels, axis=0)
            #relAng_e = np.nanstd(all_pa_rels, axis=0) / np.sqrt(len(all_pa_rels))
            # remove intermediate files
            if return_pair_counts:
                pair_count_paths = glob.glob(intermediate_save_paths + '*_paircounts.npy')
                all_pair_counts = np.asarray([np.load(region_path) for region_path in pair_count_paths])
                n_pairs = np.nansum(all_pair_counts, axis=0)
            # remove intermediate files
            if not keep_intermediate:
                for region_path in region_paths:
                    os.remove(region_path)
                if return_pair_counts:
                    for region_path in pair_count_paths:
                        os.remove(region_path)
            
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
            
            sep_bins, relAng, relAng_e, pair_counts_binned = bin_region_results(group_seps, group_paRel, all_weights = weights, R_bins=R_bins, use_sliding_pimax=use_sliding_pimax, 
                                                                                    los_sep=group_los, return_pair_counts=return_pair_counts)
            rand_signal.append(relAng)
            if return_pair_counts:
                rand_pair_counts.append(pair_counts_binned)


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
    if return_pair_counts:
        results['pair_counts'] = np.sum(np.asarray(rand_pair_counts), axis=0)
    results.write(save_path, overwrite=True)
    print('Results saved to ', save_path) 
    
    
def get_group_2pt_projected_corr(catalog, random_paths, catalog2=None, tracer_catalog=None, rp_bins=np.logspace(0, np.log10(150), 11), rpar_bins=np.linspace(0, 80, 101), 
                                 pair_max_los=1, pair_max_transverse=1, use_sliding_pimax=False, print_progress=False, save_path=None):    
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
    
    catalog2 =  make_group_catalog(catalog, los_max=pair_max_los, transverse_max=pair_max_transverse)
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

def _process_one_3D_batch(batch_index, group_batch_center_loc, group_batch_orientation, group_batch_n,
                          tracer_points, R_bins, pimax_values, return_pair_counts,
                          save_intermediate, batch_save_path, print_info, sim_label, los):
    # los: canonical LOS dict from resolve_los(); must be the one used to build the multiplet orientations
    if batch_index % 50 == 0 and print_info:
        print('working on batch', batch_index, 'sim:', sim_label, flush=True)

    if os.path.exists(batch_save_path):
        pa_rel_binned = np.load(batch_save_path)
        pair_counts = None
        if return_pair_counts:
            pair_counts_save_path = batch_save_path.replace('.npy', '_paircounts.npy')
            if os.path.exists(pair_counts_save_path):
                pair_counts = np.load(pair_counts_save_path)
        return batch_index, pa_rel_binned, pair_counts

    pa_rel_binned = calculate_rel_ang_cartesian_binAverage(
        ang_tracers=group_batch_center_loc, ang_values=group_batch_orientation,
        loc_tracers=tracer_points, loc_weights=[1]*len(tracer_points),
        E_ABS=np.ones(group_batch_n),
        R_bins=R_bins, pimax=pimax_values, return_pair_counts=return_pair_counts, **los)

    pair_counts = None
    if return_pair_counts:
        pa_rel_binned, pair_counts = pa_rel_binned
        if save_intermediate:
            pair_counts_save_path = batch_save_path.replace('.npy', '_paircounts.npy')
            np.save(pair_counts_save_path, pair_counts)

    if save_intermediate:
        np.save(batch_save_path, pa_rel_binned)

    return batch_index, pa_rel_binned, pair_counts


def get_MIA_from3D(points_3D, save_directory, R_bins = np.logspace(np.log10(5), np.log10(100), 16), pimax='variable', transverse_max = 1, los_max=1, print_info=True, sim_label='example',
                   periodic_boundary=False, n_batches = 10, save_intermediate=False, save_info=False,
                   return_pair_counts=False, n_jobs=1, los_mode=None, los_location=None, los_axis=None, box_size=None):
    '''
    A high-level function to calculate projected multiplet alignment for a set of points in 3D comoving space.
    Input points and parameters can be in any units as long as they are consistent.

    The line of sight (LOS) must be given explicitly. The same LOS is used to find the multiplets, project
    their shapes, and measure projected separations, LOS separations and position angles of the tracers:
      los_mode='axis', los_axis='x', 'y' or 'z':   every LOS is parallel to that axis (e.g. RSD applied along that axis).
      los_mode='radial', los_location=[x, y, z]:  every LOS points from the observer at los_location to the object
                                                  (e.g. RSD applied radially from that observer). los_location is in the
                                                  same coordinates as points_3D. Avoid an observer along +/-z of the
                                                  points (see get_orientation_angle_cartesian).
    -----------
    points_3D: x, y, z positions of points. type: array of shape (n_points, 3). Not modified.
    R_bins: bin edges of the transverse separation for the final measurement
    pimax: maximum line-of-sight separation for pairs of galaxies in Mpc/h. Can be a single value or an array of the same length as R_bins-1 for variable pimax. 
            default is 'variable', which uses pimax = 8 + (2/3)*R_bin_middles. From https://arxiv.org/pdf/2504.16076
    transverse_max, los_max: maximum transverse and line-of-sight separation for points to be considered in the same multiplet, in the same units as points_3D. default is 1 for each.
    sim_label (optional): number to keep track of running multiple sims
    periodic_boundary (optional): whether to use periodic boundary conditions. If True, will extend the box by adding copies of the points from each side.
    box_size (optional): side length(s) of the periodic box, scalar or length 3. Default is the extent of points_3D along each axis.
    n_jobs (int, optional): Number of parallel worker processes for batches, via joblib. Default is 1 (sequential). Set to -1 to use all available cores.
    los_mode, los_location, los_axis: line of sight, see above (and resolve_los() in coordinate_functions).

    Returns:
    Astropy table with columns:
    'R_bin_min', 'R_bin_max': edges of the transverse separation bins, in the same units as points_3D.
    'pimax': the pimax used for each bin, in the same units as points_3D.
    'relAng_plot', 'relAng_plot_e': measured alignment signal and error in each bin.
    'pair_counts' (if return_pair_counts): number of pairs in each bin, if return_pair_counts is True.
    The LOS is recorded in the table header (LOSMODE, and LOSAXIS or LOSOBSX/Y/Z) and in the saved file names.
    '''
    # validate the LOS once; `los` is then passed unchanged to every LOS-dependent step
    if los_mode is None:
        raise ValueError("Choose a line of sight: los_mode='axis' with los_axis='x', 'y' or 'z', or "
                         "los_mode='radial' with los_location=[x, y, z] (the observer, in the coordinates of points_3D). "
                         "Earlier versions used a radial LOS from the minimum corner of the box, "
                         "i.e. los_mode='radial', los_location=np.min(points_3D, axis=0).")
    los = resolve_los(los_mode, los_location, los_axis)
    if los['los_mode'] == 'radial' and los_location is None:
        raise ValueError("los_mode='radial' needs los_location (the observer position)")
    if los['los_mode'] == 'axis' and los_location is not None:
        raise ValueError("los_location is only used with los_mode='radial'; set the axis with los_axis")
    if los['los_mode'] == 'axis':
        los_tag = 'los' + 'xyz'[los['los_axis']]
    else:
        los_tag = 'losradial' + '_'.join('%g' % v for v in los['los_location'])

    R_bin_middles = (R_bins[1:] + R_bins[:-1])/2
    if isinstance(pimax, str):
        if pimax != 'variable':
            raise ValueError("pimax must be 'variable', a number, or an array of length len(R_bins)-1")
        pimax_values = 8 + (2/3)*R_bin_middles
    elif np.ndim(pimax) == 0:
        pimax_values = float(pimax)
    else:
        pimax_values = np.asarray(pimax, dtype=float)
        if pimax_values.shape != R_bin_middles.shape:
            raise ValueError("pimax must be 'variable', a number, or an array of length len(R_bins)-1")

    points_3D = np.asarray(points_3D)

    if print_info:
        print('Calculating MIA for %d points' % len(points_3D), 'with LOS', los_tag)
        
    if print_info:
        print('Finding multiplets')
    
    multiplet_table = make_group_catalog(None, comoving_points=points_3D, transverse_max=transverse_max, los_max=los_max, max_n=100, use_sky_coords=False, **los)
    if len(multiplet_table) < n_batches:
        raise ValueError('Found %d multiplets, fewer than n_batches=%d' % (len(multiplet_table), n_batches))
    if print_info:
        print('Found %d multiplets' % len(multiplet_table), 'averange number of members: %f' % np.mean(multiplet_table['n_group']))  
    # print binned counts of multiplet sizes
    if print_info:
        print('Multiplet size counts:') 
        unique, counts = np.unique(multiplet_table['n_group'], return_counts=True)
        print('size:', [u for u in np.asarray(unique)])
        print('counts:', [c for c in np.asarray(counts)])
    
    if save_info:
        # save a short text file with multiplet info
        multiplet_info_path = save_directory + '/multiplet_info_'+sim_label+'.txt'
        with open(multiplet_info_path, 'w') as f:
            f.write('Number of multiplets: %d\n' % len(multiplet_table))
            f.write('Average number of members: %f\n' % np.mean(multiplet_table['n_group']))
            unique, counts = np.unique(multiplet_table['n_group'], return_counts=True)
            f.write('Multiplet size counts:\n')
            for u, c in zip(unique, counts):
                f.write('Size: %d, Count: %d\n' % (u, c))

    if periodic_boundary:
        if los['los_mode'] == 'radial':
            warnings.warn("periodic_boundary with a radial LOS: the periodic copies keep the RSD of the original points, "
                          "which was applied along the original points' LOS, not along the copies' LOS.")
        box_min = np.min(points_3D, axis=0)
        box_max = np.max(points_3D, axis=0)
        if box_size is None:
            box_size = box_max - box_min
            if print_info:
                print('Using box_size =', box_size)
        box_size = np.broadcast_to(np.asarray(box_size, dtype=float), (3,))
        # make an array with every comination of adding or subtracting the box size to each dimmension
        new_orgins = np.array([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]) * box_size
        extended_points = np.concatenate([points_3D + new_orgin for new_orgin in new_orgins], axis=0)

        # trim to points that can pair with a point in the original box: within max(R_bins) transverse and
        # max(pimax) along the LOS (along the LOS axis for 'axis'; in every direction for 'radial')
        if los['los_mode'] == 'axis':
            extend_by = np.full(3, np.max(R_bins))
            extend_by[los['los_axis']] = np.max(pimax_values)
        else:
            extend_by = np.full(3, np.sqrt(np.max(R_bins)**2 + np.max(pimax_values)**2))
        i_keep = np.all((extended_points > box_min - extend_by) & (extended_points < box_max + extend_by), axis=1)
        tracer_points = extended_points[i_keep]
    else:
        tracer_points = points_3D


    # order group table randomly (but reproducibly)
    random.seed(42)
    indices = np.asarray(range(len(multiplet_table)))
    random.shuffle(indices)
    multiplet_table = multiplet_table[indices]

    results_base_path = save_directory + '/MIA_'+str(len(R_bins))+'bins_'+str(round(np.min(R_bins), 3))+'_'+str(round(np.max(R_bins), 3))+'_counts'+str(len(points_3D))+'_sim'+sim_label+'_'+los_tag+'_'+str(n_batches)+'batches_'
    print('Results base path:', results_base_path)

    batch_size = int(len(multiplet_table)/n_batches)
    batch_specs = []
    for i in range(int(n_batches)):
        i_start = i * batch_size
        i_end = i_start + batch_size
        group_batch = multiplet_table[i_start:i_end]
        batch_save_path = results_base_path + str(i)+'.npy'
        batch_specs.append({
            'batch_index': i,
            'center_loc': np.asarray(group_batch['center_loc']),
            'orientation': np.asarray(group_batch['orientation']),
            'n': len(group_batch),
            'batch_save_path': batch_save_path,
        })

    tracer_points = np.ascontiguousarray(tracer_points)

    if print_info:
        print('Dispatching', len(batch_specs), 'batches across', n_jobs, 'workers', flush=True)

    from joblib import Parallel, delayed
    results_list = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_one_3D_batch)(
            spec['batch_index'], spec['center_loc'], spec['orientation'], spec['n'],
            tracer_points, R_bins, pimax_values, return_pair_counts,
            save_intermediate, spec['batch_save_path'], print_info, sim_label, los,
        )
        for spec in batch_specs
    )

    results_list.sort(key=lambda r: r[0])
    pa_rel_binned_all = [r[1] for r in results_list]
    pair_counts_all = [r[2] for r in results_list if r[2] is not None]
            
    pa_rel_binned_all = np.asarray(pa_rel_binned_all)
    relAng = np.nanmean(pa_rel_binned_all, axis=0)
    relAng_e = np.nanstd(pa_rel_binned_all, axis=0) / np.sqrt(len(pa_rel_binned_all))
    if return_pair_counts:
        pair_counts_all = np.asarray(pair_counts_all)
        pair_counts_binned = np.nansum(pair_counts_all, axis=0)
    
    results = Table()
    
    results['R_bin_min'] = R_bins[:-1]
    results['R_bin_max'] = R_bins[1:]
    results['relAng_plot'] = relAng
    results['relAng_plot_e'] = relAng_e
    if isinstance(pimax_values, (int, float)):
        results['pimax'] = [pimax_values] * len(R_bins[:-1])
    else:
        results['pimax'] = pimax_values
        
    if return_pair_counts:
        results['pair_counts'] = pair_counts_binned

    results.meta['LOSMODE'] = los['los_mode']
    if los['los_mode'] == 'axis':
        results.meta['LOSAXIS'] = 'xyz'[los['los_axis']]
    else:
        results.meta['LOSOBSX'], results.meta['LOSOBSY'], results.meta['LOSOBSZ'] = [float(v) for v in los['los_location']]

    save_path = results_base_path+'.fits'
    if save_path is not None:
        results.write(save_path, overwrite=True)
        print('Results saved to ', save_path) 
        
    return results
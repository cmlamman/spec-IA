import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.io import fits
from alignment_functions.basic_alignment import *
from alignment_functions.gal_multiplets import *

'''
Functions for measuring alignment in 3D and also the autocorrelation (orientation - orientation).
'''


def get_angle_angle_correlation_fromSky(catalog, indices, use_weights=False, return_sep=False, ellipticity_component_to_use = 1):
    '''
    input: 
        catalog: astropy table with columns 'RA', 'DEC', 'E1', 'E2' and, optionally, 'WEIGHT' if use_weights is True.
        *** this is designed to measure angle-angle correlations, but the absolute absolute value of ellipticity can be included in the weight to achieve full-shape correlations. ***
        indices (array of shape NxM): indices pointing to pairs within the catalaog. Structured as N objects with a maximum of M neighbors each.
        ellipticity_component_to_use (int, 1 or 2): Whether to use the real (E1) or imaginary (E2) component of the ellipticity for angle calculation.
    output:
        rel_angs (array of shape P): angle-angle correlation, in array of shape P where P is the number of unique pairs.
        rel_seps [optional] (array of shape Px2): 2d separation between pairs in r_p, r_par. r_p is the distance projected on the same plane as the provided angles and r_par is the line-of-sight separation.
    '''
    
    # since the array 
    
    return None


def get_angle_angle_correlation_cartesian(ang_locs_0, ang_values_0, ang_locs_1=None, ang_values_1=None, weights_0=None, weights_1=None, print_progress=False, max_rpar=100, max_rp=100, estimator='x+'):
    '''
    Assumes angles are in plane perpendicular to LOS (vector from orgin to object position).
    input: 
        ang_locs_0 (array of shape Nx3): objects' 3D positions
        ang_values_0 (array of shape N): objects' orientation angles (in radians)
        ang_locs_1 and ang_values_1 (optional): if provided, will compute cross-correlation between the two sets of points. Useful if breaking up data but want to preserve correlation across boundaries, i.e. _0 is a subset of the full catalog _1.
        weights_0 and weights_1 (bool or array of shape N): can supply absolute value of ellipticity to use relative full ellipticity.
    output:
        rel_angs (array of shape P): angle-angle correlation, in array of shape P where P is the number of unique pairs.
        rel_pos (array of shape 2xP): 2D separation vector between pairs in s_perp, s_par. s_perp is the distance projected on the same plane as the provided angles and s_par is the line-of-sight separation.
    '''
    
    if ang_locs_1 is None:
        ang_locs_1 = ang_locs_0.copy()
        ang_values_1 = ang_values_0.copy()
        weights_1 = weights_0.copy() if weights_0 is not None else None
    
    if print_progress: print('making tree')
    # make tree
    tree = cKDTree(ang_locs_1)  # 'full' catalog
    # find neighbors
    if print_progress: print('finding neighbors')
    ii = tree.query_ball_point(ang_locs_0, r=np.sqrt(max_rp**2 + max_rpar**2))  # query with main objects
    if print_progress: print('found neighbors')
        
    indices0 = [len(i) for i in ii]   # for now I will not remove the double inclusion of pairs since I think that whether a point is in front of or behind another might matter for parity and cross-boundary pairs
    indices1 = np.concatenate(ii)

    coords0 = np.repeat(ang_locs_0, indices0, axis=0)
    angles0 = np.repeat(ang_values_0, indices0)
    if weights_0 is not None:
        weights0 = np.repeat(weights_0, indices0)
    # add placeholder row to ang_locs
    ang_locs_1 = np.vstack((ang_locs_1, np.full(len(ang_locs_1[0]), np.inf)))
    if weights_1 is not None:
        weights_1 = np.append(weights_1, 0)
        weights1 = weights_1[indices1]
    coords1 = ang_locs_1[indices1]
    angles1 = ang_values_1[indices1]
    
    # computing los separation (broken up to save memory
    # L
    nc = coords0**2
    nc = np.sum(nc, axis=1)
    dist_to_orgin_0 = np.sqrt(nc)
    cc = coords1**2
    cc = np.sum(cc, axis=1)
    dist_to_orgin_1 = np.sqrt(cc)
    
    if print_progress: print('calculating separations')
    los_sep = dist_to_orgin_0 - dist_to_orgin_1 # previously was abs() - add back in later?
    proj_dist = np.abs(get_proj_dist(coords0, coords1)) # previously was abs() - add back in later?

    pairs_to_keep = (los_sep < max_rpar) & (proj_dist < max_rp)
    coords0 = coords0[pairs_to_keep]
    coords1 = coords1[pairs_to_keep]
    angles0 = angles0[pairs_to_keep]
    angles1 = angles1[pairs_to_keep]
    if weights_0 is not None:
        weights_0 = weights_0[pairs_to_keep]
    else:
        weights0 = None
    if weights_1 is not None:
        weights_1 = weights_1[pairs_to_keep]
    else:
        weights1 = None
        
        
    # calculating position angles
    pa0 = get_orientation_angle_cartesian(coords0, coords1)
    pa1 = get_orientation_angle_cartesian(coords1, coords0)
    
    pa_rel0 = angles0 - pa0
    pa_rel1 = angles1 - pa1
    
    if estimator == 'x+' or estimator == '+x':
        rel_angs = np.sin(2*pa_rel0)*np.cos(2*pa_rel1)
    elif estimator == '++':
        rel_angs = np.cos(2*pa_rel0)*np.cos(2*pa_rel1)
    elif estimator == 'g+' or estimator == '+g':
        rel_angs = np.cos(2*pa_rel0)
    elif estimator == 'gg':
        rel_angs = np.ones_like(pa_rel0)  # just counts
    else:
        raise ValueError('Estimator not recognized. Allowed: x+, ++, g+, gg')
    #if weights_0 is not None:
    #    rel_angs *= weights0
    #if weights_1 is not None:
    #    rel_angs *= weights1
    
    s_perp = proj_dist[pairs_to_keep]
    s_par  = los_sep[pairs_to_keep]
    rel_pos = np.vstack((s_perp, s_par))
    
    return rel_angs, rel_pos, weights0, weights1



def get_3D_MIA_from3D_autocorr(points_3D, save_directory, 
                      s_bins = np.logspace(np.log10(5), np.log10(100), 16), 
                      mu_bins = np.cos(np.linspace(np.pi, 0, 17)),
                      transverse_max = 1, los_max=1,
                      print_info=True, sim_label='example_3D',
                      periodic_boundary=False, n_batches = 10, save_intermediate=False, estimator='x+'):
    '''
    A high-level function to calculate 3D multiplet alignment for a set of points in 3D comoving space.
    Input points and parameters can be in any units as long as they are consistent.
    -----------
    points_3D: array of shape Nx3 with the 3D comoving positions of points
    s_bins: bin edges of the separation for the final measurement
    mu_bins: bin edges of the cosine of the angle between the separation vector and the line of sight
    transverse_max and los_max: maximum transverse and line-of-sight separations to consider when finding multiplet members
    sim_label (optional): number to keep track of running multiple sims
    periodic_boundary (optional): whether to use periodic boundary conditions. If True, will extend the box by adding copies of the points from each side.
    '''
    
    # create a string which describes the provided bins, for saving later
    bin_string = 's_'+str(round(np.min(s_bins)))+'-'+str(round(np.max(s_bins)))+'_mu_'+str(round(np.min(mu_bins*100)))+'-'+str(round(np.max(mu_bins*100)))
    
    
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
            
        extend_by = np.max(s_bins)
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
        batch_save_path = save_directory + '/MIA_sim'+estimator+'_'+sim_label+'_'+bin_string+'_'+str(n_batches)+'_batches_'+str(i)+'.npy'
        # check if file exists
        if len(glob.glob(batch_save_path)) > 0:
            pa_rel_binned = np.load(batch_save_path)
            pa_rel_binned_all.append(pa_rel_binned)
            continue
        
        group_batch = multiplet_table[i_start:i_end]
        i_start = i_end
        i_end += int(len(multiplet_table)/n_batches)

        pa_rel_unbinned, separations_unbinned = get_angle_angle_correlation_cartesian(group_batch['center_loc'], group_batch['orientation'], print_progress=print_info, max_rpar=np.max(s_bins), max_rp=np.max(s_bins), estimator=estimator)
        
        sep = np.asarray(separations_unbinned)
        # expect shape (2, N)
        rp = sep[0, :]
        rpar = sep[1, :]

        s_unbinned = np.sqrt(rp**2 + rpar**2)
        mu_unbinned = np.zeros_like(s_unbinned, dtype=float)
        nz = s_unbinned > 0
        mu_unbinned[nz] = rpar[nz] / s_unbinned[nz]
        
        # bin the results
        pa_rel_binned, _, _ = np.histogram2d(s_unbinned, mu_unbinned, bins=[s_bins, mu_bins], weights=pa_rel_unbinned)
        # also get the counts in each bin
        counts_binned, _, _ = np.histogram2d(s_unbinned, mu_unbinned, bins=[s_bins, mu_bins])
        pa_rel_binned /= counts_binned

        pa_rel_binned_all.append(pa_rel_binned)
        if save_intermediate:
            np.save(batch_save_path, pa_rel_binned)
            
    pa_rel_binned_all = np.asarray(pa_rel_binned_all)
    relAng = np.nanmean(pa_rel_binned_all, axis=0)
    relAng_e = np.nanstd(pa_rel_binned_all, axis=0) / np.sqrt(len(pa_rel_binned_all))
    
    # save these 2D results in a fits file
    save_path = save_directory + '/MIA_sim_'+estimator+'_'+sim_label+'_'+str(n_batches)+'_batches_all.fits'
    
    hdu_rel = fits.PrimaryHDU(relAng.astype(np.float32))
    hdu_err = fits.ImageHDU(relAng_e.astype(np.float32), name=estimator+'_ERR')
    hdu_sbins = fits.ImageHDU(np.asarray(s_bins, dtype=np.float32), name='S_BINS')
    hdu_mubins = fits.ImageHDU(np.asarray(mu_bins, dtype=np.float32), name='MU_BINS')
    hdul = fits.HDUList([hdu_rel, hdu_err, hdu_sbins, hdu_mubins])
    
    hdul.writeto(save_path, overwrite=True)
    print('Results saved to ', save_path) 

    return relAng, relAng_e
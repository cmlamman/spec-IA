import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.io import fits
from alignment_functions.basic_alignment import *
from alignment_functions.gal_multiplets import *

'''
Functions for measuring alignment in 3D and also the autocorrelation (orientation - orientation).
'''


def get_angle_angle_correlation_cartesian(ang_locs_0, ang_values_0, ang_locs_1=None, ang_values_1=None,
                                          weights_0=None, weights_1=None, print_progress=False,
                                          max_rpar=100, max_rp=100, estimator='x+', los_mode='z',
                                          los_location=np.asarray([0, 0, 0])):
    '''
    Angle-angle correlation in 3D. Default LOS is plane-parallel along +z (appropriate for cubic
    simulation boxes); pass los_mode='radial' with los_location for a survey-like geometry.

    output:
        rel_angs (array of shape P): angle-angle correlation
        rel_pos (array of shape 2xP): signed (s_perp, s_par). s_par sign carries parity-odd info.
    '''
    if ang_locs_1 is None:
        ang_locs_1 = ang_locs_0.copy()
        ang_values_1 = ang_values_0.copy()
        weights_1 = weights_0.copy() if weights_0 is not None else None

    if print_progress: print('making tree')
    tree = cKDTree(ang_locs_1)
    if print_progress: print('finding neighbors')
    ii = tree.query_ball_point(ang_locs_0, r=np.sqrt(max_rp**2 + max_rpar**2))
    if print_progress: print('found neighbors')

    indices0 = [len(i) for i in ii]
    indices1 = np.concatenate(ii)

    coords0 = np.repeat(ang_locs_0, indices0, axis=0)
    angles0 = np.repeat(ang_values_0, indices0)
    if weights_0 is not None:
        weights0 = np.repeat(weights_0, indices0)
    ang_locs_1 = np.vstack((ang_locs_1, np.full(len(ang_locs_1[0]), np.inf)))
    if weights_1 is not None:
        weights_1 = np.append(weights_1, 0)
        weights1 = weights_1[indices1]
    coords1 = ang_locs_1[indices1]
    angles1 = ang_values_1[indices1]

    if print_progress: print('calculating separations')

    if los_mode == 'z':
        # plane-parallel LOS along +z: signed los_sep is just the z-component of the separation
        los_sep = coords0[:, 2] - coords1[:, 2]
        proj_dist = np.abs(get_proj_dist(coords0, coords1, los_mode='z'))
    elif los_mode == 'radial':
        dist_to_orgin_0 = np.sqrt(np.sum(coords0**2, axis=1))
        dist_to_orgin_1 = np.sqrt(np.sum(coords1**2, axis=1))
        los_sep = dist_to_orgin_0 - dist_to_orgin_1
        proj_dist = np.abs(get_proj_dist(coords0, coords1, pos_obs=los_location, los_mode='radial'))
    else:
        raise ValueError("los_mode must be 'radial' or 'z'")

    pairs_to_keep = (np.abs(los_sep) < max_rpar) & (proj_dist < max_rp)
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

    pa0 = get_orientation_angle_cartesian(coords0, coords1,
                                          los_location=los_location, los_mode=los_mode)
    pa1 = get_orientation_angle_cartesian(coords1, coords0,
                                          los_location=los_location, los_mode=los_mode)

    pa_rel0 = angles0 - pa0
    pa_rel1 = angles1 - pa1

    if estimator == 'x+' or estimator == '+x':
        rel_angs = np.sin(2*pa_rel0)*np.cos(2*pa_rel1)
    elif estimator == '++':
        rel_angs = np.cos(2*pa_rel0)*np.cos(2*pa_rel1)
    elif estimator == 'g+' or estimator == '+g':
        rel_angs = np.cos(2*pa_rel0)
    elif estimator == 'gg':
        rel_angs = np.ones_like(pa_rel0)
    else:
        raise ValueError('Estimator not recognized. Allowed: x+, ++, g+, gg')

    s_perp = proj_dist[pairs_to_keep]
    s_par  = los_sep[pairs_to_keep]
    rel_pos = np.vstack((s_perp, s_par))

    return rel_angs, rel_pos, weights0, weights1



def get_3D_MIA_from3D_autocorr(points_3D, save_directory,
                               s_bins=np.logspace(np.log10(5), np.log10(100), 16),
                               mu_bins=np.cos(np.linspace(np.pi, 0, 17)),
                               transverse_max=1, los_max=1,
                               print_info=True, sim_label='example_3D',
                               periodic_boundary=False, n_batches=10,
                               save_intermediate=False, estimator='x+',
                               los_mode='z', los_location=np.asarray([0, 0, 0])):
    '''
    High-level driver for 3D multiplet alignment in a sim box.
    Default LOS is plane-parallel along +z. Pass los_mode='radial' (with optional los_location)
    for a survey-like radial-LOS geometry.
    '''
    bin_string = 's_'+str(round(np.min(s_bins)))+'-'+str(round(np.max(s_bins)))+'_mu_'+str(round(np.min(mu_bins*100)))+'-'+str(round(np.max(mu_bins*100)))

    if print_info:
        print('Calculating MIA for %d points' % len(points_3D))
        print('Finding multiplets')

    points_3D -= np.min(points_3D, axis=0)
    multiplet_table = make_group_catalog(None, comoving_points=points_3D,
                                         transverse_max=transverse_max, los_max=los_max,
                                         max_n=100, use_sky_coords=False,
                                         los_location=los_location, los_mode=los_mode)
    if print_info:
        print('Found %d multiplets' % len(multiplet_table),
              'average number of members: %f' % np.mean(multiplet_table['n_group']))
        print('Multiplet size counts:')
        unique, counts = np.unique(multiplet_table['n_group'], return_counts=True)
        print('size:', [u for u in np.asarray(unique)])
        print('counts:', [c for c in np.asarray(counts)])

    if periodic_boundary:
        extend_by = np.max(s_bins)
        new_orgins = np.array([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]) * np.max(points_3D[:, 0])
        extended_points = np.array([points_3D + new_orgins[i] for i in range(len(new_orgins))])
        extended_points = np.concatenate(extended_points, axis=0)
        i_keep = (extended_points[:, 0] > np.min(points_3D[:, 0])-extend_by) & (extended_points[:, 0] < np.max(points_3D[:, 0])+extend_by)
        i_keep &= (extended_points[:, 1] > np.min(points_3D[:, 1])-extend_by) & (extended_points[:, 1] < np.max(points_3D[:, 1])+extend_by)
        i_keep &= (extended_points[:, 2] > np.min(points_3D[:, 2])-extend_by) & (extended_points[:, 2] < np.max(points_3D[:, 2])+extend_by)
        tracer_points = extended_points[i_keep]
    else:
        tracer_points = points_3D

    random.seed(42)
    indices = np.asarray(range(len(multiplet_table)))
    random.shuffle(indices)
    multiplet_table = multiplet_table[indices]

    i_end = int(len(multiplet_table)/n_batches)
    i_start = 0

    pa_rel_binned_all = []
    for i in range(int(n_batches)):
        if i % 24 == 0 and print_info:
            print('working on batch', i, 'sim:', sim_label)
        batch_save_path = save_directory + '/MIA_sim'+estimator+'_'+sim_label+'_'+bin_string+'_'+str(n_batches)+'_batches_'+str(i)+'.npy'
        if len(glob.glob(batch_save_path)) > 0:
            pa_rel_binned = np.load(batch_save_path)
            pa_rel_binned_all.append(pa_rel_binned)
            continue

        group_batch = multiplet_table[i_start:i_end]
        i_start = i_end
        i_end += int(len(multiplet_table)/n_batches)

        pa_rel_unbinned, separations_unbinned, weights0, weights1 = get_angle_angle_correlation_cartesian(
            group_batch['center_loc'], group_batch['orientation'],
            print_progress=print_info, max_rpar=np.max(s_bins), max_rp=np.max(s_bins),
            estimator=estimator, los_mode=los_mode, los_location=los_location)

        sep = np.asarray(separations_unbinned)
        rp = sep[0, :]
        rpar = sep[1, :]

        s_unbinned = np.sqrt(rp**2 + rpar**2)
        mu_unbinned = np.zeros_like(s_unbinned, dtype=float)
        nz = s_unbinned > 0
        mu_unbinned[nz] = rpar[nz] / s_unbinned[nz]

        pa_rel_binned, _, _ = np.histogram2d(s_unbinned, mu_unbinned, bins=[s_bins, mu_bins], weights=pa_rel_unbinned)
        counts_binned, _, _ = np.histogram2d(s_unbinned, mu_unbinned, bins=[s_bins, mu_bins])
        pa_rel_binned /= counts_binned

        pa_rel_binned_all.append(pa_rel_binned)
        if save_intermediate:
            np.save(batch_save_path, pa_rel_binned)

    pa_rel_binned_all = np.asarray(pa_rel_binned_all)
    relAng = np.nanmean(pa_rel_binned_all, axis=0)
    relAng_e = np.nanstd(pa_rel_binned_all, axis=0) / np.sqrt(len(pa_rel_binned_all))

    save_path = save_directory + '/MIA_sim_'+estimator+'_'+sim_label+'_'+str(n_batches)+'_batches_all.fits'
    hdu_rel = fits.PrimaryHDU(relAng.astype(np.float32))
    hdu_err = fits.ImageHDU(relAng_e.astype(np.float32), name=estimator+'_ERR')
    hdu_sbins = fits.ImageHDU(np.asarray(s_bins, dtype=np.float32), name='S_BINS')
    hdu_mubins = fits.ImageHDU(np.asarray(mu_bins, dtype=np.float32), name='MU_BINS')
    hdul = fits.HDUList([hdu_rel, hdu_err, hdu_sbins, hdu_mubins])
    hdul.writeto(save_path, overwrite=True)
    print('Results saved to ', save_path)

    return relAng, relAng_e
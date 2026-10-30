### FROM CLAUDE ###

"""Check: does splitting the multiplets into spatial regions instead of random
batches change the covariance of the multiplet alignment (MIA) measurement?

get_MIA_from3D() shuffles the multiplets and cuts them into n_batches random
batches. Each batch is correlated with ALL tracers, and the error is
std(batches) / sqrt(n_batches). Every random batch samples the same large-scale
structure, so the batch-to-batch scatter only captures shape noise and may
underestimate the large-scale (cosmic-variance) part of the covariance.

The real-data functions (rel_angle_regions / rel_angle_regions_binned) instead
split the multiplets into contiguous regions on the sky. The simulation analog
used here is to split the box into equal-count columns in the plane transverse
to the LOS (the LOS is the z axis, as in the tutorial), with the same
slice-then-sort scheme as rel_angle_regions (DEC -> y, RA -> x).

What this script does
---------------------
1. Loads the tutorial catalog and finds multiplets once (transverse_max =
   los_max = 1, LOS = z), with the tutorial's R bins (16 log edges, 5-100
   Mpc/h) and variable pimax = 8 + (2/3) R.
2. For every multiplet, computes the sum of cos(2 Phi) and the pair count in
   each R bin against the full (periodically extended) tracer set, using the
   library's own geometry functions. Any partition of the multiplets
   (random batches or spatial regions) is then just a regrouping of these
   per-multiplet sums, so many partitions can be compared at no extra cost.
3. VALIDATION: runs get_MIA_from3D as-is (save_intermediate=True) and checks
   that its saved per-batch .npy files and final table match the regrouped
   per-multiplet sums for the same random shuffle, with no multiplets dropped.
   Also checks, on a small sub-cube, that saved batch files are only reused for
   identical settings (and only with save_intermediate=True), and that a
   mismatched inferred box_size raises a warning.
4. For several numbers of regions N, compares the covariance from
     - random batches (the get_MIA_from3D scheme, averaged over many seeds),
     - transverse regions, same subsample estimator: C = Cov(x_k) / N,
     - transverse regions, delete-one jackknife:
       C = (N-1)/N * sum_k (x_{-k} - mean)(x_{-k} - mean)^T.
   Prints sigma ratios per R bin and saves a figure + .npz with all matrices.

Interpreting the result: if sigma_region / sigma_random is ~1 in all bins,
random batches are fine. Ratios above 1 (growing with R) mean the random
batches miss large-scale covariance. Compare the ratio to the seed-to-seed
scatter of sigma_random (printed) before reading much into small differences.

HOW TO RUN (from spec-IA/, needs joblib + matplotlib)
----------
    python tests/check_covariance_region_vs_random.py
    python tests/check_covariance_region_vs_random.py --skip-validation

The per-multiplet sums are cached in --outdir (delete the cache to recompute).
"""

import argparse, glob, os, random, sys, tempfile, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

from alignment_functions.gal_multiplets import make_group_catalog, get_MIA_from3D
from geometry_functions.coordinate_functions import (resolve_los, los_pair_separations,
                                                      get_orientation_angle_cartesian)

SPEC_IA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(SPEC_IA_DIR, 'example_catalogs', 'abacus_comoving-points.fits')

# settings from basic_multiplet_alignment-sims.ipynb
R_BINS = np.logspace(np.log10(5), np.log10(100), 16)
R_MID = (R_BINS[1:] + R_BINS[:-1]) / 2
PIMAX_VALUES = 8 + (2/3) * R_MID          # pimax='variable' in get_MIA_from3D
TRANSVERSE_MAX, LOS_MAX = 1, 1
LOS = resolve_los('axis', los_axis='z')
# AbacusSummit base box. The catalog is cut to |x| < 980, so the extent of the points
# (1960 in x) is NOT the period; box_size must be given explicitly.
BOX_SIZE = 2000.

N_REGIONS_LIST = [4, 9, 16, 25, 49, 100]   # perfect squares: n_side x n_side transverse columns
N_RANDOM_SEEDS = 50
N_BATCHES_VALIDATION = 10                  # as in the tutorial


############################################
# MEASUREMENT PIECES
############################################

def periodic_tracers(points_3D, box_size, R_bins, pimax_values, los):
    '''Periodic copies of points_3D, trimmed exactly as in get_MIA_from3D(periodic_boundary=True).'''
    box_min = np.min(points_3D, axis=0)
    box_max = np.max(points_3D, axis=0)
    box_size = np.broadcast_to(np.asarray(box_size, dtype=float), (3,))
    new_orgins = np.array([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]) * box_size
    extended_points = np.concatenate([points_3D + new_orgin for new_orgin in new_orgins], axis=0)
    extend_by = np.full(3, np.max(R_bins))
    extend_by[los['los_axis']] = np.max(pimax_values)
    i_keep = np.all((extended_points > box_min - extend_by) & (extended_points < box_max + extend_by), axis=1)
    return np.ascontiguousarray(extended_points[i_keep])


def per_multiplet_sums(centers, orientations, tracers, R_bins, pimax_values, los, chunk=2000, print_progress=True):
    '''
    For each multiplet, sum of cos(2 Phi) and number of tracer pairs in each R bin.
    Same pair selection as calculate_rel_ang_cartesian_binAverage (strict R and pimax cuts, unit weights).
    returns: sums, counts, each shape (n_multiplets, len(R_bins)-1)
    '''
    n_bins = len(R_bins) - 1
    sums = np.zeros((len(centers), n_bins))
    counts = np.zeros((len(centers), n_bins))
    tree = cKDTree(tracers)
    r_query = np.sqrt(R_bins[-1]**2 + np.max(pimax_values)**2)
    pimax_lookup = np.append(pimax_values, np.nan)   # pairs outside the R bins get nan -> dropped
    t0 = time.time()
    for start in range(0, len(centers), chunk):
        stop = min(start + chunk, len(centers))
        ii = tree.query_ball_point(centers[start:stop], r=r_query)
        n_nb = np.fromiter((len(i) for i in ii), dtype=np.intp, count=len(ii))
        i_mult = np.repeat(np.arange(start, stop), n_nb)
        c = centers[i_mult]
        t = tracers[np.concatenate(ii).astype(int)]

        r_p, r_par = los_pair_separations(c, t, **los)
        r_bin_i = np.digitize(r_p, bins=R_bins) - 1
        keep = (r_p > R_bins[0]) & (r_p < R_bins[-1]) & (np.abs(r_par) < pimax_lookup[r_bin_i])
        i_mult, r_bin_i, c, t = i_mult[keep], r_bin_i[keep], c[keep], t[keep]

        cos2phi = np.cos(2 * (orientations[i_mult] - get_orientation_angle_cartesian(c, t, **los)))
        flat = (i_mult - start) * n_bins + r_bin_i
        size = (stop - start) * n_bins
        sums[start:stop] = np.bincount(flat, weights=cos2phi, minlength=size).reshape(-1, n_bins)
        counts[start:stop] = np.bincount(flat, minlength=size).reshape(-1, n_bins)
        if print_progress and (start // chunk) % 10 == 0:
            print('  multiplets %d / %d  (%.0f s)' % (stop, len(centers), time.time() - t0), flush=True)
    return sums, counts


############################################
# PARTITIONS
############################################

def code_shuffle_labels(n_multiplets, n_batches):
    '''Batch label of each multiplet exactly as in get_MIA_from3D (seed 42 shuffle, then np.array_split).'''
    random.seed(42)
    indices = np.asarray(range(n_multiplets))
    random.shuffle(indices)
    return _contiguous_batches(indices, n_batches)


def random_labels(n_multiplets, n_batches, rng):
    return _contiguous_batches(rng.permutation(n_multiplets), n_batches)


def _contiguous_batches(order, n_batches):
    labels = np.full(len(order), -1)
    for i, batch in enumerate(np.array_split(order, n_batches)):
        labels[batch] = i
    return labels


def region_labels(centers, n_regions, los):
    '''
    Equal-count regions in the plane transverse to the LOS, mirroring rel_angle_regions:
    sort by the 'north' coordinate (DEC analog) into n_side slices, then each slice by 'east' (RA analog).
    '''
    a = los['los_axis']
    east, north = centers[:, (a + 1) % 3], centers[:, (a + 2) % 3]
    n_side = int(np.sqrt(n_regions))
    labels = np.full(len(centers), -1)
    north_sorter = np.argsort(north)
    for i_slice, slice_idx in enumerate(np.array_split(north_sorter, n_side)):
        east_sorted = slice_idx[np.argsort(east[slice_idx])]
        for i_reg, reg_idx in enumerate(np.array_split(east_sorted, n_side)):
            labels[reg_idx] = i_slice * n_side + i_reg
    return labels


############################################
# ESTIMATORS
############################################

def group_sums(sums, counts, labels, n_groups):
    S = np.zeros((n_groups, sums.shape[1]))
    C = np.zeros((n_groups, sums.shape[1]))
    ok = labels >= 0
    np.add.at(S, labels[ok], sums[ok])
    np.add.at(C, labels[ok], counts[ok])
    return S, C


def subsample_cov(sums, counts, labels, n_groups):
    '''The get_MIA_from3D estimator: x_k = per-group mean, C = Cov(x_k, ddof=0) / N, so sqrt(diag) = relAng_e.'''
    S, C = group_sums(sums, counts, labels, n_groups)
    x = S / C
    d = x - np.mean(x, axis=0)
    return np.mean(x, axis=0), d.T @ d / n_groups / n_groups, x


def jackknife_cov(sums, counts, labels, n_groups):
    S, C = group_sums(sums, counts, labels, n_groups)
    x_minus_k = (S.sum(0) - S) / (C.sum(0) - C)
    d = x_minus_k - np.mean(x_minus_k, axis=0)
    return (n_groups - 1) / n_groups * d.T @ d


def corr(cov):
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


############################################
# VALIDATION OF get_MIA_from3D
############################################

def validate_get_MIA_from3D(points, multiplets, sums, counts):
    print('\n=== Validating get_MIA_from3D as-is (n_batches=%d, save_intermediate=True) ===' % N_BATCHES_VALIDATION)
    with tempfile.TemporaryDirectory() as tmp:   # fresh dir: get_MIA_from3D silently reuses existing batch files
        result = get_MIA_from3D(points, save_directory=tmp, R_bins=R_BINS, pimax='variable',
                                transverse_max=TRANSVERSE_MAX, los_max=LOS_MAX, print_info=False, sim_label='covcheck',
                                periodic_boundary=True, box_size=BOX_SIZE, n_batches=N_BATCHES_VALIDATION,
                                save_intermediate=True, return_pair_counts=True,
                                los_mode='axis', los_axis='z', multiplet_catalog=multiplets)
        batch_files = sorted(glob.glob(os.path.join(tmp, '*batches_[0-9]*.npy')))
        batch_files = [f for f in batch_files if not f.endswith('_paircounts.npy')]
        batch_files.sort(key=lambda f: int(f.split('batches_')[-1].split('.npy')[0]))
        saved = np.asarray([np.load(f) for f in batch_files])
        saved_counts = np.asarray([np.load(f.replace('.npy', '_paircounts.npy')) for f in batch_files])

    labels = code_shuffle_labels(len(multiplets), N_BATCHES_VALIDATION)
    mean, cov, x = subsample_cov(sums, counts, labels, N_BATCHES_VALIDATION)
    _, C = group_sums(sums, counts, labels, N_BATCHES_VALIDATION)

    checks = {
        'number of saved batch files == n_batches': len(saved) == N_BATCHES_VALIDATION,
        'saved per-batch pair counts match': np.array_equal(saved_counts, C.astype(int)),
        'saved per-batch cos(2Phi) match': np.allclose(saved, x, rtol=0, atol=1e-6),
        'relAng_plot matches mean of batches': np.allclose(result['relAng_plot'], mean, rtol=0, atol=1e-6),
        'relAng_plot_e matches sqrt(diag(cov))': np.allclose(result['relAng_plot_e'], np.sqrt(np.diag(cov)), rtol=1e-4),
        'pair_counts == total pairs': np.array_equal(np.asarray(result['pair_counts']), C.sum(0).astype(int)),
        'no multiplets dropped (pair counts = all multiplets)': np.array_equal(np.asarray(result['pair_counts']), counts.sum(0).astype(int)),
    }
    print('max |saved batch - regrouped| = %.2e' % np.max(np.abs(saved - x)) if len(saved) == len(x) else 'shape mismatch')
    for name, ok in checks.items():
        print('  [%s] %s' % ('PASS' if ok else 'FAIL', name))
    return all(checks.values())


def check_saved_batch_reuse(points):
    '''
    Saved batch files must only be reused for identical settings, and only when save_intermediate=True.
    Runs on a small sub-cube so it is quick.
    '''
    print('\n=== Checking reuse of saved batch files (small sub-cube) ===')
    sub = points[np.all(np.abs(points) < 250, axis=1)]
    kw = dict(R_bins=R_BINS, transverse_max=TRANSVERSE_MAX, los_max=LOS_MAX, print_info=False, sim_label='reuse',
              n_batches=5, los_mode='axis', los_axis='z')
    with tempfile.TemporaryDirectory() as tmp:
        first = get_MIA_from3D(sub, tmp, pimax='variable', save_intermediate=True, **kw)
        resumed = get_MIA_from3D(sub, tmp, pimax='variable', save_intermediate=True, **kw)
        other_pimax = get_MIA_from3D(sub, tmp, pimax=20, save_intermediate=True, **kw)
        fresh = get_MIA_from3D(sub, tmp, pimax=20, save_intermediate=False, **kw)
        # corrupt the saved pimax=20 batches: a run without save_intermediate must not read them
        for f in glob.glob(os.path.join(tmp, '*' + other_pimax.meta['CFGHASH'] + '*batches_[0-9].npy')):
            np.save(f, np.full(len(R_BINS) - 1, 99.))
        not_reused = get_MIA_from3D(sub, tmp, pimax=20, save_intermediate=False, **kw)
        n_batch_files = len([f for f in glob.glob(os.path.join(tmp, '*batches_[0-9].npy'))])

    checks = {
        'resumed run (same settings) reproduces the result': np.allclose(first['relAng_plot'], resumed['relAng_plot']),
        'different pimax -> different config hash': first.meta['CFGHASH'] != other_pimax.meta['CFGHASH'],
        'different pimax -> new batch files (2 x 5 saved)': n_batch_files == 10,
        'different pimax -> different result': not np.allclose(first['relAng_plot'], other_pimax['relAng_plot']),
        'save_intermediate=False matches the saved run': np.allclose(fresh['relAng_plot'], other_pimax['relAng_plot']),
        'save_intermediate=False ignores existing files': np.allclose(not_reused['relAng_plot'], other_pimax['relAng_plot']),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with tempfile.TemporaryDirectory() as tmp:
            get_MIA_from3D(sub[np.abs(sub[:, 0]) < 200], tmp, pimax='variable', periodic_boundary=True, **kw)
    checks['warns when inferred box_size differs between axes'] = any('box_size' in str(w.message) for w in caught)

    for name, ok in checks.items():
        print('  [%s] %s' % ('PASS' if ok else 'FAIL', name))
    return all(checks.values())


############################################
# MAIN
############################################

def make_figure(results, out_path, n_show):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ordinal one-hue ramp (light -> dark = fewer -> more regions)
    ramp = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, key, title in [(axes[0, 0], 'region_sub', 'Regions, subsample estimator (as get_MIA_from3D)'),
                           (axes[0, 1], 'region_jk', 'Regions, delete-one jackknife')]:
        for color, N in zip(ramp, N_REGIONS_LIST):
            r = results[N]
            ax.plot(R_MID, np.sqrt(np.diag(r[key]) / np.diag(r['random_mean'])), color=color, lw=2,
                    marker='o', ms=4, label='N = %d' % N)
        ax.axhline(1, color='#888888', lw=0.8, zorder=0)
        ax.set_xscale('log')
        ax.set_xlabel(r'$R$ [$h^{-1}$Mpc]')
        ax.set_ylabel(r'$\sigma_{\rm region} / \sigma_{\rm random}$')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8, title='regions / batches', title_fontsize=8)

    r = results[n_show]
    for ax, cov, title in [(axes[1, 0], r['random_mean'], 'Random batches, N = %d (mean over seeds)' % n_show),
                           (axes[1, 1], r['region_jk'], 'Region jackknife, N = %d' % n_show)]:
        im = ax.imshow(corr(cov), origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
        ticks = np.arange(0, len(R_MID), 3)
        ax.set_xticks(ticks); ax.set_xticklabels(['%.0f' % R_MID[t] for t in ticks])
        ax.set_yticks(ticks); ax.set_yticklabels(['%.0f' % R_MID[t] for t in ticks])
        ax.set_xlabel(r'$R$ [$h^{-1}$Mpc]'); ax.set_ylabel(r'$R$ [$h^{-1}$Mpc]')
        ax.set_title('Correlation: ' + title, fontsize=10)
    fig.colorbar(im, ax=axes[1, :], shrink=0.8, label='correlation coefficient')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print('Figure saved to', out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_covariance_split'))
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    points = np.asarray(Table.read(CATALOG_PATH)['x_L2com'], dtype=float)
    print('Loaded %d points' % len(points))
    multiplets = make_group_catalog(None, comoving_points=points, transverse_max=TRANSVERSE_MAX, los_max=LOS_MAX,
                                    max_n=100, use_sky_coords=False, **LOS)
    centers = np.asarray(multiplets['center_loc'], dtype=float)
    orientations = np.asarray(multiplets['orientation'], dtype=float)
    print('Found %d multiplets' % len(multiplets))

    cache = os.path.join(args.outdir, 'per_multiplet_sums_%dmultiplets.npz' % len(multiplets))
    if os.path.exists(cache):
        print('Loading per-multiplet sums from', cache)
        d = np.load(cache)
        sums, counts = d['sums'], d['counts']
    else:
        print('Computing per-multiplet sums')
        tracers = periodic_tracers(points, BOX_SIZE, R_BINS, PIMAX_VALUES, LOS)
        sums, counts = per_multiplet_sums(centers, orientations, tracers, R_BINS, PIMAX_VALUES, LOS)
        np.savez(cache, sums=sums, counts=counts)

    if not args.skip_validation:
        if not validate_get_MIA_from3D(points, multiplets, sums, counts):
            print('\nValidation FAILED: the comparison below does not describe get_MIA_from3D.')
        if not check_saved_batch_reuse(points):
            print('\nSaved-batch reuse check FAILED.')

    print('\nFull-sample signal (all pairs):')
    print('  R     :', ' '.join('%8.1f' % v for v in R_MID))
    print('  <cos> :', ' '.join('%8.5f' % v for v in sums.sum(0) / counts.sum(0)))

    rng = np.random.default_rng(1)
    results = {}
    for N in N_REGIONS_LIST:
        random_covs = np.asarray([subsample_cov(sums, counts, random_labels(len(sums), N, rng), N)[1]
                                  for _ in range(N_RANDOM_SEEDS)])
        reg = region_labels(centers, N, LOS)
        results[N] = {
            'random_mean': random_covs.mean(0),
            'random_sigma_scatter': np.std(np.sqrt(np.diagonal(random_covs, axis1=1, axis2=2)), axis=0),
            'region_sub': subsample_cov(sums, counts, reg, N)[1],
            'region_jk': jackknife_cov(sums, counts, reg, N),
            'random_jk_seed0': jackknife_cov(sums, counts, random_labels(len(sums), N, rng), N),
        }
        r = results[N]
        sig_rand = np.sqrt(np.diag(r['random_mean']))
        print('\n=== N = %d (region width ~ %.0f Mpc/h) ===' % (N, BOX_SIZE / np.sqrt(N)))
        print('  R                    :', ' '.join('%6.1f' % v for v in R_MID))
        print('  sigma_random x1e3    :', ' '.join('%6.3f' % v for v in 1e3 * sig_rand))
        print('  seed scatter / sigma :', ' '.join('%6.2f' % v for v in r['random_sigma_scatter'] / sig_rand))
        print('  region_sub / random  :', ' '.join('%6.2f' % v for v in np.sqrt(np.diag(r['region_sub'])) / sig_rand))
        print('  region_jk  / random  :', ' '.join('%6.2f' % v for v in np.sqrt(np.diag(r['region_jk'])) / sig_rand))
        print('  random_jk  / random  :', ' '.join('%6.2f' % v for v in np.sqrt(np.diag(r['random_jk_seed0'])) / sig_rand))
        off = ~np.eye(len(R_MID), dtype=bool) & (np.abs(np.subtract.outer(np.arange(len(R_MID)), np.arange(len(R_MID)))) == 1)
        print('  mean adjacent-bin corr: random %.2f, region_sub %.2f, region_jk %.2f' % (
            corr(r['random_mean'])[off].mean(), corr(r['region_sub'])[off].mean(), corr(r['region_jk'])[off].mean()))

    np.savez(os.path.join(args.outdir, 'covariances.npz'), R_bins=R_BINS, pimax=PIMAX_VALUES,
             n_regions=np.asarray(N_REGIONS_LIST),
             **{'%s_N%d' % (k, N): v for N, r in results.items() for k, v in r.items()})
    make_figure(results, os.path.join(args.outdir, 'covariance_region_vs_random.png'), n_show=25)


if __name__ == '__main__':
    main()

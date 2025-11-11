import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
# make all fonts larger
plt.rcParams.update({'font.size': 14})

def plot_MIA_mu_bins(result_paths, estimator, title=None, save_path=None, cmap='seismic', ylim=None, vlim=None, mu_sym=False):
    """
    Plots the MIA results measured in bins of separation s and angle mu wrt the line of sight.
    
    Parameters:
    - result_paths: list of str
        List of file paths to the 2D FITS files containing MIA results.
    - estimator: str
        The estimator used (e.g., 'x+', 'g+', '++').
    - save_path: str or None
        If provided, saves the plot to this path. Otherwise, displays the plot.
    """
    # Load data from FITS files
    print(f'Found {len(result_paths)} result files.')
    # read first file to get shape and bin edges
    with fits.open(result_paths[0]) as h:
        sample = h[0].data.astype(np.float32)
        s_bins = h['S_BINS'].data
        mu_bins = h['MU_BINS'].data

    # stack arrays (use float32 to save memory)
    stack = np.empty((len(result_paths),) + sample.shape, dtype=np.float32)
    for i,p in enumerate(result_paths):
        try:
            with fits.open(p) as h:
                stack[i] = h[0].data.astype(np.float32)
        except TypeError:
            print(f'Error reading file {p}, skipping.')
            stack[i] = np.nan

    # compute per-pixel statistics ignoring NaNs
    mean2d = np.nanmean(stack, axis=0)
    std2d = np.nanstd(stack, axis=0)
    counts = np.sum(~np.isnan(stack), axis=0)
    stderr2d = np.zeros_like(std2d)
    nonzero = counts>0
    stderr2d[nonzero] = std2d[nonzero] / np.sqrt(counts[nonzero])
    stderr2d[~nonzero] = np.nan
    
    
    # averaging in mu to get s dependence
    mean1d_s = np.nanmean(mean2d, axis=1)
    stderr1d_s = np.nanmean(stderr2d, axis=1)
    s_bin_middles = (s_bins[1:]+s_bins[:-1])/2
    
    
    fig = plt.figure(figsize=(15,6))
    # make title for entire figure
    fig.suptitle(title, fontsize=16)
    
    # averaged over angular bins
    plt.subplot(1,2,1)
    plt.errorbar(s_bin_middles, mean1d_s*s_bin_middles, yerr=stderr1d_s*s_bin_middles, fmt='o', capsize=4)
    plt.xscale('log')
    plt.xlabel(r'3D separation $s$ [$h^{-1}$ Mpc]')
    plt.ylabel(r'$s$ MIA$_{'+estimator+r'}$ [$h^{-1}$ Mpc]')
    # draw thin grey line at y=0
    plt.axhline(0, color='grey', linestyle='--');
    if ylim is not None:
        plt.ylim(ylim)
    
    
    # RADIAL PLOT

    # monkeypatch plt.subplot so the right (1,2,2) subplot is created slightly larger
    _orig_subplot = plt.subplot
    def _subplot_wrapper(*args, **kwargs):
        # detect call to plt.subplot(1,2,2, ...)
        if len(args) >= 3 and args[0] == 1 and args[1] == 2 and args[2] == 2:
            # rect = [left, bottom, width, height] (tweak these to adjust size)
            rect = [0.52, 0.064, 0.40, 0.80]
            return plt.axes(rect, **kwargs)
        return _orig_subplot(*args, **kwargs)
    plt.subplot = _subplot_wrapper
    data = mean2d.T
    if mu_sym:
        # average over mu and -mu
        data_flipped = mean2d[:, ::-1].T
        data = 0.5 * (data + data_flipped)
    theta_edges = np.arccos(mu_bins)   # edges in radians
    r_edges = s_bins
    
    # mask NaNs
    data_ma = np.ma.masked_invalid(data)
    
    ax = plt.subplot(1,2,2, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # mesh for original
    Theta, R = np.meshgrid(theta_edges, r_edges, indexing='ij')

    vmax = np.nanmax(np.abs(mean2d))*.8
    if vlim is not None:
        vmax = vlim

    pcm1 = ax.pcolormesh(Theta, R, data_ma, cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')

    # create mirrored angular coordinates and mirrored data (reflection about vertical axis)
    Theta_mirror = np.pi - np.mod(-Theta, 2*np.pi)         # theta -> -theta (wrapped to [0,2pi))
    data_mirror = data_ma[::-1, :]                 # flip angular rows to match reflection

    # plot mirrored overlay with some transparency
    pcm2 = ax.pcolormesh(Theta_mirror, R, data_mirror, cmap=cmap, vmin=-vmax, vmax=vmax,
                        shading='auto')

    # add radial/log scale and title
    ax.set_rscale('log')

    # add a radial line in the middle of the 3rd quadrant (angle = 5*pi/4)
    angle_mid = 5 * np.pi / 4
    r_min = r_edges[0]
    r_max = r_edges[-1]
    ax.plot([angle_mid, angle_mid], [r_min, r_max], color='k', lw=1.5, zorder=5)

    # mark log(s) = 1, 1.5, 2 (i.e. s = 10, ~31.62, 100) along that line and label them
    log_ticks = [10, 30, 100]
    for lt in log_ticks:
        ax.text(angle_mid+(1/lt)**.6, lt*1.1, f'{lt:g}', color='k', fontsize=14, ha='center', va='bottom', zorder=7)

    # add the radial label near the outer end of the line
    # draw short perpendicular ticks at each marked radius (overplot the dots)
    dtheta = 0.03  # angular half-width of each tick in radians
    for rv in log_ticks:
        ax.plot([angle_mid - dtheta, angle_mid + dtheta], [rv, rv], color='k', lw=1.5, zorder=6)

    # rotated radial label aligned with the radial line
    ax.text(angle_mid*.98, r_max*.25, r"$s$ [$h^{-1}$Mpc]", color='k', fontsize=14,
            ha='center', va='top', zorder=7,
            rotation=180+np.degrees(angle_mid), rotation_mode='anchor')

    cbar = fig.colorbar(pcm1, ax=ax, pad=0.12, shrink=0.9, aspect=20)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r'MIA$_{'+estimator+r'}$')

    # label a few theta ticks by mu for clarity
    mu_ticks = [1, 0, -1]
    mu_tick_labels = [r'$\mu=1$', '', r'$\mu=-1$']
    # place mu=0 label manually, shifted slightly to the right so it doesn't overlap
    angle_mu0 = np.arccos(0.0)
    ax.text(angle_mu0, r_max * 1.1, r'$\mu=0$', ha='left', va='center', fontsize=14, zorder=10)
    theta_tick_locs = np.arccos(mu_ticks)
    ax.set_xticks(theta_tick_locs)
    ax.set_xticklabels(mu_tick_labels)
    # remove radial labels
    ax.set_yticks([])
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'Saved plot to {save_path}')
    else:
        plt.show()
import numpy as np
from astropy.table import Table
from alignment_functions.basic_alignment import *
from visualization_functions.plotting import *

def gauss_sum(x, *params):        # for multiple gaussians
    '''
    params: list of len 2*n_gaussians. [amp1, std1, amp2, std2, ...]
    '''
    y = np.zeros_like(x)
    for i in range(0, len(params), 2):
        amp = params[i]
        std = params[i+1]
        y += amp * np.exp( -( x**2/( 2 * std**2) ))
    return y

def fit_gaussians(rpar_bins, measured_weight, n_gaussians=1, amp_max=np.inf):
    p0 = [1, 1]*n_gaussians # initial guess for parameters. amplitude, width
    try:
        popt, pcov = curve_fit(gauss_sum, rpar_bins, measured_weight, p0=p0, bounds=([-np.inf, 0]*n_gaussians, [amp_max, np.inf]*n_gaussians))
    except RuntimeError:
        popt = [np.nan]*len(p0)
    return popt


def bin_relE_results(all_proj_dists, all_pa_rels, all_los_dists=None, all_weights=None, R_bins=np.logspace(0, 2, 11), pimax = 30, pimax_weights=None):
    '''
    Bin the measurements of relative ellipticity
    --------------------------------------------
    R_bins: bin edges for the projected separation, in Mpc/h
    all_proj_dists: projected separation of pairs of galaxies in Mpc/h
    all_pa_rels: relative position angle of pairs of galaxies in rad (or just the quantity to be binned)
    all_los_dists (optional): line-of-sight separation of pairs of galaxies in Mpc/h. If None, all will be used in binning.
    pimax : float or array
        Maximum line-of-sight separation for pairs of galaxies in Mpc/h. Default is 30.
        Can also be array of size (R_bins-1) to use a different pimax for each R bin
    pimax_weights (optional): if provided, pimax will be ignored. pimax_weights must be an array of gaussian parameters for the los-based weighting in each r_p bin. 
        Of shape (len(R_bins)-1, 2*ng) where ng is the number of gaussians. 
        The weight will be a gaussian mixture using the parameters provided as amp1, wid1, amp2, wid2, etc.
    '''
    
    all_binned_pa_rels = []
    # loop over every r_p bin
    for i in range(len(R_bins)-1):
        rp_min = R_bins[i]
        rp_max = R_bins[i+1]
        
        i_keep = (all_proj_dists > rp_min) & (all_proj_dists < rp_max)   # isolate pairs in r_p bin
        
        if all_los_dists is not None:
            # put some (maybe initial) cut on los
            pimax_i = pimax
            if pimax_weights is not None:
                # could add function here to set some pimax based on where weights functionally go to zero
                pimax_i = 130 # temporary
            elif type(pimax) == np.ndarray:
                pimax_i = pimax[i]
            i_keep &= all_los_dists < pimax_i
        binned_pa_rel = all_pa_rels[i_keep]
        
        total_weights = [1] * len(binned_pa_rel)
        # normal weights
        if all_weights is not None:
            total_weights *= all_weights[i_keep]
        # los weights
        if pimax_weights is not None and all_los_dists is not None:
            total_weights *= gauss_sum(all_los_dists[i_keep], *pimax_weights[i])
        #elif pimax_weights is not None:
        #    total_weights = 0
        
        msum = np.sum(binned_pa_rel*np.asarray(total_weights))
        wsum = np.sum(np.asarray(total_weights))
        binned_pa_rel = msum / wsum
        
        all_binned_pa_rels.append(binned_pa_rel)
        
    return -np.asarray(all_binned_pa_rels)
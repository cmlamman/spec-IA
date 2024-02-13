# needs to be on cf_env
from pycorr import TwoPointCorrelationFunction
import random
import numpy as np
from scipy.interpolate import interp2d
from scipy import stats
import astropy.units as u


from astropy.cosmology import LambdaCDM, z_at_value
cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)


############################################
# For caltulcating correlation functions
############################################

def format_pos_for_cf(catalog, z_column='Z'):
    z_comoving = cosmo.comoving_distance(catalog[z_column]).to(u.Mpc).value * 0.7
    return np.asarray([catalog['RA'], catalog['DEC'], z_comoving])    # z_comoving in Mpc/h

def generate_randoms_zshuffle(catalog):
    z_comoving = cosmo.comoving_distance(catalog['Z']).to(u.Mpc).value * 0.7
    random.shuffle(z_comoving)
    return np.asarray([catalog['RA'], catalog['DEC'], z_comoving])    # z_comoving in Mpc/h



def get_mu(r_p, r_par):
    return np.cos(np.arctan2(r_p, r_par))


def bin_pair_results(pair_results, nrp_bins=100, npar_bins=150, rp_max=20, par_max=30):
    '''
    Input: pair_results, a table of computed galaxy pairs and their relative alignment. Containing columns r_p, r_par, s_par, and e1_rel
    For each of the 2D bins in r_p and r_par, this function calculates the IA signal.
    Returns the IA signal and correlation function used. Both are 2D arrays of size (nrp_bins, npar_bins)
    '''
    rp_bins = np.linspace(0, rp_max, nrp_bins+1)
    rpar_bins =  np.linspace(0, par_max, npar_bins+1)

    binned_pairs = stats.binned_statistic_2d(pair_sample['r_p'], pair_sample['r_par'], pair_sample['e1_rel'], 'mean', bins=[rp_bins, rpar_bins])
    
    return binned_pairs[0]


def get_cf_binned(catalog, random_catalog="auto", nrp_bins=100, npar_bins=150, rp_max=20, par_max=30):
    '''
    Input: catalog, a table of galaxy positions. random_catalog, a table of random galaxy positions (optional).
    For each of the 2D bins in r_p and r_par, this function calculates the correlation function.
    Returns the correlation function as a 2D array of size (nrp_bins, npar_bins)
    '''
    rp_bins = np.linspace(0, rp_max, nrp_bins+1)
    rpar_bins =  np.linspace(-par_max, par_max, npar_bins*2+1)
    pos = format_pos_for_cf(catalog, z_column='Z')
    if random_catalog=="auto":
        pos_r = generate_randoms_zshuffle(catalog)
        
    corr_result = TwoPointCorrelationFunction('rppi', edges=(rp_bins, rpar_bins), position_type='rdd', data_positions1=pos, randoms_positions1=pos_r,
                                            engine='corrfunc', nthreads=4)
    corr = corr_result.corr
    corr_stacked = (np.fliplr(corr[:, :len(corr[0])//2]) +  corr[:, len(corr[0])//2:]) / 2  # stacking the +/- sides of the rpar bins
    return corr_stacked


def get_rsd_ia_weights(cf_binned, ia_binned):
    '''
    Input: cf_binned, a 2D array of the correlation function binned in r_p and r_par. size (nrp_bins, npar_bins)
    ia_binned, a 2D array of the IA signal binned in r_p and r_par
    This requires no knowledge of the actual r_p and r_par values, only the binned results.
    Returns: a 2D array of weights, same size as input arrays
    '''
    
    # find the total signal in each bin of r_p (i.e. "rp slice") and the the weight in each r_par bin for each slice
    weight_array = np.zeros_like(cf_binned)
    
    slice_sum = np.sum(ia_binned**2 / (1 + cf_binned), axis=1)  # for each bin in r_p
    slice_weights = (ia_binned / (1 + cf_binned)) / slice_sum[:, np.newaxis]  # for each r_par bin in each r_p slice
    
    # add the weights to the weight array
    weight_array = slice_weights
    
    return weight_array

def get_projected_weighted_signal(signal_2d, weights_2d):
    '''
    Input: signal_2d, a 2D array of the signal binned in r_p and r_par. size (nrp_bins, npar_bins)
    weights_2d, a 2D array of the weights binned in r_p and r_par. size (nrp_bins, npar_bins)
    Returns: a 1D array of the weighted signal, binned in r_p
    '''
    weighted_average = np.nansum(signal_2d * weights_2d, axis=1) / np.nansum(weights_2d, axis=1)
    return weighted_average
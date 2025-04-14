import numpy as np
import pandas as pd
import glob

import warnings

import scipy
from scipy import signal
from scipy import special, integrate, misc, interpolate

import astropy.cosmology.units as cu
import astropy.units as u
from astropy.constants import M_sun
u.add_enabled_units(cu);

from astropy.cosmology import LambdaCDM
H0 = 69.6
cosmo = LambdaCDM(H0=H0, Om0=0.286, Ode0=0.714)
h = H0/100


#import sys
#sys.path.append('/global/homes/c/clamman/IA/')

import pathlib
parent_path = pathlib.Path.cwd().parent
if 'IA' not in str(parent_path):   # for working in other directories. May need to rename "IA" to wherever spec-IA is located.
    parent_path = parent_path / 'IA'
    print('Parent path is not the IA directory. Assuming parent directory is IA')

# reading in a file for the power spectrum
ps_path = parent_path / 'spec-IA/example_data/AbacusSummit_base_c000_z0.800_power_nfft2048.csv' #ps_path = parent_path / 'spec-IA/example_data/AbacusSummit_base_c000_z0.800_power_nfft2048.csv'
abacus_ps_nl = pd.read_csv(ps_path) # nonlinear matter power spectrum from AbacusSummit

kz_spline_parent_paths = parent_path / 'spec-IA/example_data/kz_integral_splines/'
kz_spline_paths = list(kz_spline_parent_paths.glob('*.npy')) # pre-computed values of the kz integral at given values of pimax, made with the above power spectum

#####################################################################################################
# GENRAL COSMOLOGY FUNCTIONS
#####################################################################################################

def g(z):
    O_M = cosmo.Om(z)
    O_DE = 1 - O_M
    return (5/2) * O_M / (O_M**(4/7) - O_DE + (1 + 0.5*O_M) * (1 + (O_DE/70)))

def D(z, norm_at_z0=False):
    '''
    Return growth factor at z. 
    If norm_at_z0 is False, this will be normalized so \bar{D}(z) = (1+z)D(z) is 1 at matter-dominated era (z=infinity).
    If norma_at_z1 is True, this will be normalized so D(z) is 1 at present epoch (z=0).
    '''
    if norm_at_z0:
        return g(z) / ((1+z)*g(0))
    else: 
        z_inf = 100000
        return (g(z) / g(z_inf)) / (1+z)
    
def get_relative_bias(z, wp, D_base, wp_base):
    '''get relative bias of a galaxy sample at z with projected correlation function wp, 
    compared to a sample with growth factor D_base and projected correlation function wp_base.'''
    return (D_base / D(z)) * (wp/wp_base)**2
       

#####################################################################################################
# FUNCTION TO CONVERT BETWEEN TAU AND A_IA
#####################################################################################################
def tau_to_AIA(tau, z, norm_at_z0=False):
    '''
    convert from our tau parameter to the commonly used IA amplitude parameter A_IA.
    If norm_at_z0 is False, this will be normalized so \bar{D}(z) = (1+z)D(z) is 1 at matter-dominated era (z=infinity).
    If norma_at_z1 is True, this will be normalized so D(z) is 1 at present epoch (z=0).
    '''
    C1 = 5*10**(-14) * ((u.Mpc)**3 / h**2) / M_sun
    rho_m0 = cosmo.Om(z) * cosmo.critical_density0
    AIA = - (tau / C1) * (D(z, norm_at_z0=norm_at_z0) / rho_m0)
    return AIA.to(u.Mpc/u.Mpc).value


#####################################################################################################
# FUNCTION TO CALCULATE MODEL PREDICTION FOR GIVEN MEASUREMENT OF RELATIVE PROJECTED ELLIPTICITY
#####################################################################################################


def precompute_kz_integral(pimax_values, PS_data, directory, PS_min = 1e-4, PS_max = 2e2, n_samples=100, warning_handling='once', pi_weighting=False, gauss_params=None, pimax_rp=None, overwrite=True):
    '''
    Precompute the integral of the kz integral and save a spline as a function of K (in log space)
    This is used to speed up the calculation of rel_e_to_tau.
    
    INPUT:
    ------------------
    pimax_values: array of shape (n,). Line-of-sight distance 
    PS_data: dictonary or DataFrame of matter power spectrum values. Must contain columns 'k' and 'P'. Default is a non-linear matter power spectrum from AbacusSummit.
    directory: string. Directory path to save the spline files.
    PS_min, PS_max: floats. Range of k to use for the power spectrum [h/Mpc]
    n_samples: int. Number of samples to use for the spline. For a better integration, this function will add a few more to better sample where sincx = 0
    gauss_params: array of shape (2n,). Parameters for the gaussian fit. If pi_weighting = True, this is the parameters for the gaussian fit to the 2D correlation function.
                  will use parameters as sigma1, width1, sigma2, width2, etc.
    RETURNS:
    ------------------
    None. Saves the spline files in the directory.
    '''
    
    if overwrite == False:
        if pi_weighting == False:
            if all([os.path.exists(directory+'/kz_integral_NL_spl_pimax_'+str(pmi)+'.npy') for pmi in pimax_values]):
                print('All files exist. Skipping')
                return None
        else:
            if os.path.exists(directory+'/kz_integral_NL_spl_Wpimaxrp_'+str(pimax_rp)+'.pkl'):
                print('File exists. Skipping')
                return None
    
    warnings.filterwarnings(warning_handling)
    
    # using interpolation to get P(k)
    tck_PS = interpolate.splrep(np.log10(PS_data['k']), np.log10(PS_data['P']), s=4e-3)
    
    def get_PS(k):
        return 10**(interpolate.splev(np.log10(k), tck_PS))

    if pi_weighting == False:
        def kz_integrand(kz, K, pimax):
            '''does not include front constant of b_gal * pimax / pi'''
            k_squared = kz**2 + K**2
            sinc_value = scipy.special.sinc( (kz * pimax) / np.pi )  # THIS SCIPY SINC INCLUDES PI NORMALIZATION > MUST DIVIDE BY PI
            return get_PS(np.sqrt(k_squared)) * (K**2 / k_squared) * sinc_value
    else:
        n_gaussians = int(len(gauss_params) / 2)
        if n_gaussians == 1:
            def kz_integrand(kz, K):
                k_squared = kz**2 + K**2
                gaussian_sum_fs = get_gauss_sum_fs_1D(kz, *gauss_params)
                return get_PS(np.sqrt(k_squared)) * (K**2 / k_squared) * gaussian_sum_fs
        elif n_gaussians == 3:
            def kz_integrand(kz, K):
                k_squared = kz**2 + K**2
                gaussian_sum_fs = get_gauss_sum_fs_3D(kz, *gauss_params) 
                return get_PS(np.sqrt(k_squared)) * (K**2 / k_squared) * gaussian_sum_fs
        else:
            raise ValueError('Only 1 or 3 gaussians supported')

    Ks_sample_values = np.logspace(np.log10(PS_min), np.log10(PS_max), n_samples)    
    
    if pi_weighting == False:# or pi_weighting == True:
        # For better integration: adding more samples around where sincx = 0
        n_pi_steps = 1000 # must be integer
        max_pi = 1e3  # must be integer
        min_pi = 1e1  # must be integer
        Ks_extra_values = (np.arange(0, n_pi_steps) * np.pi * max_pi/(3*n_pi_steps)) + float(int(min_pi/np.pi))*np.pi
        Ks_sample_values = np.sort(np.concatenate((Ks_sample_values, Ks_extra_values)))
    
    if pi_weighting == False:
        for pmi in pimax_values:
            kz_integral_values = []
            for K in Ks_sample_values:
                kz_integral_values.append(scipy.integrate.romberg(kz_integrand, a=PS_min, b=PS_max, args=[K, pmi], rtol=1.48e-8, divmax=15)) # integrate over kz
            tck_kz_integral = interpolate.splrep(np.log10(Ks_sample_values), kz_integral_values, s=1e-3)
            
            # save the spline
            print('saving for pimax =', pmi)
            np.save(directory+'/kz_integral_NL_spl_pimax_'+str(pmi)+'.npy', tck_kz_integral)
    
    else:
        kz_integral_values = []
        for K in Ks_sample_values:
            value = scipy.integrate.romberg(kz_integrand, a=PS_min, b=PS_max, args=[K], rtol=1.48e-8, divmax=15)
            kz_integral_values.append(value) # integrate over kz
        tck_kz_integral = interpolate.splrep(np.log10(Ks_sample_values), kz_integral_values, s=1e-3)
        print('saving pimax-weighted values for rp =', pimax_rp)
        with open(directory+'/kz_integral_NL_spl_Wpimaxrp_'+str(n_gaussians)+'D_'+str(pimax_rp)+'.pkl', 'wb') as f:
            pickle.dump(tck_kz_integral, f)
    print('Finished')
    return None

def precompute_kz_integral_1D_gauss_limber(pimax_values, PS_data, directory, gauss_std, PS_min = 1e-4, PS_max = 2e2, n_samples=100, warning_handling='once', pimax_rp=None, overwrite=True):
    # assume that only the kz=0 contributions matter. Typically only true when r_p << r_par

    '''
    Precompute the integral of the kz integral and save a spline as a function of K (in log space)
    
    INPUT:
    ------------------
    pimax_values: array of shape (n,). Line-of-sight distance 
    PS_data: dictonary or DataFrame of matter power spectrum values. Must contain columns 'k' and 'P'. Default is a non-linear matter power spectrum from AbacusSummit.
    directory: string. Directory path to save the spline files.
    PS_min, PS_max: floats. Range of k to use for the power spectrum [h/Mpc]
    n_samples: int. Number of samples to use for the spline. For a better integration, this function will add a few more to better sample where sincx = 0
    gauss_params: array of shape (2n,). Parameters for the gaussian fit. If pi_weighting = True, this is the parameters for the gaussian fit to the 2D correlation function.
                  will use parameters as sigma1, width1, sigma2, width2, etc.
    RETURNS:
    ------------------
    None. Saves the spline files in the directory.
    '''
    
    if overwrite == False:
        if os.path.exists(directory+'/kz_integral_NL_spl_Wpimaxrp_1D_limber_'+str(pimax_rp)+'.pkl'):
            print('File exists. Skipping')
            return None
        
    Ks_sample_values = np.logspace(np.log10(PS_min), np.log10(PS_max), n_samples)    

    kz_integral_values = gauss_std * get_PS(Ks_sample_values)
    tck_kz_integral = interpolate.splrep(np.log10(Ks_sample_values), kz_integral_values, s=1e-3)
    print('saving values for rp =', pimax_rp)
    with open(directory+'/kz_integral_NL_spl_Wpimaxrp_1D_limber_'+str(pimax_rp)+'.pkl', 'wb') as f:
        pickle.dump(tck_kz_integral, f)
    print('Finished')
    return None
    
# CALCULATING FANCY J
def get_fancyJ(Rmin, Rmax, bigK):
    return (2*scipy.special.j0(Rmin*bigK) + Rmin*bigK*scipy.special.j1(Rmin*bigK) - 2*scipy.special.j0(Rmax*bigK) - Rmax*bigK*scipy.special.j1(Rmax*bigK)) * 2 / (bigK**2 * (Rmax**2 - Rmin**2))


def compute_rel_e_model(rel_e_measurement, wp_measurement, pimax_values, b_gal, z = 0.8, rel_e_randoms=None,
                 PS_min = 1e-4, PS_max = 2e2, PS_data = abacus_ps_nl, PS_z = 0.8, precomputed_kz_integral_paths = [str(kzs) for kzs in kz_spline_paths], warning_handling='once'):
    '''
    Compute a model prediction at each bin of projected separation. This code assumes a tau value of 1.
    
    INPUT - REQUIRED:
    ------------------
    rel_e_measurement: dictonary or DataFrame of projected relative ellipticity measurement between a shape catalog and a tracer catalog.
        must contain columns 'R_bin_min', 'R_bin_max', ''.
    wp: dictonary or DataFrame of projected cross-correlation function between a shape and a tracer catalog.
        must contain columns 'R_bin_min', 'R_bin_max', 'wp'.
    Rbins: array of shape (n+1,). Bin edges that rel_e_measurment and wp were made in [Mpc/h]
    pimax_values: float or array of shape (n,). Line-of-sight distance that rel_e_measurement and wp were made [Mpc/h]
    b_gal: float. bias of the galaxy catalog used for tracers
    z: float. redshift of galaxies used in measurement
    
    INPUT - OPTIONAL:
    ------------------
    rel_e_randoms: dictonary or DataFrame of projected relative ellipticity measurement between a shape catalog and a random catalog.
        must contain columns 'R_bin_min', 'R_bin_max', 'relAng_plot'.
    PS_min, PS_max: floats. Range of k to use for the power spectrum [h/Mpc]
    PS_data: dictonary or DataFrame of matter power spectrum values. Must contain columns 'k' and 'P'. Default is a non-linear matter power spectrum from AbacusSummit.
    PS_z = float. redshift of the power spectrum. Default is 0.8.
    precomputed_kz_integral_paths: list of paths to precomputed values of the kz integral afor given pi values, made with the above power spectum. 
        Must be formatted as the output of precompute_kz_integrand.
    '''
    warnings.filterwarnings(warning_handling)
    
    if rel_e_measurement['R_bin_min'][0] != wp_measurement['R_bin_min'][0]:
        print('Warning - R bins do not match between rel_e_measurement and wp_measurement!! Continuing with the R bins from rel_e_measurement')
        
    # if the redshift of the measurement and power spectrum are different, adjust using growth factor
    D_norm = 1
    if z != PS_z:
        D_norm = (D(z) / D(PS_z))**2

        
    # read in the pre-computed value of the kz integral (made with precompute_kz_integral)
    splines = [np.load(path, allow_pickle=True) for path in precomputed_kz_integral_paths]
    pimax_values = [round(float(path.split('_')[-1].split('.npy')[0]), 2) for path in precomputed_kz_integral_paths]

    def get_kz_integral_spl(K, pimax, b_gal):
        try:
            i_to_use = list(pimax_values).index(round(pimax, 2))
        except ValueError:
            print('Pimax value not found in pre-computed values. Use precompute_kz_integral() to generate first. Continuing with a pimax value of 30.0 Mpc/h')
            pimax = 30
            i_to_use = list(pimax_values).index(round(pimax, 2))
        front_constant = b_gal * pimax / np.pi
        return front_constant * (interpolate.splev(np.log10(K), splines[i_to_use]))
    
    # for last ingetral
    def K_integrand(K, Rmin, Rmax, pimax, b_gal, PS_min = 10**-4, PS_max = 100):
        kz_integral = get_kz_integral_spl(K, pimax, b_gal)
        return K * get_fancyJ(Rmin, Rmax, K) * kz_integral

    def get_model_est(Rmin, Rmax, pimax, b_gal, tau=1, PS_min = 10**-4, PS_max = 100):
        
        bar_wp = get_bar_wp(Rmin, Rmax)
        K_integral = scipy.integrate.romberg(K_integrand, a=PS_min, b=PS_max, args=[Rmin, Rmax, pimax, b_gal], rtol=1.48e-8, divmax=15) # integrate over K
        
        return tau * K_integral / (2*pimax + bar_wp)
    
    randoms = 0
    if rel_e_randoms is not None:
        if rel_e_randoms['R_bin_min'][0] != rel_e_measurement['R_bin_min'][0]:
            print('Warning - R bins do not match between rel_e_measurement and rel_e_randoms!! Continuing with the R bins from rel_e_measurement')
        randoms = rel_e_randoms['relAng_plot']
        
    R_bin_centers = (rel_e_measurement['R_bin_min']+rel_e_measurement['R_bin_max'])/2
    R_bin_edges = np.append(np.asarray(rel_e_measurement['R_bin_min']), rel_e_measurement['R_bin_max'][-1])
    #ia_measurement = rel_e_measurement['relAng_plot'] - randoms
    wp_values = wp_measurement['wp']

    def get_wp(R):
        return np.interp(R, R_bin_centers, wp_values)

    # getting \bar{w_p}, per-bin
    def wp_integrand(R):
        return R * get_wp(R)

    def get_bar_wp(Rmin, Rmax):
        wp_integral = scipy.integrate.romberg(wp_integrand, a=Rmin, b=Rmax, rtol=1.48e-8, divmax=15) # integrate over R
        return (2 / (Rmax**2 - Rmin**2)) * wp_integral
    
    # for last ingetral
    def K_integrand(K, Rmin, Rmax, pimax, b_gal, PS_min = 10**-4, PS_max = 100):
        kz_integral = get_kz_integral_spl(K, pimax, b_gal)
        return K * get_fancyJ(Rmin, Rmax, K) * kz_integral
    
    def get_model_est(Rmin, Rmax, pimax, b_gal, tau=1, PS_min = 10**-4, PS_max = 100):
        bar_wp = get_bar_wp(Rmin, Rmax)
        K_integral = scipy.integrate.romberg(K_integrand, a=PS_min, b=PS_max, args=[Rmin, Rmax, pimax, b_gal], rtol=1.48e-8, divmax=15) # integrate over K
        
        return -tau * K_integral / (2*pimax + bar_wp)
    
    # computing the model prediction in each bin of projected separation
    model_estimates = []
    r_bin_centers = []
    for rt in rel_e_measurement:
        try:
            model_est = get_model_est(rt['R_bin_min'], rt['R_bin_max'], rt['pimax'], b_gal=b_gal, tau=1)
        except ValueError:
            print('no values found for pimax = ', rt['pimax'])
            break
        model_estimates.append(model_est)
        r_bin_centers.append((rt['R_bin_min']+rt['R_bin_max'])/2)
    model_estimates = np.asarray(model_estimates)
    r_bin_centers = np.asarray(r_bin_centers)
    
    return r_bin_centers, model_estimates*D_norm


def get_rel_e_model(R_bin_min, R_bin_max, pimax_values, wp_values, b_gal, z = 0.8, 
                 PS_min = 1e-4, PS_max = 2e2, PS_data = abacus_ps_nl, PS_z = 0.8, precomputed_kz_integral_paths = [str(kzs) for kzs in kz_spline_paths], warning_handling='once'):
    '''
    UPDATED VERSION OF ABOVE
    Compute a model prediction at each bin of projected separation. This code assumes a tau value of 1.
    
    INPUT - REQUIRED:
    ------------------
    'R_bin_min', 'R_bin_max': floats. Bin edges of the projected separation
    pimax_values: float or array of shape (n,). Line-of-sight distance that rel_e_measurement and wp were made [Mpc/h]
    wp: projected cross-correlation function between a shape and a tracer catalog. Must be made in the same bins as the rel_e_measurement
    b_gal: float. bias of the galaxy catalog used for tracers
    z: float. redshift of galaxies used in measurement
    
    INPUT - OPTIONAL:
    ------------------
    PS_min, PS_max: floats. Range of k to use for the power spectrum [h/Mpc]
    PS_data: dictonary or DataFrame of matter power spectrum values. Must contain columns 'k' and 'P'. Default is a non-linear matter power spectrum from AbacusSummit.
    PS_z = float. redshift of the power spectrum. Default is 0.8.
    precomputed_kz_integral_paths: list of paths to precomputed values of the kz integral afor given pi values, made with the above power spectum. 
        Must be formatted as the output of precompute_kz_integrand.
    '''
    warnings.filterwarnings(warning_handling)
        
    # if the redshift of the measurement and power spectrum are different, adjust using growth factor
    D_norm = 1
    if z != PS_z:
        D_norm = (D(z) / D(PS_z))**2

        
    # read in the pre-computed value of the kz integral (made with precompute_kz_integral)
    splines = [np.load(path, allow_pickle=True) for path in precomputed_kz_integral_paths]
    pimax_kz_values = [round(float(path.split('_')[-1].split('.npy')[0]), 2) for path in precomputed_kz_integral_paths]

    def get_kz_integral_spl(K, pimax, b_gal):
        try:
            i_to_use = list(pimax_kz_values).index(round(pimax, 2))
        except ValueError:
            print('Pimax value of ', pimax, 'not found in pre-computed values. Use precompute_kz_integral() to generate first. Continuing with a pimax value of 30.0 Mpc/h')
            pimax = 30
            i_to_use = list(pimax_kz_values).index(round(pimax, 2))
        front_constant = b_gal * pimax / np.pi
        return front_constant * (interpolate.splev(np.log10(K), splines[i_to_use]))
    
    # for last ingetral
    def K_integrand(K, Rmin, Rmax, pimax, b_gal, PS_min = 10**-4, PS_max = 100):
        kz_integral = get_kz_integral_spl(K, pimax, b_gal)
        return K * get_fancyJ(Rmin, Rmax, K) * kz_integral

    def get_model_est(Rmin, Rmax, pimax, b_gal, tau=1, PS_min = 10**-4, PS_max = 100):
        
        bar_wp = get_bar_wp(Rmin, Rmax)
        K_integral = scipy.integrate.romberg(K_integrand, a=PS_min, b=PS_max, args=[Rmin, Rmax, pimax, b_gal], rtol=1.48e-8, divmax=15) # integrate over K
        
        return tau * K_integral / (2*pimax + bar_wp)
        
    R_bin_centers = (R_bin_min+R_bin_max)/2
    R_bin_edges = np.append(np.asarray(R_bin_min), R_bin_max[-1])

    def get_wp(R):
        return np.interp(R, R_bin_centers, wp_values)

    # getting \bar{w_p}, per-bin
    def wp_integrand(R):
        return R * get_wp(R)

    def get_bar_wp(Rmin, Rmax):
        wp_integral = scipy.integrate.romberg(wp_integrand, a=Rmin, b=Rmax, rtol=1.48e-8, divmax=15) # integrate over R
        return (2 / (Rmax**2 - Rmin**2)) * wp_integral
    
    # for last ingetral
    def K_integrand(K, Rmin, Rmax, pimax, b_gal, PS_min = 10**-4, PS_max = 100):
        kz_integral = get_kz_integral_spl(K, pimax, b_gal)
        return K * get_fancyJ(Rmin, Rmax, K) * kz_integral
    
    def get_model_est(Rmin, Rmax, pimax, b_gal, tau=1, PS_min = 10**-4, PS_max = 100):
        bar_wp = get_bar_wp(Rmin, Rmax)
        K_integral = scipy.integrate.romberg(K_integrand, a=PS_min, b=PS_max, args=[Rmin, Rmax, pimax, b_gal], rtol=1.48e-8, divmax=15) # integrate over K
        return -tau * K_integral / (2*pimax + bar_wp)
    
    # computing the model prediction in each bin of projected separation
    model_estimates = []
    for i in range(len(R_bin_min)):
        model_est = get_model_est(R_bin_min[i], R_bin_max[i], pimax_values[i], b_gal=b_gal, tau=1)
        model_estimates.append(model_est)
    model_estimates = np.asarray(model_estimates)
    
    return model_estimates*D_norm
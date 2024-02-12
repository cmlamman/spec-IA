# needs to be on cf_env
from pycorr import TwoPointCorrelationFunction
import random
import numpy as np
from scipy.interpolate import interp2d


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
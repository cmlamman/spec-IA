import numpy as np
from astropy.table import Table, join, vstack
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.cosmology import LambdaCDM, z_at_value

cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)

def get_cosmo_points(data, cosmology=cosmo):
    '''convert from astropy table of RA, DEC, and redshift to 3D cartesian coordinates in Mpc/h'''
    comoving_dist = cosmo.comoving_distance(data['Z']).to(u.Mpc)
    points = coordinates.spherical_to_cartesian(np.abs(comoving_dist), np.asarray(data['DEC'])*u.deg, np.asarray(data['RA'])*u.deg)     # in Mpc
    return np.asarray(points).transpose() * cosmology.h                                                                                 # in Mpc/h


def get_pair_coords(obs_pos1, obs_pos2, use_center_origin=True, cosmology=cosmo):
    '''
    Takes in observed positions of galaxy pairs and returns comoving coordinates, in Mpc/h, with the orgin at the center of the pair. 
    The first coordinate (x-axis) is along the LOS
    The second coordinate (y-axis) is along 'RA'
    The third coordinate (z-axis) along 'DEC', i.e. aligned with North in origional coordinates.
    
    INPUT
    -------
    obs_pos1, obs_pos2: table with columns: 'RA', 'DEC', z_column
    use_center_origin: True for coordinate orgin at center of pair, othersise centers on first position
    cosmology: astropy.cosmology
    
    RETURNS
    -------
    numpy array of cartesian coordinates, in Mpc/h. Shape (2,3)

    '''
    cartesian_coords = get_cosmo_points(vstack([obs_pos1, obs_pos2]), cosmology=cosmology)  # in Mpc/h
    # find center position of coordinates
    origin = cartesian_coords[0]
    if use_center_origin==True:
        origin = np.mean(cartesian_coords, axis=0)
    cartesian_coords -= origin
    return cartesian_coords                 # in Mpc/h
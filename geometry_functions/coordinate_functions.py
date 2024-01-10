# Useful functions for dealing with the coordinates of and making sky catalogs

import numpy as np
from astropy.table import Table, join, vstack
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM, z_at_value

cosmo = LambdaCDM(H0=69.6, Om0=0.286, Ode0=0.714)
def rw1_to_z(rw1):
    return rw1*0.2443683701202549+0.0037087929968548927
def comoving_to_z(d_comoving): # in units of Mpc/h
    return z_at_value(cosmo.comoving_distance, d_comoving * 0.7 * u.Mpc)

def v_to_cz(v, z):
    '''returns comoving Mpc / h of positions with RSD'''
    return 0.7 * (cosmo.comoving_distance(z) + ((1 + z) * (v * u.km / u.s) / cosmo.H(z))).to(u.Mpc).value

def wrap_180(angles):
    angles -= (angles >=180) * 360
    return angles

def wrap_pi(angles):
    angles -= (angles >=np.pi) * 2*np.pi
    return angles

# converting between deg and radians ( if not already in astropy anlge)
def rad_to_deg(ang_rad):
    return ang_rad * 180 / np.pi
def deg_to_rad(ang_deg):
    return ang_deg * np.pi / 180


#################################################################
## functions to limit a table of objects to only those within a given boundary

def limit_region(targets, ra1=200., ra2=205., dec1=0., dec2=5.):
    '''input targets [astropy table] and ra/dec limits'''
    try:
        return targets[(targets['RA']>ra1)&(targets['RA']<ra2)&(targets['DEC']>dec1)&(targets['DEC']<dec2)]
    except KeyError:
        return targets[(targets['TARGET_RA']>ra1)&(targets['TARGET_RA']<ra2)&(targets['TARGET_DEC']>dec1)&(targets['TARGET_DEC']<dec2)]

def radial_region(targets, ra, dec, r):
    '''limit to region within r [deg] of given coords [deg]'''
    return targets[(get_sep(ra, dec, targets['RA'], targets['DEC'])<r)]

def donut_region(targets, ra, dec, r_min, r_max):
    '''limit to region within r [deg] of given coords [deg]'''
    seps = get_sep(ra, dec, targets['RA'], targets['DEC'])
    return targets[(seps<=r_max) & (seps>r_min)]

#################################################################

def get_sep(ra1, dec1, ra2, dec2, u_coords='deg', u_result=u.rad):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - astropy quantity of separation 
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    return (c1.separation(c2)).to(u_result)
    
def get_pa(ra1, dec1, ra2, dec2, u_coords='deg', u_result=u.rad):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - separation [deg]
    - astropy quantity of position angle of second galaxy relative to first [deg], E of N
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    pa = c1.position_angle(c2).to(u_result)
    return pa


def get_sep_pa(ra1, dec1, ra2, dec2, u_coords='deg'):
    '''
    Input: ra and decs [deg] for two objects. 
    Returns: 
    - separation [deg]
    - position angle of second galaxy relative to first [deg], E of N
    '''
    c1 = SkyCoord(ra1, dec1, unit=u_coords, frame='icrs', equinox='J2000.0')
    c2 = SkyCoord(ra2, dec2, unit=u_coords, frame='icrs', equinox='J2000.0')
    sep = c1.separation(c2).to(u.rad)
    pa = c1.position_angle(c2).to(u.rad)
    return sep, pa

def get_points(data):
    '''convert from astropy table of RA and DEC to cartesian coordinates on a unit sphere'''
    points = SkyCoord(data['RA'], data['DEC'], unit='deg', frame='icrs', equinox='J2000.0')
    points = points.cartesian   # old astropy: points.representation = 'cartesian'
    return np.dstack([points.x.value, points.y.value, points.z.value])[0]



def add_rsd(z, v_3d, pos_3d, poo_3d=np.asarray([-3700, 0, 0])*.7):
    '''
    z: redshift
    v_3d: 3d velocities, units of km/s. shape is (n positions, 3)
    pos_3d: 3d positions, units of Mpc/h. shape is (n positions, 3)
    poo_3d: 3d position of oberver. shape is (3,) in units of Mpc/h
    returns comoving distance with rsd, units of Mpc/h
    '''
    los_vector = pos_3d - poo_3d
    los_unit = los_vector / np.linalg.norm(los_vector, axis=1)[:, np.newaxis]
    v = np.sum(v_3d*los_unit, axis=1)
    return 0.7 * (cosmo.comoving_distance(z) + ((1 + z) * (v * u.km / u.s) / cosmo.H(z))).to(u.Mpc).value




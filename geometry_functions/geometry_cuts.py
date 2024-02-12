from coordinate_functions import *

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

def cylindrical_cut(pair_table, rp_max=20, rpar_max=30):
    return pair_table[((pair_table['r_p']<20)&(pair_table['r_par']<30))] #pair_table[((pair_table['r_p']<20)&(pair_table['r_par']<30))]

#################################################################
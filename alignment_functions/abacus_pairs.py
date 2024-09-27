'''
Functions designed to find and calculate relavant IA quantities for pairs of galaxies in Abacus
----------------------------------------------------------------------------------------------
These are designed to run on a pre-made abacus light cone catalog, centered on RA = 0, DEC=0, and z=0.8, which contains the columns: 
    x_L2com: 3D cartesian positions float32[3]
    v_L2com: 3D cartesian velocities float32[3]
    RA: right ascension in degrees float32
    DEC: declination in degrees float32
    Z_noRSD: cosmological redshift float32
    Z_withRSD: redshift with RSD float32
    E1: projected ellipticity of halo along LOS float32
'''

from alignment_functions.basic_alignment import get_pair_distances, get_rel_es
from astropy.table import Table
from scipy.spatial import cKDTree
import numpy as np
import os

def generate_IA_pairs(abacus_catalog, save_name, max_los_dist=100, max_proj_dist=100, max_neighbors=6400, overwrite=True, n_batches=10):
    '''assumes galaxies are in a light cone centered on RA=0, DEC=0, z=0.8'''
    
    batch_name = save_name+"_batch"+str(n_batches)+'.fits'
    if overwrite==False:
        # check if file exists
        if os.path.exists(batch_name):
            print('Files exist for all batches, exiting ')
            return None
            
    # make tree of entire catalog
    tree = cKDTree(abacus_catalog['x_L2com'])
    
    # drawing centers from area well with uniform survey geometry
    abacus_centers = abacus_catalog[(
        (abacus_catalog['Z_noRSD'] < 1.3) &  # used to be just "Z", .95
        (abacus_catalog['Z_noRSD'] > 0.7) &   # .58
        (np.abs(abacus_catalog['RA'])<10) & 
        (np.abs(abacus_catalog['DEC'])<10))]
    # randomize order or abacus_centers
    abacus_centers = abacus_centers[np.random.RandomState(seed=42).permutation(len(abacus_centers))]
    
    dist_upper_bound = np.sqrt(max_los_dist**2 + max_proj_dist**2) + 20 # adding a bit to make sure include ones smeared from FOG
    
    for i in range(n_batches):
        batch_name = save_name+"_batch"+str(i)+'.fits'
        if overwrite==False:
            # check if file exists
            if os.path.exists(batch_name):
                continue
            
        center_batch = abacus_centers[i*len(abacus_centers)//n_batches:(i+1)*len(abacus_centers)//n_batches]
        
        # find pairs
        dd, ii = tree.query(center_batch['x_L2com'], distance_upper_bound = dist_upper_bound, k=max_neighbors)
        
        # calculate separations relative to LOS
        r_projected, r_parallel, s_parallel = get_pair_distances(abacus_catalog, ii)
        
        pair_results = Table()
        pair_results['r_p'] = r_projected
        pair_results['r_par'] = r_parallel 
        pair_results['s_par'] = s_parallel
        
        e1_re = get_rel_es(abacus_catalog, ii)
        pair_results['e1_rel'] = e1_re[0]
        
        # cylindrical cut down to relevant pairs
        ii_keep = ((pair_results['r_p']<max_proj_dist)&(pair_results['r_par']<max_los_dist))
        pair_results = pair_results[ii_keep]  ### to improve code, this cut should be done before calculating s_par and e1_rel
        
        pair_results.write(batch_name, overwrite=overwrite)
    
    return None
    
    ## TO DO:  need to calculate correlation function for this box. 
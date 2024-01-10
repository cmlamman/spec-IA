import numpy as np
import matplotlib.pyplot as plt
from ..geometry_functions.coordinate_functions import *

def plot_points(cat, s=.1, coordinates='sky'):
    '''Plot summary plots of Abacus ellipsoids'''
    fig = plt.figure(figsize=(18,5))
    plt.subplot(131)
    if coordinates=='sky':
        plt.scatter(wrap_180(cat['RA']), cat['DEC'], alpha=.02, s=s)
        plt.xlabel('RA'); plt.ylabel('DEC')
        plt.subplot(132)
        
        plt.title('No RSD');
        comoving_noRSD = cosmo.comoving_distance(cat['Z_noRSD']) * .7
        plt.xlabel('Radial Comoving [Mpc/h]'); plt.ylabel('DEC');
        plt.scatter(comoving_noRSD, cat['DEC'], alpha=.02, s=s)
        
        plt.subplot(133)
        plt.title('With RSD')
        comoving_withRSD = cosmo.comoving_distance(cat['Z_withRSD']) * .7
        plt.xlabel('Radial Comoving [Mpc/h]'); plt.ylabel('DEC');
        plt.scatter(comoving_withRSD, cat['DEC'], alpha=.02, s=s)
    
    elif coordinates=='cartesian':
        plt.scatter(cat['x_L2com'][::,2], cat['x_L2com'][::,1], alpha=.02, s=s)
        plt.xlabel('z'); plt.ylabel('y')
        plt.subplot(132)        
        plt.title('No RSD');
        plt.xlabel('x'); plt.ylabel('y');
        plt.scatter(cat['x_L2com'][::,0], cat['x_L2com'][::,1], alpha=.02, s=s)
        plt.subplot(133)
        plt.title('With RSD')
        comoving_withRSD = cosmo.comoving_distance(cat['Z_withRSD']) * .7
        plt.xlabel('Radial Comoving [Mpc/h]'); plt.ylabel('DEC');
        plt.scatter(comoving_withRSD, cat['DEC'], alpha=.02, s=s)
        
        
def four_plot(x, y, c):
    '''plot 4 copies of the same plot, with different x and y signs'''
    return [x, x, -x, -x], [y, -y, y, -y], [c,c,c,c]
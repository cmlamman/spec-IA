import numpy as np
import matplotlib.pyplot as plt
from geometry_functions.coordinate_functions import *
import matplotlib.colors as colors


##############################################
# PLOTTING POINTS
##############################################

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


def plot_cat_points(cat, s=.1, coordinates='sky'):
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
        
        
##############################################
# PLOTTING RESULTS - r_p, r_par space
##############################################
        
def four_plot(x, y, c):
    '''plot 4 copies of the same plot, with different x and y signs'''
    return [x, x, -x, -x], [y, -y, y, -y], [c,c,c,c]

def reflect4(rp_bins, rpar_bins, matrix):
    plot_quad = np.concatenate((np.flipud(matrix), matrix), axis=0)
    plot_quad = np.concatenate((np.fliplr(plot_quad), plot_quad), axis=1)
    rp_bins_quad =  np.concatenate([-np.flipud(rp_bins), rp_bins])
    rpar_bins_quad =  np.concatenate([-np.flipud(rpar_bins), rpar_bins])
    return rp_bins_quad, rpar_bins_quad, plot_quad


def plot_quad_reflect(rp_bins, rpar_bins, matrix, plot_args = {'cmap':'BuPu', 'levels':100}, save_name=False, title='', ctitle=''):
    rp_bins_quad, rpar_bins_quad, matrix_quad = reflect4(rp_bins, rpar_bins, matrix)
    
    fig = plt.figure(figsize=(4,5))
    fts=14 
    lbs=12
    
    plt.contourf(rp_bins_quad, rpar_bins_quad, matrix_quad, **plot_args);

    plt.axis('equal')
    color_bar = plt.colorbar()
    color_bar.set_label(label=ctitle,size=fts);
    color_bar.ax.tick_params(labelsize=lbs,size=fts)
    plt.xlabel('Projected Distance [Mpc/h]', fontsize=fts)
    plt.ylabel('Transverse Distance with RSD[Mpc/h]', fontsize=fts);
    plt.xticks(fontsize=lbs); plt.yticks(fontsize=lbs);
    plt.tick_params(size=8, which='both')
    plt.title(title)
    if save_name!=False:
        fig.savefig('/pscratch/sd/c/clamman/plots_to_download/'+save_name+'.png', dpi=400, bbox_inches='tight')
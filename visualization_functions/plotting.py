import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

from geometry_functions.coordinate_functions import *
import matplotlib.colors as colors
import glob


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

def get_combined_quad_plot(matrix, rp_bins, rpar_bins):
    plot_quad = np.concatenate((np.flipud(matrix), matrix), axis=0)
    plot_quad = np.concatenate((np.fliplr(plot_quad), plot_quad), axis=1)
    rp_bins_quad =  np.concatenate([-np.flipud(rp_bins), rp_bins])
    rpar_bins_quad =  np.concatenate([-np.flipud(rpar_bins), rpar_bins])
    return rp_bins_quad, rpar_bins_quad, plot_quad


def plot_quad_reflect(rp_bins, rpar_bins, matrix, plot_args = {'cmap':'BuPu', 'levels':100}, save_name=False, title='', ctitle=''):
    rp_bins_quad, rpar_bins_quad, matrix_quad = get_combined_quad_plot(matrix, rp_bins, rpar_bins)

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
        
        
        
##############################################
# PLOTTING ALIGNMENT RESULTS
##############################################

def plot_groupAlignment_results(result_paths, random_paths=None, groupCorr_paths=None, labels=['label'], colors=['b'], linestyles=['-'],
                                fts=16, figsize=(10,8), save_path=None, dpi=200, use_pimax_gradient=False, ylim=None):
    
    fig = plt.figure(figsize=figsize)
    falpha = 0.08 # shading of error region
    
    for i in range(len(result_paths)):
        results = Table.read(result_paths[i])
        
        # sorting out pimax
        pimax = results['pimax']
        # if all pimax have the same value...
        if np.all(pimax==pimax[0]):
            pimax_label = '\n'+r'$\Pi_{\rm max} = '+str(pimax[0])+'h^{-1}$ Mpc'
        elif use_pimax_gradient==True:
            pimax_label = ''
            # make the plot have a background grey gradient which corresponds to the pimax value in each R_bin
            plt.text(results['R_bin_min'][0]+.12, -.35, r'$\Pi_{\rm max}$ [$h^{-1}$Mpc]', fontsize=fts-3, ha='left', alpha=.3)
            for j in range(len(results)):
                pvalue = 1 - ((results['pimax'][j] / np.max(results['pimax'])-.1)/5)
                pcolor = (pvalue**1.2, pvalue**1.2, pvalue)
                plt.fill_between([results['R_bin_min'][j], results['R_bin_max'][j]], [-.5, -.5], [.5, .5], alpha=1, color=pcolor, zorder=-1)
                # make a vertical line separating ecah bin
                plt.plot([results['R_bin_min'][j], results['R_bin_min'][j]], [-.5, .5], color='grey', linewidth=.1, zorder=0)
                # write pimax value at the bottom
                plt.text((results['R_bin_min'][j]+results['R_bin_max'][j])/2, -.4, str(round(results['pimax'][j], 2)), fontsize=fts-5, ha='center', alpha=.3)
        
        else:
            pimax_label = '\n'+r'Variable $\Pi_{\rm max} '
        
        if random_paths[i]==None:
            print('no randoms found for', labels[i])
            randoms = 0; randoms_e = 0
            
        else:
            random_batch_paths = glob.glob(random_paths[i])
            if len(random_batch_paths)>1:
                print('multiple random files found for', labels[i], 'averageing')
                random_table_batch = [Table.read(path) for path in random_batch_paths]
                randoms = np.mean(np.asarray([random_table['relAng_plot'] for random_table in random_table_batch]), axis=0)
                randoms_e = np.sqrt(np.sum(np.asarray([random_table['relAng_plot_e']**2 for random_table in random_table_batch]), axis=0))
            else:
                random_table = Table.read(random_paths[i])
                randoms = random_table['relAng_plot']
                randoms_e = random_table['relAng_plot_e']
                
            if round(np.max(random_table['R_bin_max']))!=round(np.max(results['R_bin_max'])) or len(random_table)!=len(results):
                print('randoms and results have different R binning!')
                
        
        # projected 2-point correlation between groups and tracers
        if groupCorr_paths!=None:
            groupCorr = Table.read(groupCorr_paths[i])
        else:
            groupCorr = 1
        
        ang_plot = results['relAng_plot'] - randoms
        ang_plot_e = np.sqrt(results['relAng_plot_e']**2 + randoms_e**2)
        xvalues = (results['R_bin_min'] + results['R_bin_max'])/2
        
        ang_plot *= groupCorr
        ang_plot_e  *= groupCorr
        
        # plot the errorbars as a shaded region
        plt.fill_between(xvalues, xvalues*(ang_plot-ang_plot_e), xvalues*(ang_plot+ang_plot_e), alpha=falpha, color=colors[i]);
        plt.errorbar(xvalues, xvalues*ang_plot, yerr=xvalues*ang_plot_e, label=labels[i], alpha=1, color=colors[i],
                    marker='o', markersize=4, capsize=3, elinewidth=1, linewidth=2.2, linestyle=linestyles[i], zorder=0);
        
    plt.xlabel(r'Transverse separation, $R$ [$h^{-1}$Mpc]', fontsize=fts)
    plt.ylabel(r'Relative orientation of groups, $R$cos(2$\Phi$)'+pimax_label, fontsize=fts)
    if groupCorr_paths!=None:
        plt.ylabel(r'Relative orientation of groups, $w_p(R)\times R$cos(2$\Phi$)'+pimax_label, fontsize=fts)
    plt.xscale('log')
    # add dotted line at 0
    plt.plot([0,150], [0,0], color='grey', linewidth=.4, zorder=0, dashes=(16,16))
    plt.legend(fontsize=fts)
    # make axis labels larger
    plt.xticks(fontsize=fts-2); plt.yticks(fontsize=fts-2)
    #if use_pimax_gradient==True:
    #    plt.colorbar(label=r'$\Pi_{\rm max}$ [Mpc]', pad=0.01)
    
    if save_path!=None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    plt.ylim(ylim)
    plt.show()
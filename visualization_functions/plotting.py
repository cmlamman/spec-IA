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

def plot_MIA_results(result_paths, random_paths=[None], labels=['label'], colors=['b'], linestyles=['-'],
                                fts=16, figsize=(10,8), save_path=None, dpi=200, use_pimax_gradient=False, ylim=None, title='Multiplet Alignment', 
                                plot_just_randoms=False, i_end=None, legend_title='', column_multiply=None, 
                                ylabel=r'Multiplet relative orientation, $R$cos(2$\Phi$)', baseline = 0, 
                                legend=True, legend_loc='lower left', save_mrt='no', falpha=.08):
    '''
    Function to plot the results of multiplet alignmnet measurements.
    result_paths: list of paths to tables of results containing these columns: 
                'R_bin_min','R_bin_max': min and max of r_p bins, assumed to be in Mpc/h
                'relAng_plot','relAng_plot_e': relative orientation of multiplets and and error
                'pimax' (optional): oimax in Mpc/h, needed if using the pimax gradient background
                'multiplier' (optional): plotted results will be multiplied by this factor, as long as column_multiply is set to 'multiplier'
                'relAng_plot','relAng_plot_e', and optionally: 'pimax','multiplier'
    random_paths: list of paths to tables of randoms containing same columns as above. Will be subtracted from the results. Errors will be added in quadrature.
    labels: list of labels for each result, will be used in legend
    colors, linestyles: lists, arguments will be used in plotting the results
    fts: fontsize for the plot
    figsize, title, legend_title, ylabel, dpi, legend_loc: arguments for the plot and saving it
    column_multiply: float, list of floats, or string. Will multiply the results by thsis value. If a list, will be used for each result. If string, will use this column in results table.
    save_path: path to save the plot, default is None
    use_pimax_gradient: if True, will plot a gradient background based on the pimax value of each bin, as provided in the results table
    ylim: tuple of y min and y max values for the plot, default is None
    plot_just_randoms: if True, will plot only the randoms.
    i_end: if not None, will only plot the first i_end points of the results
    basline: value to subtract from the results, default is 0
    legend: if True, will plot the legend
    save_mrt: str. if anything except 'no', will save the results as MRTs to the argument provided
    falpha: alpha value for the fill_between plot, default is 0.08
    '''
    
    ymin = ylim[0] if ylim!=None else -.35                      # figuring out the bounds of the plot and relevant scalings for the height of text
    yscale = (ylim[1] - ymin)/.85 if ylim!=None else 1
    
    fig = plt.figure(figsize=figsize)
    
    for i in range(len(result_paths)):                          # loop over every path provided
        results = Table.read(result_paths[i])
        
        if random_paths[i]==None:
            print('no randoms found for', labels[i])
            randoms = 0; randoms_e = 0
            
        else:
            random_batch_paths = glob.glob(random_paths[i])
            if len(random_batch_paths)>1:
                print(len(random_batch_paths), 'random files found for', labels[i], 'averaging')
                random_table_batch = [Table.read(path) for path in random_batch_paths]
                randoms = np.average(np.asarray([random_table['relAng_plot'] for random_table in random_table_batch]), axis=0, weights=(1/np.asarray([random_table['relAng_plot_e']**2 for random_table in random_table_batch])))
                randoms_e = np.std(np.asarray([random_table['relAng_plot'] for random_table in random_table_batch]), axis=0) / np.sqrt(len(random_table_batch))
                random_table = random_table_batch[0]
            else:
                random_table = Table.read(random_paths[i])
                randoms = random_table['relAng_plot']
                randoms_e = random_table['relAng_plot_e']
                
            if round(np.max(random_table['R_bin_max']))!=round(np.max(results['R_bin_max'])) or len(random_table)!=len(results):
                print('randoms and results have different R binning!')        
        
        mult_factor = 1                                            # for scaling the plotted results                      
        if column_multiply!=None:
            if type(column_multiply[i])==str:
                mult_factor = results[column_multiply]
            elif type(column_multiply[i])==float:
                mult_factor = column_multiply[i]
        
        ang_plot = (results['relAng_plot'] - randoms)              # subtracting randoms and computing total error
        ang_plot_e = np.sqrt(results['relAng_plot_e']**2 + randoms_e**2)
        
        if plot_just_randoms==True:
            ang_plot = randoms
            ang_plot_e = randoms
        
        xvalues = (results['R_bin_min'] + results['R_bin_max'])/2
        ang_plot *= mult_factor
        ang_plot_e  *= mult_factor
        ang_plot -= baseline
        
        if i_end==None:                                             # for just plotting a subset of the provided results
            i_end = len(xvalues)
        else:
            if len(xvalues)<10:
                i_end = len(xvalues)-1
        # plot the errorbars as a shaded region
        plt.fill_between(xvalues[:i_end], (xvalues*(ang_plot-ang_plot_e))[:i_end], (xvalues*(ang_plot+ang_plot_e))[:i_end], alpha=falpha, color=colors[i]);
        plt.errorbar(xvalues[:i_end], (xvalues*ang_plot)[:i_end], yerr=(xvalues*ang_plot_e)[:i_end], label=labels[i], color=colors[i],
                     marker='o', markersize=6, capsize=5, elinewidth=1.5, linewidth=2, linestyle=linestyles[i], zorder=0, alpha=1);
        
        # print average SNR
        snr = np.mean((ang_plot/ang_plot_e)[:i_end])
        print('snr--------')
        print(labels[i], ' : ', snr)
        print('-----------')
        
        if save_mrt != 'no':
            # saving above plot as MRTs
            if i==0:
                towrite = Table()
                towrite['R'] = Column((xvalues[:i_end]).round(3) * u.Mpc / u.h, description='transverse separation')
                towrite['pimax']  = Column((results['pimax'][:i_end]).round(3) * u.Mpc / u.h, description='maximum LOS separation')
            towrite['relative_orientation_'+str(i)] = Column(ang_plot[:i_end].round(4), description='relative multiplet orientation of '+labels[i])
            towrite['relative_orientation_'+str(i)+'_e'] = Column(ang_plot_e[:i_end].round(4), description='relative multiplet orientation error')
        
    results = Table.read(result_paths[0]) 
    # sorting out pimax
    pimax = results['pimax'][:i_end]
    # if all pimax have the same value...
    if np.all(pimax==pimax[0]):
        pimax_label = '\n'+r'$\Pi_{\rm max} = '+str(pimax[0])+'h^{-1}$ Mpc'
    elif use_pimax_gradient==True:
        pimax_label = ''
        # make the plot have a background grey gradient which corresponds to the pimax value in each R_bin
        plt.text(results['R_bin_min'][0]+.15, ymin+0.05*yscale, r'Maximum LOS distance [$h^{-1}$Mpc]', fontsize=fts-3, ha='left', alpha=.3)
        for j in range(len(results))[:i_end]:
            pvalue = .99 -  ((results['pimax'][j] / 91.58-.1)/5)
            pcolor = (pvalue**1.5, pvalue**1.5, pvalue)
            pcolor = [p**.5 for p in pcolor]
            
            plt.fill_between([results['R_bin_min'][j], results['R_bin_max'][j]], [-1, -1], [1.5, 1.5], alpha=1, color=pcolor, zorder=-1)
            # make a vertical line separating ecah bin
            plt.plot([results['R_bin_min'][j], results['R_bin_min'][j]], [-1, 1.5], color='grey', linewidth=.1, zorder=0)
            # write pimax value at the bottom
            plt.text((results['R_bin_min'][j]+results['R_bin_max'][j])/2, ymin+0.02*yscale, str(round(results['pimax'][j], 2)), fontsize=fts-5, ha='center', alpha=.3)
    else:
        pimax_label = '\n'+r'Variable $\Pi_{\rm max}$'
            
        
    plt.xlabel(r'Transverse separation, $R$ [$h^{-1}$Mpc]', fontsize=fts)
    plt.ylabel(ylabel+pimax_label, fontsize=fts)
    plt.xscale('log')
    # add dotted line at 0
    plt.plot([0,150], [0,0], color='grey', linewidth=.4, zorder=0, dashes=(16,16))
    if legend==True and legend_loc=='lower left':
        plt.legend(fontsize=fts-3, loc=legend_loc, bbox_to_anchor=(0.05, 0.15), title=legend_title)#bbox_to_anchor=(0.15, 0.12)) (0.3, 0.1)
    elif legend==True:
        plt.legend(fontsize=fts-3, loc=legend_loc, title=legend_title)
    # make axis labels larger
    #plt.xticks(fontsize=fts-2); plt.yticks(fontsize=fts-2)
    plt.xticks(fontsize=fts); plt.yticks(fontsize=fts)
    
    plt.title(title)
    plt.ylim(ylim)
    plt.xlim([np.min(results['R_bin_min'][:i_end]), np.max(results['R_bin_max'][:i_end])])
    if save_path!=None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    
    if save_mrt != 'no':
        towrite.write('multipletIA_paper_mrts/'+save_mrt+'.dat', format='ascii.mrt', overwrite=True)
    
    return ang_plot, ang_plot_e
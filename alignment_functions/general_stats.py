import numpy as np
import astropy.units as u

def remove_astropyu(values_list, unit=u.Mpc):
    if isinstance(values_list[0], u.Quantity):
        return [v.to(u.Mpc).value for v in values_list]
    else:
        return values_list

def get_bin_centers(bin_arr):
    return (bin_arr[1:] + bin_arr[:-1])/2

def bin_sum_not_scipy(x_values, y_values, x_bins, statistic='sum', err=False, weights=None):
    '''
    alternative to scipy, which sometimes has issues handling binnind with very non-uniform footprints
    x_bins = bin edges
    '''
    inds = np.digitize(x_values, x_bins)
    
    if statistic=='sum':
        y_sums = []
        y_errs = []
        for ind in range(len(x_bins)-1):
            y_bin = y_values[(inds==(ind+1))]
            y_sums.append(np.nansum(y_bin))#; y_std.append(np.std(y_bin))
            y_errs.append(np.nanstd(y_bin) / np.sqrt(len(y_bin)))
        if err==True:
            return np.asarray(remove_astropyu(y_sums)), np.asarray(remove_astropyu(y_errs))
        else:
            return np.asarray(remove_astropyu(y_sums))#, np.asarray(y_std)
    
    if statistic=='mean':
        y_sums = []
        y_errs = []
        for ind in range(len(x_bins)-1):
            y_bin = y_values[(inds==(ind+1))]
            y_sums.append(np.nanmean(y_bin))#; y_std.append(np.std(y_bin))
            y_errs.append(np.nanstd(y_bin) / np.sqrt(len(y_bin)))
        if err==True:
            return np.asarray(remove_astropyu(y_sums)), np.asarray(remove_astropyu(y_errs))
        else:
            return np.asarray(remove_astropyu(y_sums))
        
        
def bin_results(seps, reles, rp_bins, weights=None): 
    '''sep_max really does nothing'''
    
    if(weights is None):
        weights = np.asarray([1]*len(reles))
        
    i_keep = (seps<np.max(rp_bins))
    reles = reles[i_keep]
    seps = seps[i_keep]
    weights = weights[i_keep]
    
    rp_bin_centers = (rp_bins[1:]+rp_bins[:-1])/2   # just using centers of bins here. but could change to do mean rp value in each bin
        
    msum = bin_sum_not_scipy(seps, reles*weights, x_bins=rp_bins, statistic='sum')
    wsum = bin_sum_not_scipy(seps, weights, x_bins=rp_bins, statistic='sum')
    wmeans = msum / wsum

    return rp_bin_centers, wmeans
        
def get_cov_matrix_from_regions(signal_regions):
    '''signa_regions is array of shape (n_regions, n_bins)'''
    
    cv = signal_regions.T
    cn = len(cv)
    cv_kk = np.zeros(cn**2).reshape((cn, cn))
    epsilon = 1e-10  # small epsilon value to avoid division by zero
    for i in range(len(cv)):
        for j in range(len(cv)):
            cv_i = np.sum(cv[i] * cv[i])
            cv_j = np.sum(cv[j] * cv[j])
            cv_kk[i][j] = np.sum(cv[i] * cv[j]) / np.sqrt((cv_i + epsilon) * (cv_j + epsilon))
    
    return cv_kk
    
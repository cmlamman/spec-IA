import numpy as np
import astropy.units as u

def remove_astropyu(values_list, unit=u.Mpc):
    if isinstance(values_list[0], u.Quantity):
        return [v.to(u.Mpc).value for v in values_list]
    else:
        return values_list

def get_bin_centers(bin_arr):
    return (bin_arr[1:] + bin_arr[:-1])/2

def bin_sum_not_scipy(x_values, y_values, x_bins, statistic='sum', err=False):
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
            y_sums.append(np.sum(y_bin))#; y_std.append(np.std(y_bin))
            y_errs.append(np.std(y_bin) / np.sqrt(len(y_bin)))
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
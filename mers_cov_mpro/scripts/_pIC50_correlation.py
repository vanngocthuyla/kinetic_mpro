import numpy as np
import os
import pandas as pd

import scipy
from scipy import stats
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from _chemical_reactions import ChemicalReactions


def _wm(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def _cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - _wm(x, w)) * (y - _wm(y, w))) / np.sum(w)


def _corr(x, y, w):
    """Weighted Correlation"""
    return _cov(x, y, w) / np.sqrt(_cov(x, x, w) * _cov(y, y, w))


def corr_matrix(data, keys, method='pearsonr'):
    """
    Input: dataframe
    Keys: column names of dataframe to calculate correlation coefficients
    Params:
        The method to calculate correlation coefficients can be 'pearsonr', 'spearmanr' or 'kendall'
    """
    table = pd.DataFrame(columns=keys, index=range(len(keys)))
    for i in range(1, len(keys)):
        for j in range(i):
            # x = pd.DataFrame.to_numpy(data[keys[i]])
            x = data[keys[i]]
            # y = pd.DataFrame.to_numpy(data[keys[j]])
            y = data[keys[j]]
            dat = pd.DataFrame([x, y], index=['X', 'Y']).T
            dat = dat.dropna()
            if len(dat)>2:
                #Calculate the correlation
                if method=='pearsonr':
                    coef, p = stats.pearsonr(dat.X, dat.Y)
                if method=='spearmanr':
                    coef, p = stats.spearmanr(dat.X, dat.Y)
                if method=='kendall':
                    coef, p = stats.kendalltau(dat.X, dat.Y)
                text = r'n=' +str(len(dat)) +'; r=' +str('%5.3f' %coef) +'; p=' +str('%5.3e' %p)
            else:
                text = r'n=' +str(len(dat))
            table.iloc[i][keys[j]] = text
    # table = table.rename(index={0: "Fluorescence", 1: "Antiviral_Leuven", 2: "Antiviral_Zitzman", 3: "Antiviral_Takeda", 4: "Antiviral_IIBR"})
    return table


def rmsd_matrix(data, keys, method='RMSD'):
    """
    Parameters
    ----------
    data        : dataset contains the set of variables for correlation analysis
    keys        : column names of dataframe to calculate correlation coefficients
    method      : the method to calculate RMSD, can be 'RMSD', or 'aRMSD'

    return the RMSD given two datasets of two variables
    """
    table = pd.DataFrame(columns=keys, index=range(len(keys)))
    for i in range(1, len(keys)):
        for j in range(i):
            # x = pd.DataFrame.to_numpy(data[keys[i]])
            x = data[keys[i]]
            # y = pd.DataFrame.to_numpy(data[keys[j]])
            y = data[keys[j]]
            dat = pd.DataFrame([x, y], index=['X', 'Y']).T
            dat = dat.dropna()
            if len(dat)>2:
                if method=='RMSD':
                    rmsd = np.sqrt(np.mean((dat.X - dat.Y)**2))
                elif method == 'aRMSD':
                    rmsd = np.sqrt(np.mean((dat.X - dat.Y - (np.mean(dat.X) - np.mean(dat.Y)))**2))
                text = r'n=' +str(len(dat)) +'; r=' +str('%5.3f' %rmsd)
            else:
                text = r'n=' +str(len(dat))
            table.iloc[i][keys[j]] = text
    return table


def corr_pearsonr_N_sample(x, y, ax=None, **kwargs):
    """
    Parameters
    ---------- 
    x           : pd.DataFrams, dataset of one variable
    y           : pd.DataFrams, dataset of other variable

    Return the correlation coefficients of two panda dataframes in the correlogram
    """

    #Remove the NaN data
    data = pd.DataFrame([x, y], index=['X', 'Y']).T
    data = data.dropna()
    if len(data)>2:
        #Calculate the correlation
        coef, p = stats.pearsonr(data.X, data.Y)
        #Make the label
        label = r'r=' + str('%5.3f' %coef) + '; p=' + str('%5.3e' %p) + '\nN=' + str(len(data))
    else:
        #Make the label
        label = 'nan'
    #count how many annotations are already present and add the label to the plot
    if ax is None:
        ax = plt.gca()
    n = len([c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)])
    pos = (.1, .8 - .1*n)
    colors = ['blue','orange']
    ax.annotate(label, xy=pos, xycoords=ax.transAxes, color=colors[n])


def _corr_coef(x, y, method='pearsonr'):
    """
    Parameters
    ----------
    x           : numpy.array, datasets of one variable
    y           : numpy.array, datasets of other variable
    method      : The method to calculate correlation coefficients can be 'pearsonr', 'spearmanr', 'kendall', 'RMSD', or 'aRMSD'

    return the correlation given two datasets of two variables
    """
    if method=='pearsonr':
        corr, p = stats.pearsonr(x, y)
    elif method=='spearmanr':
        corr, p = stats.spearmanr(x, y)
    elif method=='kendall':
        corr, p = stats.kendalltau(x, y)
    elif method=='RMSD':
        corr = np.sqrt(np.mean((x - y)**2))
        p = None
    elif method == 'aRMSD':
        corr = np.sqrt(np.mean((x - y - (np.mean(x) - np.mean(y)))**2))
        p = None
    return corr, p


def corr_bootstrap(x, y, n_bootstrap=100, method='pearsonr'):
    """
    Parameters
    ----------
    x           : numpy.array, dataset of one variable
    y           : numpy.array, dataset of other variable
    n_bootstrap : Number of bootstrap samples
    method      : The method to calculate correlation coefficients can be 'pearsonr', 'spearmanr', 'kendall', 'RMSD', or 'aRMSD'

    return the correlation given a dataset consists of paired observations where each (x_i, y_i) pair is related to each other
    """
    bootstrap_correlations = []
    for _ in range(n_bootstrap):
        # Generate random indices with replacement
        indices = np.random.choice(len(x), size=len(y), replace=True)

        # Select corresponding elements from both x and y arrays
        resampled_x = x[indices]
        resampled_y = y[indices]

        # Calculate correlation coefficient
        corr, p = _corr_coef(x, y, method)
        bootstrap_correlations.append(corr)

        # Analyze the distribution
        mean_corr = np.mean(bootstrap_correlations)
        std_corr = np.std(bootstrap_correlations)
        # confidence_interval = np.percentile(bootstrap_correlations, [2.5, 97.5])

    return mean_corr, std_corr


def corr_bootstrap_matrix(data, keys, n_bootstrap=100, method='pearsonr'):
    """
    Parameters
    ----------
    data        : dataset contains the set of variables for correlation analysis
    keys        : column names of dataframe to calculate correlation coefficients
    method      : The method to calculate correlation coefficients can be 'pearsonr', 'spearmanr', 'kendall', 'RMSD', or 'aRMSD'
    """
    table = pd.DataFrame(columns=keys, index=range(len(keys)))
    for i in range(1, len(keys)):
        for j in range(i):
            # x = pd.DataFrame.to_numpy(data[keys[i]])
            x = data[keys[i]]
            # y = pd.DataFrame.to_numpy(data[keys[j]])
            y = data[keys[j]]
            dat = pd.DataFrame([x, y], index=['X', 'Y']).T
            dat = dat.dropna()
            x = np.array(dat.X)
            y = np.array(dat.Y)
            if len(dat)>2:
                mean_corr, std_corr = corr_bootstrap(x, y, n_bootstrap, method)
                text = r'n=' +str(len(dat)) +'; corr=' +str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
            else:
                text = r'n=' +str(len(dat))
            table.iloc[i][keys[j]] = text
    return table


def corr_leave_p_out(x, y, p=2, method='pearsonr'):
    """
    Parameters
    ----------
    x           : numpy.array, dataset of one variable
    y           : numpy.array, dataset of other variable
    p           : int, number of observation that is left out of the correlation analysis
    method      : The method to calculate correlation coefficients can be 'pearsonr', 'spearmanr', 'kendall', 'RMSD', or 'aRMSD'

    return the correlation given a dataset consists of paired observations where each (x_i, y_i) pair is related to each other
    """
    assert p<len(x), print("p should lower than number of observation")

    bootstrap_correlations = []
    # Loop over data for leave-p-out cross-validation
    for i in range(len(x) - p + 1):
        # Create dataset by leaving out p data points
        indices = np.delete(np.arange(len(x)), np.s_[i:i+p])

        # Select corresponding elements from both x and y arrays
        select_x = x[indices]
        select_y = y[indices]

        # Calculate correlation coefficient
        corr, _ = _corr_coef(select_x, select_y, method)
        bootstrap_correlations.append(corr)

        # Analyze the distribution
        mean_corr = np.mean(bootstrap_correlations)
        std_corr = np.std(bootstrap_correlations)
        # confidence_interval = np.percentile(bootstrap_correlations, [2.5, 97.5])

    return mean_corr, std_corr


def corr_leave_p_out_matrix(data, keys, p=2, method='pearsonr'):
    """
    Parameters
    ----------
    data        : dataset contains the set of variables for correlation analysis
    p           : int, number of observation that is left out of the correlation analysis
    keys        : column names of dataframe to calculate correlation coefficients
    method      : The method to calculate correlation coefficients can be 'pearsonr', 'spearmanr', 'kendall', 'RMSD', or 'aRMSD'
    """
    table = pd.DataFrame(columns=keys, index=range(len(keys)))
    for i in range(1, len(keys)):
        for j in range(i):
            # x = pd.DataFrame.to_numpy(data[keys[i]])
            x = data[keys[i]]
            # y = pd.DataFrame.to_numpy(data[keys[j]])
            y = data[keys[j]]
            dat = pd.DataFrame([x, y], index=['X', 'Y']).T
            dat = dat.dropna()
            x = np.array(dat.X)
            y = np.array(dat.Y)
            if len(dat)>2:
                mean_corr, std_corr = corr_leave_p_out(x, y, p, method)
                text = r'n=' +str(len(dat)) +'; corr=' +str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
            else:
                text = r'n=' +str(len(dat))
            table.iloc[i][keys[j]] = text

    return table


def _pd_mean_std(df, name):
    ID = np.unique(df['ID'])
    mean = []
    std = []
    for _ID in ID:
        mean.append(df[name][df.ID == _ID].mean())
        std.append(df[name][df.ID == _ID].std())
    return pd.DataFrame([ID, mean, std], index=['ID', name, name+'_std']).T
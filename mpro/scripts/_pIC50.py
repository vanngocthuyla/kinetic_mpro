import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import os
import pandas as pd
import pickle
import arviz as az
from scipy.optimize import minimize, curve_fit

import matplotlib.pyplot as plt
import seaborn as sns

from _chemical_reactions import ChemicalReactions
from _kinetics import ReactionRate_DimerOnly


def f_curve_vec(x, R_b, R_t, x_50, H):
    """
    Dose-response curve function

    Parameters:
    ----------
    x   : array
          log_10 of concentration of inhibitor
    R_b : float
          bottom response
    R_t : float
          top response
    x_50: float
          logIC50
    H   : float
          hill slope
    ----------
    return an array of response
    """
    return R_b+(R_t-R_b)/(1+10**(x*H-x_50*H))


def parameter_estimation(data, theta=None, variances=None, itnmax=100, tol=1e-4):
    """
    Fitting non-linear regression without control
    Parameters:
    ----------
    data      : list of two element: x, y, c1, c2.
                x is vector of concentration
                y is vector of responsex and y must be same length
    theta     : vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    ----------
    return [theta, ASE, variance]
    """

    # Initial value and boundary for theta
    min_y = min(data['y'])
    max_y = max(data['y'])
    range_y = max_y - min_y

    if theta is None:
        theta = [min_y, max_y, data['x'][np.argmin(np.square(data['y']-np.mean(data['y'])))], 1.0]
        upper = [min_y + 0.25*range_y, max_y + 0.25*range_y, 0, 20]
        lower = [min_y - 0.25*range_y, max_y - 0.25*range_y, -20, -20]
    else:
        upper = [theta[0] + 0.25*range_y, theta[1] + 0.25*range_y, 0, 20]
        lower = [theta[0] - 0.25*range_y, theta[1] - 0.25*range_y, -20, -20]

    fit_f, var_matrix, _, mes, ier = curve_fit(f_curve_vec, xdata=np.array(data['x']), ydata=np.array(data['y']),
                                               absolute_sigma=True, p0=theta, 
                                               bounds=(lower, upper), full_output=True)

    # Estimate ASE for theta
    y_hat = f_curve_vec(data['x'], *fit_f)
    sigma = np.sqrt(np.sum((y_hat - data['y'])**2)/(len(y_hat)-4))
    ASE = np.sqrt(np.diag(var_matrix))*sigma #unscale_SE*sigma

    if ier in [1, 2, 3, 4]:
        mle = [fit_f, ASE, np.array([sigma**2])]
    else:
        fit_f = ASE = [0, 0, 0, 0]
        mle = [fit_f, ASE, None]

    return mle


def scaling_data(y, bottom, top):
    """
    This function is used to normalize data by mean of top and bottom control

    Parameters:
    ----------
    y     : vector of response
    bottom: mean of vector of control on the bottom
    top   : mean of vector of control on the top
    ----------

    return vector of normalized response
    """
    min_y = min(bottom, top)
    max_y = max(bottom, top)
    return (y-min_y)/abs(max_y - min_y)*100


def f_parameter_estimation(x, y):
    """
    Fitting non-linear regression for concentration-response datasets and estimate 4 parameters
    
    Parameters:
    ----------
    x         : vector of log10 concentration
    y         : vector of response
    theta     : vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    ----------
    return Rb, Rt, pIC50, hill slope and pIC90
    """
    assert len(x)==len(y), print("Vectors of concentration and data should have the same length.")
    min_y = min(y)
    max_y = max(y)
    range_y = max_y - min_y

    theta0 = [min_y, max_y, x[np.argmin(np.square(y-np.mean(y)))], 1.0]
    upper = [min_y + 0.25*range_y, max_y + 0.25*range_y, 0, 20]
    lower = [min_y - 0.25*range_y, max_y - 0.25*range_y, -20, -20]

    try:
        res, var_matrix, _, mes, ier = curve_fit(f_curve_vec, xdata=x, ydata=y, absolute_sigma=True, p0=theta0,
                                                 bounds=(lower, upper), full_output=True)
        if ier in [1, 2, 3, 4]:
            [Rb, Rt, x50, H] = res
            if H!=0:
                pIC90 = f_pIC90(-x50, H)
                return [Rb, Rt, -x50, H, pIC90]
            else:
                return [Rb, Rt, -x50, H, 0]
        else: 
            return [0, 0, 0, 0, 0]
    except: 
        return [0, 0, 0, 0, 0]


def _adjust_trace(dat, logK_dE_alpha):
    """
    Adjust trace by values from MAP

    Parameters:
    ----------
    dat          : dataframe converted from trace using arviz.InferenceData.to_dataframe(arviz.convert_to_inference_data(trace))
    logK_dE_alpha: dict, information of fixed logK, dE and alpha
    ----------
    return new dataframes with added parameters from logK_dE_alpha
    """ 
    params_list = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'kcat_MS', 'kcat_DS', 'kcat_DSS']
    for param in params_list:
        if (not param in dat.columns) and (param in logK_dE_alpha.keys()):
            dat.insert(len(dat.columns), param, logK_dE_alpha[param]*np.ones(len(dat)))
    return dat


def _pIC_hill(df, logDtot, logStot, logItot):
    """
    The function first simulates the dimer-only concentration-response curve (CRC) from mcmc trace, 
    then estimate the pIC50 and hill slopes for each CRC
    
    Parameters:
    ----------
    df        : dataframe, each row corresponding to one set of parameter from mcmc trace
    logDtot   : vector of dimer concentration
    logStot   : vector of substrate concentration
    logItot   : vector of inhibitor concentration
    ----------
    return list of 5 parameters
    """

    f_v = vmap(lambda logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, kcat_DS, kcat_DSI, kcat_DSS: ReactionRate_DimerOnly(logDtot, logStot, logItot, logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, kcat_DS, kcat_DSI, kcat_DSS))
    v_sim = f_v(jnp.array(df.logK_S_D), jnp.array(df.logK_S_DS), jnp.array(df.logK_I_D), jnp.array(df.logK_I_DI), jnp.array(df.logK_S_DI), jnp.array(df.kcat_DS), jnp.array(df.kcat_DSI), jnp.array(df.kcat_DSS))

    v_min = [jnp.min(v) for v in v_sim]
    v_max = [jnp.max(v) for v in v_sim]

    f_scaling = vmap(lambda v, _min, _max: (v - _min)/(_max - _min)*100)
    x = np.log10(np.exp(logItot))
    ys = f_scaling(v_sim, np.array(v_min), np.array(v_max))

    f_theta = lambda y: f_parameter_estimation(x, np.array(y))
    thetas = np.array(list(map(f_theta, ys)))

    return thetas.T


def table_pIC_hill_one_inhibitor(inhibitor, mcmc_dir, logDtot, logStot, logItot, 
                                 logK_dE_alpha=None, OUTDIR=None):
    """
    For one inhibitor, dimer-only pIC50s can be simulated given the specified values of 
    dimer/substrate concentrations and kinetic parameters from mcmc trace. 

    Parameters:
    ----------
    inhibitor       : name of inhibitor
    mcmc_dir        : str, directory of traces
    logDtot         : vector of dimer concentration
    logStot         : vector of substrate concentration
    logItot         : vector of inhibitor concentration
    measure         : statistical measure, can be 'mean' or 'median'
    logK_dE_alpha   : dict, information of fixed logK, dE and alpha
    ----------
    return table of kinetic parameters, pIC50, and hill slope for each inhibitor
    """
    inhibitor_dir = inhibitor[7:12]
    inhibitor_name = inhibitor[:12]

    if not os.path.isfile(os.path.join(mcmc_dir, inhibitor_dir, 'traces.pickle')):
        return None

    trace = pickle.load(open(os.path.join(mcmc_dir, inhibitor_dir, 'traces.pickle'), "rb"))
    data = az.InferenceData.to_dataframe(az.convert_to_inference_data(trace))

    nthin = int(len(data)/100)
    if logK_dE_alpha is not None:
        df = _adjust_trace(data.iloc[::nthin, :].copy(), logK_dE_alpha)
    else:
        df = data.iloc[::nthin, :].copy()

    thetas = _pIC_hill(df, logDtot, logStot, logItot)
    Rb_list = thetas[0]
    Rt_list = thetas[1]
    pIC50_list = thetas[2]
    hill_list = thetas[3]
    pIC90_list = thetas[4]

    df.insert(len(df.columns), 'pIC50', pIC50_list)
    df.insert(len(df.columns), 'hill', hill_list)
    df.insert(len(df.columns), 'pIC90', pIC90_list)

    if OUTDIR is not None:
        plt.figure()
        for i in range(len(pIC50_list)):
            if hill_list[i]>0:
                temp = np.linspace(np.log10(1E-12), np.log10(1E-3), 50)
                plt.plot(temp, f_curve_vec(temp, Rb_list[i], Rt_list[i], -pIC50_list[i], hill_list[i]), "-")
        plt.savefig(os.path.join('Plot', 'CRC_'+inhibitor_name))

        df.to_csv(os.path.join(OUTDIR, 'Parameters', inhibitor_name+".csv"), index=False)
        plt.figure()
        sns.kdeplot(data=df[df.hill>0], x='pIC50', shade=True, alpha=0.1);
        plt.savefig(os.path.join(OUTDIR, 'Plot', inhibitor_name))

    df_inhibitor = df[['logK_I_D', 'logK_I_DI', 'logK_S_DI', 'pIC50', 'hill', 'pIC90']]
    df_inhibitor = df_inhibitor[df_inhibitor.hill>0]
    
    return df_inhibitor


def _pIC_hill_one_inhibitor(inhibitor, mcmc_dir, logDtot, logStot, logItot, 
                            measure='mean', logK_dE_alpha=None, OUTDIR=None):
    """
    For one inhibitor, dimer-only pIC50s can be simulated given the specified values of 
    dimer/substrate concentrations and kinetic parameters from mcmc trace. 

    Parameters:
    ----------
    inhibitor       : name of inhibitor
    mcmc_dir        : str, directory of traces
    logDtot         : vector of dimer concentration
    logStot         : vector of substrate concentration
    logItot         : vector of inhibitor concentration
    measure         : statistical measure, can be 'mean' or 'median'
    logK_dE_alpha   : dict, information of fixed logK, dE and alpha
    ----------
    return mean/median of pIC50 and hill slope for each inhibitor
    """
    assert measure in ['mean', 'median'], print("Please check the statistical measure again.")
        
    df = table_pIC_hill_one_inhibitor(inhibitor, mcmc_dir, logDtot, logStot, logItot, 
                                      logK_dE_alpha, OUTDIR)
    if measure == 'mean':
        return np.mean(df.pIC50), np.std(df.pIC50), np.mean(df.hill), np.std(df.hill), np.mean(df.pIC90), np.std(df.pIC90), 
    else:
        return np.median(df.pIC50), np.std(df.pIC50), np.median(df.hill), np.std(df.hill), np.median(df.pIC90), np.std(df.pIC90), 


def table_pIC_hill_multi_inhibitor(inhibitor_list, mcmc_dir, logDtot, logStot, logItot, 
                                   measure='mean', logK_dE_alpha=None, OUTDIR=None):
    """
    For a set of inhibitors, dimer-only pIC50s can be simulated given the specified values of 
    dimer/substrate concentrations and kinetic parameters from mcmc trace. 

    Parameters:
    ----------
    inhibitor_list  : list of inhibitor
    mcmc_dir        : str, directory of traces
    logDtot         : vector of dimer concentration
    logStot         : vector of substrate concentration
    logItot         : vector of inhibitor concentration
    logK_dE_alpha   : dict, information of fixed logK, dE and alpha
    ----------
    return table of kinetic parameters, pIC50, and hill slope for the whole dataset of multiple inhibitors
    """
    if OUTDIR is not None: 
        if not os.path.exists(os.path.join(OUTDIR, 'Parameters')):
            os.makedirs(os.path.join(OUTDIR, 'Parameters'))
        if not os.path.exists(os.path.join(OUTDIR, 'Plot')):
            os.makedirs(os.path.join(OUTDIR, 'Plot'))

    f_table = _pIC_hill_one_inhibitor
    args = [mcmc_dir, logDtot, logStot, logItot, measure, logK_dE_alpha, OUTDIR]
    list_median_std = np.array(list(map(lambda i: f_table(i, *args), inhibitor_list)))
    
    pIC50_median = list_median_std.T[0]
    pIC50_std = list_median_std.T[1]
    hill_median = list_median_std.T[2]
    hill_std = list_median_std.T[3]
    pIC90_median = list_median_std.T[4]
    pIC90_std = list_median_std.T[5]

    table_mean = pd.DataFrame([inhibitor_list, pIC50_median, hill_median, pIC90_median], index=['ID', 'pIC50', 'hill', 'pIC90'])
    table_std = pd.DataFrame([inhibitor_list, pIC50_std, hill_std, pIC90_std], index=['ID', 'pIC50_std', 'hill_std', 'pIC90_std'])        

    table = pd.merge(table_mean.T, table_std.T, on='ID')

    return table


def f_pIC90(pIC50, hill):
    """
    Calculating pIC90 given pIC50 and hill slope
        logIC90 = np.log10(IC50) + 1/hill*np.log10(9)
        logIC90 = logIC50 + 1/hill*np.log10(9)
        -pIC90 = -pIC50 + 1/hill*np.log10(9)
        pIC90 = pIC50 - 1/hill*np.log10(9)
    """
    return pIC50 - 1/hill*np.log10(9)


def _pd_mean_std_pIC(df, name_pIC='pIC50', measure='mean'):
    """
    Given a dataframe of cellular IC50/hill slope values of multiple experiments, 
    return mean/std of pIC50 for each inhibitor.

    Parameters:
    ----------
    df          : input dataframe
    name_pIC    : name of column
    measure     : statistical measure, can be 'mean' or 'median'
    ----------
    """
    assert measure in ['mean', 'median'], print("Please check the statistical measure again.")
    
    ID = np.unique(df['ID'])
    mean = []
    std = []
    for _ID in ID:
        if measure == 'mean':
            mean.append(df[name_pIC][df.ID == _ID].mean())
        else:
            mean.append(df[name_pIC][df.ID == _ID].median())
        std.append(df[name_pIC][df.ID == _ID].std())
    
    return pd.DataFrame([ID, mean, std], index=['ID', name_pIC, name_pIC+'_std']).T


def _correct_ID(df, correct='add'):
    """
    Correct the ID from ASAP-0000153 to ASAP-0000153-001 or vice versa
    """
    assert correct in ['add', 'drop'], print("ID can only be correct by adding or dropping -001.")
    
    colnames = df.columns
    ID = np.unique(df['ID'])
    if correct == 'add':
        new_ID = [_ID+'-001' for _ID in df['ID']]
    else: 
        new_ID = [_ID[:12] for _ID in df['ID']]
    df_update = df.copy()
    df_update = df_update.drop('ID', axis=1)
    df_update.insert(0, 'ID', new_ID)
    
    return df_update
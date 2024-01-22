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

def Dimerization(logMtot, logKd):
    """
    Parameters:
    ----------
    logMtot : numpy array
        Log of the total protein concentation summed over bound and unbound species (units of log molar)
    logKd   : float
        Log of the dissociation constant of dimerization (units of log molar)
    ----------
    Return  : equilibrium concentrations of species for dimerization of protein
    """
    Mtot = jnp.exp(logMtot)
    Kd = jnp.exp(logKd)
    x = (4*Mtot + Kd - jnp.sqrt(Kd**2 + 8*Mtot*Kd))/8
    log_concs = {}
    log_concs['D'] = np.log(x)
    log_concs['M'] = np.log(Mtot - 2*x)

    return log_concs


@jax.jit
def DimerOnlyModel(logDtot, logStot, logItot,
                   logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI):
    """
    Compute equilibrium concentrations for a binding model in which a ligand and substrate
    competitively binds to a dimer, or dimer complexed with a ligand.

    Parameters
    ----------
    logDtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logK_S_D : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logK_I_D : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS: float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI: float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex

    All dissociation constants are in units of log molar
    """
    dtype = jnp.float64
    logDtot = logDtot.astype(dtype)  # promote to dtype
    logStot = logStot.astype(dtype)  # promote to dtype
    logItot = logItot.astype(dtype)  # promote to dtype

    species = ['D','I','S','DI','DII','DS','DSI','DSS']
    # relationships between dissociation constants due to loops
    logK_I_DS = logK_I_D + logK_S_DI - logK_S_D

    reactions = [{'D':1, 'S':1, 'DS': -1},
                 {'DS':1, 'S':1, 'DSS':-1},
                 {'D':1, 'I':1, 'DI':-1},
                 {'DI':1, 'I':1, 'DII':-1},
                 {'DS':1, 'I':1, 'DSI':-1},     # Substrate and inhibitor binding
                 {'DI':1, 'S':1, 'DSI':-1}
                 ]
    conservation_equations = [{'D':+2,'DI':+2,'DII':+2, 'DS':+2,'DSI': +2,'DSS':+2}, # Total protein
                              {'S':+1,'DS':+1,'DSI':+1,'DSS':+2}, # Total substrate
                              {'I':+1,'DI':+1,'DII':+2,'DSI':+1} # Total ligand
                             ]
    binding_model = ChemicalReactions(reactions, conservation_equations)
    f_log_c = vmap(lambda logD, logS, logI: binding_model.logceq(jnp.array([logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_I_DS, logK_S_DI]),
                                                                 jnp.array([logD, logS, logI])))
    log_c = f_log_c(logDtot, logStot, logItot).T
    sorted_species = sorted(['D','I','S','DI','DII','DS','DSI','DSS'])
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])
    return log_concs


@jax.jit
def ReactionRate_DimerOnly(logDtot, logStot, logItot,
                           logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI,
                           kcat_DS=0., kcat_DSI=1., kcat_DSS=1.):
"""
    Reaction Rate
      v = kcat_+DS*[DS] + kcat_DSI*[DSI] + kcat_DSS*[DSS]
    
    Parameters
    ----------
    logDtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot : numpy array
        Log of the total ligand concentation summed over bound and unbound species      
    logK_S_D : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logK_I_D : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS: float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI: float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    kcat_DS: float
        Rate constant of dimer-substrate complex
    kcat_DSI: float
        Rate constant of dimer-substrate-inhibitor complex
    kcat_DSS: float
        Rate constant of dimer-substrate-substrate complex
    
    All dissociation constants are in units of log molar 
    """
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.
    log_concs = DimerOnlyModel(logDtot, logStot, logItot,
                               logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI)
    v = kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v


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
        upper = [min_y + 0.25*range_y, max_y + 0.25*range_y, 20, 20]
        lower = [min_y - 0.25*range_y, max_y - 0.25*range_y, -20, -20]
    else:
        upper = [theta[0] + 0.25*range_y, theta[1] + 0.25*range_y, 20, 20]
        lower = [theta[0] - 0.25*range_y, theta[1] - 0.25*range_y, -20, -20]

    fit_f, var_matrix = curve_fit(f_curve_vec, xdata=np.array(data['x']), ydata=np.array(data['y']),
                                  absolute_sigma=True, p0=theta,
                                  bounds=(lower, upper))

    # Estimate ASE for theta
    y_hat = f_curve_vec(data['x'], *fit_f)
    sigma = np.sqrt(np.sum((y_hat - data['y'])**2)/(len(y_hat)-4))
    ASE = np.sqrt(np.diag(var_matrix))*sigma #unscale_SE*sigma

    mle = [fit_f, ASE, np.array([sigma**2])]

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
    return x50 and hill slope
    """
    assert len(x)==len(y), print("Vectors of concentration and data should have the same length.")
    min_y = min(y)
    max_y = max(y)
    range_y = max_y - min_y

    theta0 = [min_y, max_y, x[np.argmin(np.square(y-np.mean(y)))], 1.0]
    upper = [min_y + 0.25*range_y, max_y + 0.25*range_y, 20, 20]
    lower = [min_y - 0.25*range_y, max_y - 0.25*range_y, -20, -20]

    try:
        res, var_matrix = curve_fit(f_curve_vec, xdata=x, ydata=y, absolute_sigma=True, p0=theta0,
                                    bounds=(lower, upper))
        if res[3]>0:
            return res[2], res[3]
        else: 
            return 0, 0
    except: 
        return 0, 0


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
    dat.insert(3, 'logK_S_DS', logK_dE_alpha['logK_S_DS']*np.ones(len(dat)))
    dat.insert(3, 'logK_S_D', logK_dE_alpha['logK_S_D']*np.ones(len(dat)))
    dat.insert(3, 'logK_S_M', logK_dE_alpha['logK_S_M']*np.ones(len(dat)))
    dat.insert(3, 'logKd', logK_dE_alpha['logKd']*np.ones(len(dat)))

    dat.insert(2, 'kcat_DSS', logK_dE_alpha['kcat_DSS']*np.ones(len(dat)))
    dat.insert(2, 'kcat_DS', logK_dE_alpha['kcat_DS']*np.ones(len(dat)))
    dat.insert(2, 'kcat_MS', np.zeros(len(dat)))
    return dat


def _pIC50_hill(df, logDtot, logStot, logItot):
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
    return list of pIC50 and list of hill slope
    """
    pIC50_list = []
    hill_list = []

    f_v = vmap(lambda logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, kcat_DS, kcat_DSI, kcat_DSS: ReactionRate_DimerOnly(logDtot, logStot, logItot, logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, kcat_DS, kcat_DSI, kcat_DSS))
    v_sim = f_v(jnp.array(df.logK_S_D), jnp.array(df.logK_S_DS), jnp.array(df.logK_I_D), jnp.array(df.logK_I_DI), jnp.array(df.logK_S_DI), jnp.array(df.kcat_DS), jnp.array(df.kcat_DSI), jnp.array(df.kcat_DSS))

    v_min = [jnp.min(v) for v in v_sim]
    v_max = [jnp.max(v) for v in v_sim]

    f_scaling = vmap(lambda v, _min, _max: (v - _min)/(_max - _min)*100)
    x = np.log10(np.exp(logItot))
    ys = f_scaling(v_sim, np.array(v_min), np.array(v_max))

    f_theta = lambda y: f_parameter_estimation(x, np.array(y))
    thetas = np.array(list(map(f_theta, ys)))

    pIC50_list = -thetas.T[0]
    hill_list = thetas.T[1]

    return pIC50_list, hill_list


def _table_pIC50_hill(inhibitor_list, mcmc_dir, logDtot, logStot, logItot, logK_dE_alpha=None, OUTDIR=None):
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

    table_mean = pd.DataFrame()
    table_std = pd.DataFrame()
    
    for n, inhibitor in enumerate(inhibitor_list):

        inhibitor_dir = inhibitor[7:12]
        inhibitor_name = inhibitor[:12]

        if not os.path.isfile(os.path.join(mcmc_dir, inhibitor_dir, 'traces.pickle')):
            continue

        trace = pickle.load(open(os.path.join(mcmc_dir, inhibitor_dir, 'traces.pickle'), "rb"))
        data = az.InferenceData.to_dataframe(az.convert_to_inference_data(trace))

        nthin = int(len(data)/100)
        if logK_dE_alpha is not None:
            df = _adjust_trace(data.iloc[::nthin, :].copy(), logK_dE_alpha)
        else:
            df = data.iloc[::nthin, :].copy()

        pIC50_list, hill_list = _pIC50_hill(df, logDtot, logStot, logItot)

        df.insert(len(df.columns), 'pIC50', pIC50_list)
        df.insert(len(df.columns), 'hill', hill_list)
        
        if OUTDIR is not None:
            df.to_csv(os.path.join(OUTDIR, 'Parameters', inhibitor_name+".csv"), index=False)
            plt.figure()
            sns.kdeplot(data=df[df.hill>0], x='pIC50', shade=True, alpha=0.1);
            plt.savefig(os.path.join(OUTDIR, 'Plot', inhibitor_name))

        df_inhibitor = df[['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI', 'pIC50', 'hill']]
        df_inhibitor = df_inhibitor[df_inhibitor.hill>0]
        table_mean.insert(len(table_mean.columns), inhibitor_name, df_inhibitor.median())
        table_std.insert(len(table_std.columns), inhibitor_name, df_inhibitor.std())

    table_mean = table_mean.T
    table_std = table_std.T.rename(columns={'logK_I_M': 'logK_I_M_std', 'logK_I_D': 'logK_I_D_std', 
                                            'logK_I_DI': 'logK_I_DI_std', 'logK_S_DI': 'logK_S_DI_std', 
                                            'pIC50': 'pIC50_std', 'hill': 'hill_std'})
    table = pd.concat([table_mean, table_std], axis=1)
    return table
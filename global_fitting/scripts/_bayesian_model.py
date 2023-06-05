import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

from _kinetics import ReactionRate, MonomerConcentration, CatalyticEfficiency
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses
from _params_extraction import extract_logK, extract_kcat
from _bayesian_model_multi_enzymes import fitting_each_dataset


def prior_group_uniform(params_logK, params_kcat, logKd_min = -20, logKd_max = 0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    params_logK : dict of all dissociation constants
    params_kcat : dict of all kcat 
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    return list of prior distribution for kinetics parameters
    
    """
    params_logK_update = {}
    params_kcat_update = {}
    for name_logK in params_logK:
        if params_logK[name_logK] is None:
            name_logK_dist = uniform_prior(name_logK, logKd_min, logKd_max)
            params_logK_update[name_logK] = name_logK_dist
        else:
            params_logK_update[name_logK] = params_logK[name_logK]

    for name_kcat in params_kcat:
        if params_kcat[name_kcat] is None:
            name_kcat_dist = uniform_prior(name_kcat, kcat_min, kcat_max)
            params_kcat_update[name_kcat] = name_kcat_dist
        else:
            params_kcat_update[name_kcat] = params_kcat[name_kcat]
    
    return params_logK_update, params_kcat_update


def prior_group_informative(prior_information):
    """
    Parameters:
    ----------
    prior_information : list of dict to assign prior distribution for kinetics parameters
    Examples: 
        prior_information = []
        prior_information.append({'type':'logK', 'name': 'logKd', 'dist': 'normal', 'loc': 0, 'scale': 1})
        prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'dist': None, 'value': 0.})
        prior_information.append({'type':'kcat', 'name': 'kcat_DS', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})

        The function returns three variables:
            logK ~ N(0, 1)
            kcat_MS = 0
            kcat_DS ~ U(0, 1)
        These variables will be saved into two lists, which is params_logK or params_kcat
    ----------
    return two lists of prior distribution for kinetics parameters
    
    """

    params_logK = {}
    params_kcat = {}
    for prior in prior_information:
        if prior['type'] == 'logK':
            if prior['dist'] is None:
                params_logK[prior['name']] = prior['value']
            elif prior['dist'] == 'normal':
                params_logK[prior['name']] = normal_prior(prior['name'], prior['loc'], prior['scale'])
            elif prior['dist'] == 'uniform':
                params_logK[prior['name']] = uniform_prior(prior['name'], prior['lower'], prior['upper'])
        
        if prior['type'] == 'kcat':
            if prior['dist'] is None:
                params_kcat[prior['name']] = prior['value']
            elif prior['dist'] == 'normal':
                params_kcat[prior['name']] = normal_prior(prior['name'], prior['loc'], prior['scale'])
            elif prior['dist'] == 'uniform':
                params_kcat[prior['name']] = uniform_prior(prior['name'], prior['lower'], prior['upper'])
    
    return params_logK, params_kcat


def global_fitting_jit(data_rate, data_AUC = None, data_ice = None, 
                       init_params_logK = None, init_params_kcat = None,
                       logKd_min = -20, logKd_max = 0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK : dict of all dissociation constants
    params_kcat : dict of all kcat 
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each dataset
    
    """

    # Define priors
    if init_params_logK is None:
        init_params_logK = {}
        for i in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
            init_params_logK[i] = None
    if init_params_kcat is None: 
        init_params_kcat = {}
        for i in ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']:
            init_params_kcat[i] = None

    params_logK, params_kcat = prior_group_uniform(init_params_logK, init_params_kcat, logKd_min, logKd_max, kcat_min, kcat_max)
    [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK(params_logK)
    [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat(params_kcat)
    
    if data_rate is not None: 
        fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                             'rate', 'log_sigma_rate')
    
    if data_AUC is not None: 
        fitting_each_dataset('AUC', data_AUC, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI], 
                             'auc', 'log_sigma_auc')

    if data_ice is not None: 
        fitting_each_dataset('ICE', data_ice, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                             'ice', 'log_sigma_ice')


def global_fitting(experiments, init_params_logK = None, init_params_kcat = None,
                   logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK : dict of all dissociation constants
    params_kcat : dict of all kcat 
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each dataset
    
    """
    
    # Define priors
    if init_params_logK is None:
        init_params_logK = {}
        for i in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
            init_params_logK[i] = None
    if init_params_kcat is None: 
        init_params_kcat = {}
        for i in ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']:
            init_params_kcat[i] = None

    params_logK, params_kcat = prior_group_uniform(init_params_logK, init_params_kcat, logKd_min, logKd_max, kcat_min, kcat_max)
    [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK(params_logK)
    [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat(params_kcat)

    for (n, experiment) in enumerate(experiments):
        name_tech = experiment['type']

        if name_tech == 'kinetics':
            data_rate = [experiment['v'], experiment['logMtot'], experiment['logStot'], experiment['logItot']]
            fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                 f'rate:{n}', f'{name_tech}_log_sigma:{n}')

        if name_tech == 'AUC':
            data_AUC = [experiment['M'], experiment['logMtot'], experiment['logStot'], experiment['logItot']]
            fitting_each_dataset('AUC', data_AUC, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI], 
                                 f'auc:{n}', f'{name_tech}_log_sigma:{n}')
        
        if name_tech == 'catalytic_efficiency':
            data_ice = [experiment['Km_over_kcat'], experiment['logMtot'], experiment['logStot'], experiment['logItot']]
            fitting_each_dataset('ICE', data_ice, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                 f'ice:{n}', f'{name_tech}_log_sigma:{n}')


def global_fitting_informative(data_rate, data_AUC = None, data_ice = None, prior_inform=None, 
                               logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):  
    """
    Parameters:
    ----------
    experiments : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    prior_inform: information to assign prior distribution for kinetics parameters
    params_logK : dict of all dissociation constants
    params_kcat : dict of all kcat 
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each dataset

    """

    # Define priors
    if prior_inform is None:
        init_params_logK = {}
        for i in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
            init_params_logK[i] = None
        init_params_kcat = {}
        for i in ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']:
            init_params_kcat[i] = None
        params_logK, params_kcat = prior_group_uniform(init_params_logK, init_params_kcat, logKd_min, logKd_max, kcat_min, kcat_max)
    else: 
        params_logK, params_kcat = prior_group_informative(prior_inform)

    [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK(params_logK)
    [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat(params_kcat)

    if data_rate is not None: 
        fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                             'rate', 'log_sigma_rate')
    
    if data_AUC is not None: 
        fitting_each_dataset('AUC', data_AUC, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI], 
                             'auc', 'log_sigma_auc')

    if data_ice is not None: 
        fitting_each_dataset('ICE', data_ice, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                             'ice', 'log_sigma_ice')
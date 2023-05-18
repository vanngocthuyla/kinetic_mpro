import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
import numpy as np

from _kinetics import ReactionRate, MonomerConcentration, CatalyticEfficiency
from _bayesian_model import uniform_prior, normal_prior, logsigma_guesses


def extract_logK_n_idx(params_logK, idx):
    """
    Parameters:
    ----------
    params_logK : dict of all dissociation constants
    idx         : index of enzyme
    ----------
    convert dictionary of dissociation constants to an array of values depending on the index of enzyme
    """
    # Dimerization
    logKd = params_logK[f'logKd:{idx}'] 
    # Binding Substrate
    logK_S_M = params_logK['logK_S_M']
    logK_S_D = params_logK['logK_S_D']
    logK_S_DS = params_logK['logK_S_DS']
    # Binding Inhibitor
    logK_I_M = params_logK['logK_I_M']
    logK_I_D = params_logK['logK_I_D']
    logK_I_DI = params_logK['logK_I_DI']
    # Binding both substrate and inhititor
    logK_S_DI = params_logK['logK_S_DI']
    return [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI]


def extract_kcat_n_idx(params_kcat, idx):
    """
    Parameters:
    ----------
    params_kcat : dict of all kcats
    idx         : index of enzyme
    ----------
    convert dictionary of kcats to an array of values depending on the index of enzyme
    """
    kcat_MS = params_kcat[f'kcat_MS:{idx}']
    kcat_DS = params_kcat[f'kcat_DS:{idx}']
    kcat_DSI = params_kcat[f'kcat_DSI:{idx}']
    kcat_DSS = params_kcat[f'kcat_DSS:{idx}']
    return [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS]


def check_prior_group(prior_information, n_enzymes):
    """
    Parameters:
    ----------
    prior_information : list of dict to assign prior distribution for kinetics parameters
    n_enzymes         : number of enzymes
    
    Examples: 
        prior_information = []
        prior_information.append({'type':'logKd', 'name': 'logKd', 'dist': 'normal', 'loc': [0, 2], 'scale': [1, 3]})
        prior_information.append({'type':'logK', 'name': 'logK_S_D', 'dist': 'uniform', 'lower': -20, 'upper': 0})
        prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'dist': None, 'value': [1., 0.]})
    ----------
    The function returns:
        prior_information.append({'type':'logKd', 'name': 'logKd', 'dist': 'normal', 'loc': np.array([0, 2]), 'scale': np.array([1, 3])})
        prior_information.append({'type':'logK', 'name': 'logK_S_D', 'dist': 'uniform', 'lower': -20, 'upper': 0})
        prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'dist': None, 'value': np.array([1., 0.])})
    """
    prior_update = []
    for prior in prior_information: 
        name = prior['name']
        if prior['type'] in ['logKd', 'kcat']:
            if prior['dist'] == 'normal': 
                if type(prior['loc']) == float or type(prior['loc']) == int:
                    loc = np.repeat(prior['loc'], n_enzymes)
                else:
                    loc = np.asarray(prior['loc'])
                if type(prior['scale']) == float or type(prior['scale']) == int:
                    scale = np.repeat(prior['scale'], n_enzymes)
                else:
                    scale = np.asarray(prior['scale'])
                prior_update.append({'type': prior['type'], 'name': prior['name'], 
                                     'dist': prior['dist'], 'loc': loc, 'scale': scale})
            elif prior['dist'] == 'uniform':
                if type(prior['lower']) == float or type(prior['lower']) == int:
                    lower = np.repeat(prior['lower'], n_enzymes)
                else:
                    lower = np.asarray(prior['lower'])
                if type(prior['upper']) == float or type(prior['upper']) == int:
                    upper = np.repeat(prior['upper'], n_enzymes)
                else:
                    upper = np.asarray(prior['upper'])
                prior_update.append({'type': prior['type'], 'name': prior['name'], 
                                     'dist': prior['dist'], 'lower': lower, 'upper': upper})
            else: 
                if type(prior['value']) == float or type(prior['value']) == int:
                    values = np.repeat(prior['value'], n_enzymes)
                else:
                    values = np.asarray(prior['value'])
                prior_update.append({'type': prior['type'], 'name': prior['name'], 
                                     'dist': prior['dist'], 'value': values})
        else:
            prior_update.append(prior)
    return prior_update


def prior_group_multi_enzyme(prior_information, n_enzymes):
    """
    Parameters:
    ----------
    prior_information : list of dict to assign prior distribution for kinetics parameters
    n_enzymes         : number of enzymes
    
    Examples: 
        prior_information = []
        prior_information.append({'type':'logKd', 'name': 'logKd', 'dist': 'normal', 'loc': np.array([0, 2]), 'scale': np.array([1, 3])})
        prior_information.append({'type':'logK', 'name': 'logK_S_D', 'dist': 'uniform', 'lower': -20, 'upper': 0})
        prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'dist': None, 'value': np.array([1., 0.])})

        The function returns three variables:
            logK:0 ~ N(0, 1)
            logK:1 ~ N(2, 3)
            logK_S_D ~ U(-20, 0)
            kcat_MS:0 = 1.
            kcat_MS:1 = 0.
        These variables will be saved into two lists, which is params_logK or params_kcat
    ----------
    return two lists of prior distribution for kinetics parameters
    """
    params_logK = {}
    params_kcat = {}
    
    for prior in prior_information:
        name = prior['name']
        if prior['type'] == 'logKd':
            for n in range(n_enzymes):
                if prior['dist'] is None:
                    params_logK[f'{name}:{n}'] = prior['value'][n]
                else:
                    if prior['dist'] == 'normal':
                        params_logK[f'{name}:{n}'] = normal_prior(f'{name}:{n}', prior['loc'][n], prior['scale'][n])
                    elif prior['dist'] == 'uniform':
                        params_logK[f'{name}:{n}'] = uniform_prior(f'{name}:{n}', prior['lower'][n], prior['upper'][n])
        
        if prior['type'] == 'logK':
            if prior['dist'] is None:
                params_logK[name] = prior['value']
            elif prior['dist'] == 'normal':
                params_logK[name] = normal_prior(name, prior['loc'], prior['scale'])
            elif prior['dist'] == 'uniform':
                params_logK[name] = uniform_prior(name, prior['lower'], prior['upper'])

        if prior['type'] == 'kcat':
            for n in range(n_enzymes):
                if prior['dist'] is None:
                    params_kcat[f'{name}:{n}'] = prior['value'][n]
                elif prior['dist'] == 'normal':
                    params_kcat[f'{name}:{n}'] = normal_prior(f'{name}:{n}', prior['loc'][n], prior['scale'][n])
                elif prior['dist'] == 'uniform':
                    params_kcat[f'{name}:{n}'] = uniform_prior(f'{name}:{n}', prior['lower'][n], prior['upper'][n])
    
    return params_logK, params_kcat


def define_uniform_prior_group(logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Return list of dict to assign prior distribution for kinetics parameters
    """
    prior_infor = []
    prior_infor.append({'type':'logKd', 'name': 'logKd', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_S_M', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_S_D', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_S_DS', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_I_M', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_I_D', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_I_DI', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_S_DI', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})

    prior_infor.append({'type':'kcat', 'name': 'kcat_MS', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
    prior_infor.append({'type':'kcat', 'name': 'kcat_DS', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
    prior_infor.append({'type':'kcat', 'name': 'kcat_DSS', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
    prior_infor.append({'type':'kcat', 'name': 'kcat_DSI', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
    return prior_infor


def global_fitting_multi_enzyme(experiments, prior_infor=None, 
                                logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each dataset (data_rate, data_AUC, data_ICE) contains response, logMtot, lotStot, logItot
    prior_infor : list of dict to assign prior distribution for kinetics parameters
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each enzyme
    """
    n_enzymes = len(experiments)

    # Define priors
    if prior_infor is None:
        init_prior_infor = define_uniform_prior_group(logKd_min, logKd_max, kcat_min, kcat_max)
        prior_infor = check_prior_group(init_prior_infor, n_enzymes)
    params_logK, params_kcat = prior_group_multi_enzyme(prior_infor, n_enzymes)

    for n, expt in enumerate(experiments):
        [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK_n_idx(params_logK, n)
        [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat_n_idx(params_kcat, n)
    
        data_rate = expt['kinetics']
        data_AUC = expt['AUC']
        data_ice = expt['ICE']

        if data_rate is not None: 
            [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data_rate
            rate_model = ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                      logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                      kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
            log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
            log_sigma_rate = uniform_prior(f'log_sigma_rate:{n}', lower=log_sigma_rate_min, upper=log_sigma_rate_max)
            sigma_rate = jnp.exp(log_sigma_rate)
            
            numpyro.sample(f'rate:{n}', dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)
        
        if data_AUC is not None: 
            [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data_AUC
            auc_model = MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot, 
                                            logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                            logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
            log_sigma_auc_min, log_sigma_auc_max = logsigma_guesses(auc) 
            log_sigma_auc = uniform_prior(f'log_sigma_auc:{n}', lower=log_sigma_auc_min, upper=log_sigma_auc_max)
            sigma_auc = jnp.exp(log_sigma_auc)
            
            numpyro.sample(f'auc:{n}', dist.Normal(loc=auc_model, scale=sigma_auc), obs=auc)

        if data_ice is not None: 
            [ice, ice_logMtot, ice_logStot, ice_logItot] = data_ice
            ice_model = 1./CatalyticEfficiency(ice_logMtot, ice_logItot, 
                                              logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                              logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                              kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
            log_sigma_ice_min, log_sigma_ice_max = logsigma_guesses(ice) 
            log_sigma_ice = uniform_prior(f'log_sigma_ice:{n}', lower=log_sigma_ice_min, upper=log_sigma_ice_max)
            sigma_ice = jnp.exp(log_sigma_ice)
            numpyro.sample(f'ice:{n}', dist.Normal(loc=ice_model, scale=sigma_ice), obs=ice)

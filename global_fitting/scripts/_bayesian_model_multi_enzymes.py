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
    if f'logKd:{idx}' in params_logK.keys(): logKd = params_logK[f'logKd:{idx}']
    else: logKd = params_logK['logKd']
    
    # Binding Substrate
    if f'logK_S_M:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_S_M:{idx}']
    else: logK_S_M = params_logK['logK_S_M']
    if f'logK_S_D:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_S_D:{idx}']
    else: logK_S_D = params_logK['logK_S_D']
    if f'logK_S_DS:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_S_DS:{idx}']
    else: logK_S_DS = params_logK['logK_S_DS']
    
    # Binding Inhibitor
    if f'logK_I_M:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_I_M:{idx}']
    else: logK_I_M = params_logK['logK_I_M']
    if f'logK_I_D:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_I_D:{idx}']
    else: logK_I_D = params_logK['logK_I_D']
    if f'logK_I_DI:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_I_DI:{idx}']
    else: logK_I_DI = params_logK['logK_I_DI']
    
    # Binding both substrate and inhititor
    if f'logK_S_DI:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_S_DI:{idx}']
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
    
    if f'kcat_MS:{idx}' in params_kcat.keys(): kcat_MS=params_kcat[f'kcat_MS:{idx}'] 
    else: kcat_MS=params_kcat['kcat_MS']
    if f'kcat_DS:{idx}' in params_kcat.keys(): kcat_DS = params_kcat[f'kcat_DS:{idx}'] 
    else: kcat_DS = params_kcat['kcat_DS']
    if f'kcat_DSI:{idx}' in params_kcat.keys(): kcat_DSI = params_kcat[f'kcat_DSI:{idx}'] 
    else: kcat_DSI = params_kcat['kcat_DSI']
    if f'kcat_DSS:{idx}' in params_kcat.keys(): kcat_DSS = params_kcat[f'kcat_DSS:{idx}'] 
    else: kcat_DSS = params_kcat['kcat_DSS']
    
    return [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS]


def check_prior_group(prior_information, n_enzymes):
    """
    Parameters:
    ----------
    prior_information : list of dict to assign prior distribution for kinetics parameters
    n_enzymes         : number of enzymes
    
    Examples: 
        prior_information = []
        prior_information.append({'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'normal', 'loc': [0, 2], 'scale': [1, 3]})
        prior_information.append({'type':'logK', 'name': 'logK_S_D', 'fit': 'global', 'dist': 'uniform', 'lower': -20, 'upper': 0})
        prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'fit': 'local', 'dist': None, 'value': [1., 0.]})
    ----------
    The function returns:
        prior_information.append({'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'normal', 'loc': np.array([0, 2]), 'scale': np.array([1, 3])})
        prior_information.append({'type':'logK', 'name': 'logK_S_D', 'fit': 'globals', 'dist': 'uniform', 'lower': -20, 'upper': 0})
        prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'fit': 'local', 'dist': None, 'value': np.array([1., 0.])})
    """
    prior_update = []
    for prior in prior_information:
        assert prior['type'] in ['logKd', 'logK', 'kcat'], "Paramter type should be logKd, logK or kcat."
        assert prior['fit'] in ['global', 'local'], "Please declare correctly if the parameter(s) would be fit local/global."
        assert prior['dist'] in ['normal', 'uniform', None], "The prior of parameters can be a value (None = no distribution) or can be normal/uniform distribution."
        
        name = prior['name']
        if prior['fit'] == 'local':
            if prior['dist'] == 'normal': 
                if type(prior['loc']) == float or type(prior['loc']) == int:
                    loc = np.repeat(prior['loc'], n_enzymes)
                else:
                    loc = np.asarray(prior['loc'])
                if type(prior['scale']) == float or type(prior['scale']) == int:
                    scale = np.repeat(prior['scale'], n_enzymes)
                else:
                    scale = np.asarray(prior['scale'])
                prior_update.append({'type': prior['type'], 'name': prior['name'], 'fit': prior['fit'],
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
                prior_update.append({'type': prior['type'], 'name': prior['name'], 'fit': prior['fit'],
                                     'dist': prior['dist'], 'lower': lower, 'upper': upper})
            else: 
                if type(prior['value']) == float or type(prior['value']) == int:
                    values = np.repeat(prior['value'], n_enzymes)
                else:
                    values = np.asarray(prior['value'])
                prior_update.append({'type': prior['type'], 'name': prior['name'], 'fit': prior['fit'],
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
        prior_information.append({'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'normal', 'loc': np.array([0, 2]), 'scale': np.array([1, 3])})
        prior_information.append({'type':'logK', 'name': 'logK_S_D', 'fit': 'global', 'dist': 'uniform', 'lower': -20, 'upper': 0})
        prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'fit': 'local', 'dist': None, 'value': np.array([1., 0.])})

        The function returns five parameters:
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
        assert prior['type'] in ['logKd', 'logK', 'kcat'], "Paramter type should be logKd, logK or kcat."
        assert prior['fit'] in ['global', 'local'], "Please declare correctly if the parameter(s) would be fit local/global."
        assert prior['dist'] in ['normal', 'uniform', None], "The prior of parameters can be a value (None = no distribution) or can be normal/uniform distribution."

        if prior['type'] in ['logKd', 'logK']:
            if prior['fit'] == 'local':
                for n in range(n_enzymes):
                    if prior['dist'] is None:
                        params_logK[f'{name}:{n}'] = prior['value'][n]
                    else:
                        if prior['dist'] == 'normal':
                            params_logK[f'{name}:{n}'] = normal_prior(f'{name}:{n}', prior['loc'][n], prior['scale'][n])
                        elif prior['dist'] == 'uniform':
                            params_logK[f'{name}:{n}'] = uniform_prior(f'{name}:{n}', prior['lower'][n], prior['upper'][n])
            elif prior['fit'] == 'global': 
                if prior['dist'] is None:
                    params_logK[name] = prior['value']
                elif prior['dist'] == 'normal':
                    params_logK[name] = normal_prior(name, prior['loc'], prior['scale'])
                elif prior['dist'] == 'uniform':
                    params_logK[name] = uniform_prior(name, prior['lower'], prior['upper'])

        if prior['type'] == 'kcat':
            if prior['fit'] == 'local':
                for n in range(n_enzymes):
                    if prior['dist'] is None:
                        params_kcat[f'{name}:{n}'] = prior['value'][n]
                    elif prior['dist'] == 'normal':
                        params_kcat[f'{name}:{n}'] = normal_prior(f'{name}:{n}', prior['loc'][n], prior['scale'][n])
                    elif prior['dist'] == 'uniform':
                        params_kcat[f'{name}:{n}'] = uniform_prior(f'{name}:{n}', prior['lower'][n], prior['upper'][n])
            elif prior['fit'] == 'global':
                if prior['dist'] is None:
                    params_kcat[name] = prior['value']
                if prior['dist'] == 'normal':
                    params_kcat[name] = normal_prior(name, prior['loc'], prior['scale'])
                elif prior['dist'] == 'uniform':
                    params_kcat[name] = uniform_prior(name, prior['lower'], prior['upper'])
    
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
    prior_infor.append({'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_S_M', 'fit': 'gloal', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_S_D', 'fit': 'gloal', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_S_DS', 'fit': 'gloal', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_I_M', 'fit': 'gloal', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_I_D', 'fit': 'gloal', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_I_DI', 'fit': 'gloal', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
    prior_infor.append({'type':'logK', 'name': 'logK_S_DI', 'fit': 'gloal', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})

    prior_infor.append({'type':'kcat', 'name': 'kcat_MS', 'fit': 'gloal', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
    prior_infor.append({'type':'kcat', 'name': 'kcat_DS', 'fit': 'gloal', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
    prior_infor.append({'type':'kcat', 'name': 'kcat_DSS', 'fit': 'gloal', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
    prior_infor.append({'type':'kcat', 'name': 'kcat_DSI', 'fit': 'gloal', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
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


def fitting_each_dataset(type_expt, data, params, name_response, name_log_sigma):
    """
    Parameters:
    ----------
    type_expt     : str, 'kinetics', 'AUC', or 'ICE'
    data          : list, each dataset contains response, logMtot, lotStot, logItot
    params        : list of kinetics parameters
    name_reponse  : str, name of posterior
    name_log_sigma: str, name of log_sigma for each dataset
    ----------
    Return likelihood from data and run the Bayesian model using given prior information of parameters
    """
    assert type_expt in ['kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, or ICE."

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        rate_model = ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, *params)
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
        log_sigma_rate = uniform_prior(name_log_sigma, lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma_rate = jnp.exp(log_sigma_rate)
        numpyro.sample(name_response, dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)

    if type_expt == 'AUC':
        [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data
        auc_model = MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot, *params)
        log_sigma_auc_min, log_sigma_auc_max = logsigma_guesses(auc) 
        log_sigma_auc = uniform_prior(name_log_sigma, lower=log_sigma_auc_min, upper=log_sigma_auc_max)
        sigma_auc = jnp.exp(log_sigma_auc)
        
        numpyro.sample(name_response, dist.Normal(loc=auc_model, scale=sigma_auc), obs=auc)

    if type_expt == 'ICE':
        [ice, ice_logMtot, ice_logStot, ice_logItot] = data
        ice_model = 1./CatalyticEfficiency(ice_logMtot, ice_logItot, *params)
        log_sigma_ice_min, log_sigma_ice_max = logsigma_guesses(ice) 
        log_sigma_ice = uniform_prior(name_log_sigma, lower=log_sigma_ice_min, upper=log_sigma_ice_max)
        sigma_ice = jnp.exp(log_sigma_ice)
        numpyro.sample(name_response, dist.Normal(loc=ice_model, scale=sigma_ice), obs=ice)


def global_fitting_multi_enzyme_multi_var(experiments, prior_infor=None, 
                                          logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).
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

    for idx, expt in enumerate(experiments):
        idx_expt = expt['index']
        [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK_n_idx(params_logK, idx)
        [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat_n_idx(params_kcat, idx)
    
        if type(expt['kinetics']) is dict: 
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None:
                    fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                         f'rate:{idx_expt}:{n}', f'log_sigma_rate:{idx_expt}:{n}')
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                     f'rate:{idx_expt}', f'log_sigma_rate:{idx_expt}')
        
        if type(expt['AUC']) is dict: 
            for n in range(len(expt['AUC'])):
                data_AUC = expt['AUC'][n]
                if data_AUC is not None: 
                    fitting_each_dataset('AUC', data_AUC, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI], 
                                        f'auc:{idx_expt}:{n}', f'log_sigma_auc:{idx_expt}:{n}')
        else:            
            data_AUC = expt['AUC']
            if data_AUC is not None: 
                fitting_each_dataset('AUC', data_AUC, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI], 
                                    f'auc:{idx_expt}', f'log_sigma_auc:{idx_expt}')

        if type(expt['ICE']) is dict:
            for n in range(len(expt['ICE'])):
                data_ice = expt['ICE'][n]
                if data_ice is not None: 
                    fitting_each_dataset('ICE', data_ice, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                        f'ice:{idx_expt}:{n}', f'log_sigma_ice:{idx_expt}:{n}')
        else:
            data_ice = expt['ICE']
            if data_ice is not None: 
                fitting_each_dataset('ICE', data_ice, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                    f'ice:{idx_expt}', f'log_sigma_ice:{idx_expt}')
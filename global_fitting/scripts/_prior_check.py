import jax
import jax.numpy as jnp
import numpy as np

from _prior_distribution import uniform_prior, normal_prior


def convert_prior_from_dict_to_list(prior, fit_E_S, fit_E_I):
    """
    ----------
    Parameters:
        prior  : dict of prior distribution
        fit_E_S: bool, fitting kinetics model of enzyme and substrate
        fit_E_I: bool, fitting kinetics model of enzyme and inhibitor
    Return     : a list of prior distribution for kinetics parameters
    ----------
    For example: 
        prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'local','dist': None, 'value': [-11, -14]}
        prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
    Return     :
        prior_infor = [{'type': 'logKd', 'name': 'logKd',  'fit': 'local', 'dist': None, 'value': [-11, -14]},
                       {'type': 'logK','name': 'logK_S_M','fit': 'global','dist': 'uniform','lower': -20.0,'upper': 0.0}]
    """
    prior_infor = []
    if 'logKd' in prior.keys():
        prior_infor.append(dict([(key, prior['logKd'][key]) for key in prior['logKd'].keys()]))

    for name in ['logK_S_M', 'logK_S_D', 'logK_S_DS']:
        if name in prior.keys():
            if fit_E_S:
                prior_infor.append(dict([(key, prior[name][key]) for key in prior[name].keys()]))
            else: 
                prior_infor.append({'type':'logK', 'name': name, 'fit': prior[name]['fit'], 'dist': None, 'value': 0})

    for name in ['logK_I_M', 'logK_I_D', 'logK_I_DI']:
        if name in prior.keys():
            if fit_E_I:
                prior_infor.append(dict([(key, prior[name][key]) for key in prior[name].keys()]))
            else: 
                prior_infor.append({'type':'logK', 'name': name, 'fit': prior[name]['fit'], 'dist': None, 'value': 0})

    if 'logK_S_DI' in prior.keys():
        if fit_E_S and fit_E_I:
            prior_infor.append(dict([(key, prior['logK_S_DI'][key]) for key in prior['logK_S_DI'].keys()]))
        else:
            prior_infor.append({'type':'logK', 'name': 'logK_S_DI', 'fit': prior[name]['fit'], 'dist': None, 'value': 0})
        
    for name in ['kcat_MS', 'kcat_DS', 'kcat_DSS']:
        if name in prior.keys():
            if fit_E_S:
                prior_infor.append(dict([(key, prior[name][key]) for key in prior[name].keys()]))
            else: 
                prior_infor.append({'type':'kcat', 'name': name, 'fit': prior[name]['fit'], 'dist': None, 'value': 0})

    if 'kcat_DSI' in prior.keys():
        if fit_E_S and fit_E_I:
            prior_infor.append(dict([(key, prior['kcat_DSI'][key]) for key in prior['kcat_DSI'].keys()]))
        else:
            prior_infor.append({'type':'kcat', 'name': 'kcat_DSI', 'fit': prior[name]['fit'], 'dist': None, 'value': 0})

    return prior_infor


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
        assert prior['type'] in ['logKd', 'logK', 'kcat'], "Parameter type should be logKd, logK or kcat."
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
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


def _check_prior_normal(prior, n_enzymes):
    """
    Parameters:
    ----------
    prior    : dict to assign prior distribution for kinetics parameters
    n_enzymes: number of enzymes
    
    Examples: 
        n_enzymes = 2
        prior_1 = {'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'normal', 'loc': [0, 2], 'scale': [1, 3]}
    ----------
    The function returns:
        prior_update_1 = {'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'normal', 'loc': np.array([0, 2]), 'scale': np.array([1, 3])}
    """
    if type(prior['loc']) == float or type(prior['loc']) == int:
        loc = np.repeat(prior['loc'], n_enzymes)
    else:
        loc = np.asarray(prior['loc'])
    if type(prior['scale']) == float or type(prior['scale']) == int:
        scale = np.repeat(prior['scale'], n_enzymes)
    else:
        scale = np.asarray(prior['scale'])
    prior_update = {'type': prior['type'], 'name': prior['name'], 'fit': prior['fit'],
                    'dist': prior['dist'], 'loc': loc, 'scale': scale}
    return prior_update


def _check_prior_uniform(prior, n_enzymes):
    """
    Parameters:
    ----------
    prior    : dict to assign prior distribution for kinetics parameters
    n_enzymes: number of enzymes
    
    Examples: 
        n_enzymes = 2
        prior_2 = {'type':'logK', 'name': 'logK_S_D', 'fit': 'global', 'dist': 'uniform', 'lower': -20, 'upper': 0}
    ----------
    The function returns:
        prior_update_2 = {'type':'logK', 'name': 'logK_S_D', 'fit': 'globals', 'dist': 'uniform', 'lower': -20, 'upper': 0}
    """
    if type(prior['lower']) == float or type(prior['lower']) == int:
        lower = np.repeat(prior['lower'], n_enzymes)
    else:
        lower = np.asarray(prior['lower'])
    if type(prior['upper']) == float or type(prior['upper']) == int:
        upper = np.repeat(prior['upper'], n_enzymes)
    else:
        upper = np.asarray(prior['upper'])
    prior_update = {'type': prior['type'], 'name': prior['name'], 'fit': prior['fit'],
                    'dist': prior['dist'], 'lower': lower, 'upper': upper}
    return prior_update


def _check_prior_fixed_value(prior, n_enzymes):
    """
    Parameters:
    ----------
    prior    : dict to assign prior distribution for kinetics parameters
    n_enzymes: number of enzymes
    
    Examples: 
        n_enzymes = 2
        prior_3 = {'type':'kcat', 'name': 'kcat_MS', 'fit': 'local', 'dist': None, 'value': [1., 0.]}
    ----------
    The function returns:
        prior_update_1 = {'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'normal', 'loc': np.array([0, 2]), 'scale': np.array([1, 3])}
        prior_update_2 = {'type':'logK', 'name': 'logK_S_D', 'fit': 'globals', 'dist': 'uniform', 'lower': -20, 'upper': 0}
        prior_update_3 = {'type':'kcat', 'name': 'kcat_MS', 'fit': 'local', 'dist': None, 'value': np.array([1., 0.])}
    """
    if type(prior['value']) == float or type(prior['value']) == int or prior['value'] is None:
        values = np.repeat(prior['value'], n_enzymes)
    else:
        values = np.asarray(prior['value'])
    prior_update = {'type': prior['type'], 'name': prior['name'], 'fit': prior['fit'],
                    'dist': prior['dist'], 'value': values}
    return prior_update


def _check_prior_one_dist(prior, n_enzymes):
    """
    Parameters:
    ----------
    prior    : dict to assign prior distribution for kinetics parameters
    n_enzymes: number of enzymes
    
    Examples: 
        n_enzymes = 2
        prior_1 = {'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'normal', 'loc': [0, 2], 'scale': [1, 3]}
        prior_2 = {'type':'logK', 'name': 'logK_S_D', 'fit': 'global', 'dist': 'uniform', 'lower': -20, 'upper': 0}
        prior_3 = {'type':'kcat', 'name': 'kcat_MS', 'fit': 'local', 'dist': None, 'value': [1., 0.]}
    ----------
    The function returns:
        prior_update_1 = {'type':'logKd', 'name': 'logKd', 'fit': 'local', 'dist': 'normal', 'loc': np.array([0, 2]), 'scale': np.array([1, 3])}
        prior_update_2 = {'type':'logK', 'name': 'logK_S_D', 'fit': 'globals', 'dist': 'uniform', 'lower': -20, 'upper': 0}
        prior_update_3 = {'type':'kcat', 'name': 'kcat_MS', 'fit': 'local', 'dist': None, 'value': np.array([1., 0.])}
    """
    assert prior['dist'] in ['normal', 'uniform', None], "The prior of parameters can be a value (None = no distribution) or can be normal/uniform distribution."
      
    if prior['dist'] == 'normal': 
        prior_update = _check_prior_normal(prior, n_enzymes)
    elif prior['dist'] == 'uniform':
        prior_update = _check_prior_uniform(prior, n_enzymes)
    else: 
        prior_update = _check_prior_fixed_value(prior, n_enzymes)
    return prior_update


def _check_prior_update_by_index(param, index):
    
    assert len(param) == len(index), "The number of values assigned for paramters should be consistent with the number of experiments."

    index_update = index*1.0
    index_update[index_update == 0] = np.nan
    index_update
    param_update = []
    for n, value in enumerate(param):
        if np.isnan(index_update[n]): 
            param_update.append(np.nan)
        else:
            param_update.append(value)
    return np.array(param_update)


def _check_prior_multi_dist(prior, n_enzymes):
    """
    Parameters:
    ----------
    prior    : dict to assign prior distribution for kinetics parameters
    n_enzymes: number of enzymes
    
    """
    distribution = prior['dist']
    prior_update_multi_dist = {'type': prior['type'], 'name': prior['name'], 
                               'fit': prior['fit'], 'dist': prior['dist']}
    for dist in distribution:
        assert dist in ['normal', 'uniform', None], "The prior of parameters can be a value (None = no distribution) or can be normal/uniform distribution."
        index = np.asarray(distribution)==dist
        if dist == 'normal': 
            prior_update = _check_prior_normal(prior, n_enzymes)
            prior_update_multi_dist['loc'] = _check_prior_update_by_index(prior_update['loc'], index)
            prior_update_multi_dist['scale'] = _check_prior_update_by_index(prior_update['scale'], index)
        elif dist == 'uniform':
            prior_update = _check_prior_uniform(prior, n_enzymes)
            prior_update_multi_dist['lower'] = _check_prior_update_by_index(prior_update['lower'], index)
            prior_update_multi_dist['upper'] = _check_prior_update_by_index(prior_update['upper'], index)
        else: 
            prior_update = _check_prior_fixed_value(prior, n_enzymes)
            prior_update_multi_dist['value'] = _check_prior_update_by_index(prior_update['value'], index)
    return prior_update_multi_dist


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
        
        dist = prior['dist']
        if type(dist) == str or dist is None:
            if prior['fit'] == 'local':
                prior_update.append(_check_prior_one_dist(prior, n_enzymes))
            else:
                prior_update.append(prior)
        elif np.sum([(None in dist), ('uniform' in dist), ('normal' in dist)]) == 1:
            prior_2 = dict([(key, prior[key]) for key in prior.keys() if key!='dist'])
            prior_2['dist'] = dist[0]
            if prior_2['fit'] == 'local':
                prior_update.append(_check_prior_one_dist(prior_2, n_enzymes))
            else:
                prior_update.append(prior_2)
        else: 
            prior_update.append(_check_prior_multi_dist(prior, n_enzymes))
    return prior_update


def _prior_group_multi_enzyme(prior_information, n_enzymes, params_name, shared_params=None, 
                              set_K_I_M_equal_K_S_M=False, set_K_S_DI_equal_K_S_DS=False, 
                              set_kcat_DSS_equal_kcat_DSI=False):
    """
    Parameters:
    ----------
    prior_information : list of dict to assign prior distribution for kinetics parameters
    n_enzymes         : number of enzymes
    params_name       : 'logK' or 'kcat'

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
    ----------
    return a list of prior distribution given a type of kinetics parameters (logK or kcat)
    """
    params = {}
    for prior in prior_information:
        if prior['type'] in params_name:
            assert prior['type'] in ['logKd', 'logK', 'kcat'], "Paramter type should be logKd, logK or kcat."
            assert prior['fit'] in ['global', 'local'], "Please declare correctly if the parameter(s) would be fit local/global."

            name = prior['name']

            if set_K_I_M_equal_K_S_M and name=='logK_I_M':
                continue
            if set_K_S_DI_equal_K_S_DS and name=='logK_S_DI':
                continue
            if set_kcat_DSS_equal_kcat_DSI and name=='kcat_DSS':
                continue
            
            if prior['fit'] == 'local':
                for n in range(n_enzymes):
                    if shared_params is not None:
                        if name in shared_params.keys() and n == shared_params[name]['assigned_idx']:
                            continue
                            
                    if type(prior['dist']) == str or prior['dist'] is None:
                        dist = prior['dist']
                    else:
                        dist = prior['dist'][n]
                    if dist == 'normal':
                        params[f'{name}:{n}'] = normal_prior(f'{name}:{n}', prior['loc'][n], prior['scale'][n])
                    elif dist == 'uniform':
                        params[f'{name}:{n}'] = uniform_prior(f'{name}:{n}', prior['lower'][n], prior['upper'][n])
                    elif dist is None:
                        if prior['value'][n] is not None:
                            params[f'{name}:{n}'] = prior['value'][n]
                        else:
                            params[f'{name}:{n}'] = None

            elif prior['fit'] == 'global':
                if prior['dist'] == 'normal':
                    params[name] = normal_prior(name, prior['loc'], prior['scale'])
                elif prior['dist'] == 'uniform':
                    params[name] = uniform_prior(name, prior['lower'], prior['upper'])
                elif prior['dist'] is None:
                    if prior['value'] is not None:
                        params[name] = prior['value']
                    else:
                        params[name] = None
    return params


def prior_group_multi_enzyme(prior_information, n_enzymes, shared_params=None, set_K_I_M_equal_K_S_M=False, 
                             set_K_S_DI_equal_K_S_DS=False, set_kcat_DSS_equal_kcat_DSI=False):
    """
    Parameters:
    ----------
    prior_information : list of dict to assign prior distribution for kinetics parameters
    n_enzymes         : number of enzymes
    ----------
    return two lists of prior distribution for kinetics parameters
    """
    params_logK = _prior_group_multi_enzyme(prior_information, n_enzymes, ['logKd', 'logK'], shared_params, 
                                            set_K_I_M_equal_K_S_M, set_K_S_DI_equal_K_S_DS, False)
    params_kcat = _prior_group_multi_enzyme(prior_information, n_enzymes, ['kcat'], shared_params, 
                                            False, False, set_kcat_DSS_equal_kcat_DSI)

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
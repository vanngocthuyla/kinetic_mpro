import numpy as np


def extract_logK(params_logK):
    """
    Parameters:
    ----------
    params_logK : dict of all dissociation constants
    ----------
    convert dictionary of dissociation constants to an array of values
    """
    # Dimerization
    if 'logKd' in params_logK.keys(): logKd = params_logK['logKd']
    else: logKd = None
    
    # Binding Substrate
    if 'logK_S_M' in params_logK.keys(): logK_S_M = params_logK['logK_S_M']
    else: logK_S_M = None
    if 'logK_S_D' in params_logK.keys(): logK_S_D = params_logK['logK_S_D']
    else: logK_S_D = None
    if 'logK_S_DS' in params_logK.keys(): logK_S_DS = params_logK['logK_S_DS']
    else: logK_S_DS = None
    
    # Binding Inhibitor
    if 'logK_I_M' in params_logK.keys(): logK_I_M = params_logK['logK_I_M']
    else: logK_I_M = None
    if 'logK_I_D' in params_logK.keys(): logK_I_D = params_logK['logK_I_D']
    else: logK_I_D = None
    if 'logK_I_DI' in params_logK.keys(): logK_I_DI = params_logK['logK_I_DI']
    else: logK_I_DI = None
    
    # Binding both substrate and inhititor
    if 'logK_S_DI' in params_logK.keys(): logK_S_DI = params_logK['logK_S_DI']
    else: logK_S_DI = None
    
    return [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI]


def extract_kcat(params_kcat):
    """
    Parameters:
    ----------
    params_kcat : dict of all kcats
    ----------
    convert dictionary of kcats to an array of values
    """
    if 'kcat_MS' in params_kcat.keys(): kcat_MS = params_kcat['kcat_MS']
    else: kcat_MS = 0.
    if 'kcat_DS' in params_kcat.keys(): kcat_DS = params_kcat['kcat_DS']
    else: kcat_DS = 0.
    if 'kcat_DSI' in params_kcat.keys(): kcat_DSI = params_kcat['kcat_DSI']
    else: kcat_DSI = 0.
    if 'kcat_DSS' in params_kcat.keys(): kcat_DSS = params_kcat['kcat_DSS']
    else: kcat_DSS = 0.
    return [kcat_DS, kcat_DSI, kcat_DSS]


def _extract_param_n_idx(name, params_dict, idx, shared_params=None):
    """
    Parameters:
    ----------
    name          : name of parameters extracted
    params_dict   : dict of all dissociation constants
    idx           : index of enzyme
    shared_params : dict of information for shared parameters
    ----------
    extract a parameter given a dictionary of kinetic parameters

    """
    if f'{name}:{idx}' in params_dict.keys():  
        if shared_params is not None: 
            if name in shared_params.keys() and idx == shared_params[name]['assigned_idx']:
                idx_update = shared_params[name]['shared_idx']
                param = params_dict[f'{name}:{idx_update}']
                print(f'{name}:{idx}', "will not be used. Shared params:", f'{name}:{idx_update}')
        else: 
            param = params_dict[f'{name}:{idx}']
    elif name in params_dict.keys(): param = params_dict[name]
    else: param = None
    return param


def extract_logK_n_idx(params_logK, idx, shared_params=None):
    """
    Parameters:
    ----------
    params_logK   : dict of all dissociation constants
    idx           : index of enzyme
    shared_params : dict of information for shared parameters
    ----------
    convert dictionary of dissociation constants to an array of values depending on the index of enzyme

    If there is information of shared parameter, such as shared logKd between expt_1 and expt_2, given the 
    information in shared_params:
        shared_params = {}
        shared_params['logKd'] = {'assigned_idx': 2, 'shared_idx': 1}
    In this case, logKd:2 will not be used to fit and logKd:1 is used. 

    """
    # Dimerization
    logKd = _extract_param_n_idx('logKd', params_logK, idx, shared_params)

    # Binding Substrate
    logK_S_M = _extract_param_n_idx('logK_S_M', params_logK, idx, shared_params)
    logK_S_D = _extract_param_n_idx('logK_S_D', params_logK, idx, shared_params)
    logK_S_DS = _extract_param_n_idx('logK_S_DS', params_logK, idx, shared_params)

    # Binding Inhibitor
    logK_I_M = _extract_param_n_idx('logK_I_M', params_logK, idx, shared_params)
    logK_I_D = _extract_param_n_idx('logK_I_D', params_logK, idx, shared_params)
    logK_I_DI = _extract_param_n_idx('logK_I_DI', params_logK, idx, shared_params)

    # Binding both substrate and inhititor
    logK_S_DI = _extract_param_n_idx('logK_S_DI', params_logK, idx, shared_params)

    return [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI]


def extract_kcat_n_idx(params_kcat, idx, shared_params=None):
    """
    Parameters:
    ----------
    params_kcat   : dict of all kcats
    idx           : index of enzyme
    shared_params : dict of information for shared parameters
    ----------
    convert dictionary of kcats to an array of values depending on the index of enzyme
    """
    kcat_MS = _extract_param_n_idx('kcat_MS', params_kcat, idx, shared_params)
    kcat_DS = _extract_param_n_idx('kcat_DS', params_kcat, idx, shared_params)
    kcat_DSI = _extract_param_n_idx('kcat_DSI', params_kcat, idx, shared_params)
    kcat_DSS = _extract_param_n_idx('kcat_DSS', params_kcat, idx, shared_params)

    return [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS]


def _convert_str_to_array_float(string):
    
    string = string.replace('nan', 'None').replace('  ', ' ').replace(' ', ' ').replace('[ ', '').replace('[', '').replace('] ', '').replace(']', '')
    string = string.split(' ')
    array = np.array(string)
    for i in range(len(array)):
        if array[i] == 'None':
            array[i] = None
        else:
            array[i] = float(array[i])
    return array


def _convert_str_to_array_string(string):
    
    string = string.replace('nan', 'None').replace('[','').replace(']', '').replace('\'', '').replace('   ', '').replace('  ', '').replace(' ', '')
    array = np.array(string.split(','))
    for i in range(len(array)):
        if array[i] == 'None':
            array[i] = None
    return array


def _prior_group_name(prior_information, n_enzymes, params_name=None):
    """
    Parameters:
    ----------
    prior_information : list of dict to assign prior distribution for kinetics parameters
    n_enzymes         : number of enzymes
    params_name       : 'logK' or 'kcat'

    return a list of dictionary of information of prior distribution
    """
    params_dict = {}
    if params_name is None: params_name = ['logKd', 'logK', 'kcat']

    for prior in prior_information:
        if prior['type'] in params_name:
            assert prior['type'] in ['logKd', 'logK', 'kcat'], "Paramter type should be logKd, logK or kcat."
            assert prior['fit'] in ['global', 'local'], "Please declare correctly if the parameter(s) would be fit local/global."

            name = prior['name']
            if prior['fit'] == 'local':
                for n in range(n_enzymes):
                    n_dist = _convert_str_to_array_string(prior['dist'])
                    if isinstance(n_dist, np.ndarray) and len(n_dist)>1:
                        dist = n_dist[n]
                    else:
                        dist = n_dist

                    if dist == 'normal':
                        if type(prior['loc']) == str:
                            loc = _convert_str_to_array_float(prior['loc'])[n]
                        else:
                            loc = prior['loc']
                        if type(prior['scale']) == str:
                            scale = _convert_str_to_array_float(prior['scale'])[n]
                        else:
                            scale = prior['scale']
                        params_dict[f'{name}:{n}'] = {'dist': 'normal', 'loc': float(loc), 'scale': float(scale)}
                    elif dist == 'uniform':
                        if type(prior['lower']) == str:
                            lower = _convert_str_to_array_float(prior['lower'])[n]
                        else:
                            lower = prior['lower']
                        if type(prior['upper']) == str:
                            upper = _convert_str_to_array_float(prior['upper'])[n]
                        else:
                            upper = prior['upper']
                        params_dict[f'{name}:{n}'] = {'dist': 'uniform', 'lower': float(lower), 'upper': float(upper)}
                    else:
                        if prior['value'][n] is not None:
                            if type(prior['value']) == str:
                                value = _convert_str_to_array_float(prior['value'])[n]
                            else:
                                value = prior['value']
                            params_dict[f'{name}:{n}'] = {'dist': None, 'value': value}
                        else:
                            params_dict[f'{name}:{n}'] = None

            elif prior['fit'] == 'global':
                if prior['dist'] == 'normal':
                    params_dict[name] = {'dist': 'normal', 'loc': float(prior['loc']), 'scale': float(prior['scale'])}
                elif prior['dist'] == 'uniform':
                    params_dict[name] = {'dist': 'uniform', 'lower': float(prior['lower']), 'upper': float(prior['upper'])}
                elif prior['dist'] is None or np.isnan(prior['dist']):
                    if prior['value'] is not None:
                        params_dict[name] = {'dist': None, 'value': float(prior['value'])}
                    else:
                        params_dict[name] = None
    return params_dict
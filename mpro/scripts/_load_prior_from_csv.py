import numpy as np


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
                        elif isinstance(prior['loc'], np.ndarray):
                            loc = prior['loc'][n]
                        else:
                            loc = prior['loc']
                        if type(prior['scale']) == str:
                            scale = _convert_str_to_array_float(prior['scale'])[n]
                        elif isinstance(prior['scale'], np.ndarray):
                            scale = prior['scale'][n]
                        else:
                            scale = prior['scale']
                        params_dict[f'{name}:{n}'] = {'dist': 'normal', 'loc': float(loc), 'scale': float(scale)}
                    elif dist == 'uniform':
                        if type(prior['lower']) == str:
                            lower = _convert_str_to_array_float(prior['lower'])[n]
                        elif isinstance(prior['lower'], np.ndarray):
                            lower = prior['lower'][n]
                        else:
                            lower = prior['lower']
                        if type(prior['upper']) == str:
                            upper = _convert_str_to_array_float(prior['upper'])[n]
                        elif isinstance(prior['upper'], np.ndarray):
                            upper = prior['upper'][n]
                        else:
                            upper = prior['upper']
                        params_dict[f'{name}:{n}'] = {'dist': 'uniform', 'lower': float(lower), 'upper': float(upper)}
                    else:
                        if prior['value'][n] is not None:
                            if type(prior['value']) == str:
                                value = _convert_str_to_array_float(prior['value'])[n]
                            elif isinstance(prior['value'], np.ndarray):
                                value = prior['value'][n]
                            else:
                                value = prior['value']
                            params_dict[f'{name}:{n}'] = {'dist': None, 'value': float(value)}
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


def _convert_str_to_array_float(string):
    
    if isinstance(string, str):
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
    
    if isinstance(string, str):
        string = string.replace('nan', 'None').replace('[','').replace(']', '').replace('\'', '').replace('   ', '').replace('  ', '').replace(' ', '')
        string = string.split(',')
    array = np.array(string)
    for i in range(len(array)):
        if array[i] == 'None':
            array[i] = None
    return array
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

    # # Solubility of inhibitor
    # if 'logKsp' in params_logK.keys(): logKsp = params_logK['logKsp']
    # else: logKsp = None
    
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
        if shared_params is not None and name in shared_params.keys() and idx == shared_params[name]['assigned_idx']:
            idx_update = shared_params[name]['shared_idx']
            param = params_dict[f'{name}:{idx_update}']
            print(f'{name}:{idx}', "will not be used. Shared params:", f'{name}:{idx_update}')
        else:
            param = params_dict[f'{name}:{idx}']
    elif name in params_dict.keys(): param = params_dict[name]
    else: param = None
    return param


def extract_logK_n_idx(params_logK, idx, shared_params=None, 
                       set_K_I_M_equal_K_S_M=False, set_K_S_DI_equal_K_S_DS=False):
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
    if set_K_I_M_equal_K_S_M and logK_S_M is not None: 
        logK_I_M = logK_S_M
    else: 
        logK_I_M = _extract_param_n_idx('logK_I_M', params_logK, idx, shared_params)
    logK_I_D = _extract_param_n_idx('logK_I_D', params_logK, idx, shared_params)
    logK_I_DI = _extract_param_n_idx('logK_I_DI', params_logK, idx, shared_params)

    # Binding both substrate and inhititor
    if set_K_S_DI_equal_K_S_DS and logK_S_DS is not None: 
        logK_S_DI = logK_S_DS
    else: 
        logK_S_DI = _extract_param_n_idx('logK_S_DI', params_logK, idx, shared_params)

    # # Solubility of inhibitor
    # logKsp = _extract_param_n_idx('logKsp', params_logK, idx, shared_params)

    return [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI]


def extract_kcat_n_idx(params_kcat, idx, shared_params=None, set_kcat_DSS_equal_kcat_DS=False, 
                       set_kcat_DSI_equal_kcat_DS=False, set_kcat_DSI_equal_kcat_DSS=False):
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
    if set_kcat_DSS_equal_kcat_DS and kcat_DS is not None:
        kcat_DSS = kcat_DS
    else:
        kcat_DSS = _extract_param_n_idx('kcat_DSS', params_kcat, idx, shared_params)
    if set_kcat_DSI_equal_kcat_DS and kcat_DS is not None:
        kcat_DSI = kcat_DS
    elif set_kcat_DSI_equal_kcat_DSS and kcat_DSS is not None:
        kcat_DSI = kcat_DSS
    else:
        kcat_DSI = _extract_param_n_idx('kcat_DSI', params_kcat, idx, shared_params)

    return [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS]
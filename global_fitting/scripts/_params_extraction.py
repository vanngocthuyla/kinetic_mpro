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
    logKd = params_logK['logKd']
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


def extract_kcat(params_kcat):
    """
    Parameters:
    ----------
    params_kcat : dict of all kcats
    ----------
    convert dictionary of kcats to an array of values
    """
    kcat_MS = params_kcat['kcat_MS']
    kcat_DS = params_kcat['kcat_DS']
    kcat_DSI = params_kcat['kcat_DSI']
    kcat_DSS = params_kcat['kcat_DSS']
    return [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS]


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


def extract_logK_WT(params_logK):
    """
    Parameters:
    ----------
    params_logK : dict of all dissociation constants
    ----------
    convert dictionary of dissociation constants to an array of values
    """
    logK_S_D = params_logK['logK_S_D']
    logK_S_DS = params_logK['logK_S_DS']
    # Binding Inhibitor
    logK_I_D = params_logK['logK_I_D']
    logK_I_DI = params_logK['logK_I_DI']
    # Binding both substrate and inhititor
    logK_S_DI = params_logK['logK_S_DI']
    return [logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI]


def extract_kcat_WT(params_kcat):
    """
    Parameters:
    ----------
    params_kcat : dict of all kcats
    ----------
    convert dictionary of kcats to an array of values
    """
    kcat_DS = params_kcat['kcat_DS']
    kcat_DSI = params_kcat['kcat_DSI']
    kcat_DSS = params_kcat['kcat_DSS']
    return [kcat_DS, kcat_DSI, kcat_DSS]


def extract_logK_n_idx_WT(params_logK, idx):
    """
    Parameters:
    ----------
    params_logK : dict of all dissociation constants
    idx         : index of enzyme
    ----------
    convert dictionary of dissociation constants to an array of values depending on the index of enzyme
    """
    # Substrate Inhibitor
    if f'logK_S_D:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_S_D:{idx}']
    else: logK_S_D = params_logK['logK_S_D']
    if f'logK_S_DS:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_S_DS:{idx}']
    else: logK_S_DS = params_logK['logK_S_DS']
    # Binding Inhibitor
    if f'logK_I_D:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_I_D:{idx}']
    else: logK_I_D = params_logK['logK_I_D']
    if f'logK_I_DI:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_I_DI:{idx}']
    else: logK_I_DI = params_logK['logK_I_DI']
    # Binding both substrate and inhititor
    if f'logK_S_DI:{idx}' in params_logK.keys(): logKd = params_logK[f'logK_S_DI:{idx}']
    else: logK_S_DI = params_logK['logK_S_DI']
    return [logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI]


def extract_kcat_n_idx_WT(params_kcat, idx):
    """
    Parameters:
    ----------
    params_kcat : dict of all kcats
    idx         : index of enzyme
    ----------
    convert dictionary of kcats to an array of values depending on the index of enzyme
    """
    if f'kcat_DS:{idx}' in params_kcat.keys(): kcat_DS = params_kcat[f'kcat_DS:{idx}'] 
    else: kcat_DS = params_kcat['kcat_DS']
    if f'kcat_DSI:{idx}' in params_kcat.keys(): kcat_DSI = params_kcat[f'kcat_DSI:{idx}'] 
    else: kcat_DSI = params_kcat['kcat_DSI']
    if f'kcat_DSS:{idx}' in params_kcat.keys(): kcat_DSS = params_kcat[f'kcat_DSS:{idx}'] 
    else: kcat_DSS = params_kcat['kcat_DSS']
    return [kcat_DS, kcat_DSI, kcat_DSS]
import jax
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform

from _kinetics import ReactionRate
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses, lognormal_prior
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _prior_check import check_prior_group, prior_group_multi_enzyme, define_uniform_prior_group


def fitting_each_dataset(type_expt, data, params, alpha=None, alpha_min=0., alpha_max=2.,
                         Etot=None, log_sigma_rate=None, index=''):
    """
    Parameters:
    ----------
    type_expt       : str, 'kinetics', 'AUC', or 'ICE'
    data            : list, each dataset contains response, logMtot, lotStot, logItot
    params          : list of kinetics parameters
    alpha           : float, normalization factor
    alpha_min       : float, lower values of uniform distribution for prior of alpha
    alpha_max       : float, upper values of uniform distribution for prior of alpha
    Etot            : float, enzyme concentration in nM
    log_sigma_rate  : float, measurement error under log scale
    index           : str, index of fitting dataset
    ----------
    Return likelihood from data and run the Bayesian model using given prior information of parameters
    """
    assert type_expt in ['kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, or ICE."

    if type_expt == 'kinetics':
        [rate, logMtot, logStot, logItot] = data

        if Etot is None: logE = logMtot
        else: logE = jnp.log(Etot*1E-9)

        rate_model = ReactionRate(logE, logStot, logItot, *params)

        if alpha is None:
            alpha = uniform_prior(f'alpha:{index}', lower=alpha_min, upper=alpha_max)
        
        if log_sigma_rate is None:
            log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
            log_sigma_rate = uniform_prior(f'log_sigma:{index}', lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma = jnp.exp(log_sigma_rate)

        numpyro.sample(f'rate:{index}', dist.Normal(loc=rate_model*alpha, scale=sigma), obs=rate)


def _dE_priors(experiments, dE, prior_type='lognormal'):
    """
    Parameters:
    ----------
    experiments : list of dict, each contains one experiment.
        - One experiment may contain multiple experimental datasets, including data_rate, data_AUC, data_ICE,
        and other information like type of enzyme, index, list of plate for all datasets. 
        - Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        - Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).    
    dE          : upper bound for the uniform prior of enzyme uncertainty if uniform distrubtion
                  or uncertainty of enzyme concentration of lognormal prior
    ----------
    Return list of enzyme uncertainty
    """
    error_E_list = {}
    _all_logMtot = []
    for idx, expt in enumerate(experiments):
        if type(expt['kinetics']) is dict:
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                _all_logMtot.append(data_rate[1])
        else:
            data_rate = expt['kinetics']
            _all_logMtot.append(data_rate[1])
    _all_logMtot = np.unique(np.concatenate(_all_logMtot))

    for _logConc in _all_logMtot:
        name = str(round(np.exp(_logConc)*1E9))
        if f'dE:{name}' not in error_E_list.keys():
            if prior_type == 'lognormal':
                error_E_list[f'dE:{name}'] = lognormal_prior(f'dE:{name}', jnp.exp(_logConc)*1E9, dE*jnp.exp(_logConc)*1E9)
            else:
                error_E_list[f'dE:{name}'] = uniform_prior(f'dE:{name}', 0, 1.)
    return error_E_list


def _dE_find_prior(data, error_E_list, prior_type='lognormal'):
    """
    Parameters:
    ----------
    data          : list, each dataset contains response, logMtot, lotStot, logItot
    error_E_list  : list of percentage error for enzyme concentration
    ----------
    Return array of prior information for enzyme concentration
    """
    [rate, logMtot, logStot, logItot] = data
    error_E = []
    for _logConc in logMtot:
        name = str(round(np.exp(_logConc)*1E9))
        if f'dE:{name}' in error_E_list.keys():
            error_E.append(error_E_list[f'dE:{name}'])
    return jnp.array(error_E)


def _alpha_priors(experiments, lower=0., upper=2.):
    """
    Parameters:
    ----------
    experiments : list of dict, each contains one experiment.
        - One experiment may contain multiple experimental datasets, including data_rate, data_AUC, data_ICE,
        and other information like type of enzyme, index, list of plate for all datasets. 
        - Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        - Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).    
    lower       : float, lower values of uniform distribution for prior of alpha
    upper       : float, upper values of uniform distribution for prior of alpha

    ----------
    Return dict of alphas as a list of normalization factor for different plates for all experiments
    """
    plate_list = []
    for idx, expt in enumerate(experiments):
        if type(expt['kinetics']) is dict: 
            for plate in expt['plate']:
                if plate != 'ES':
                    plate_list.append(plate)
        else:
            plate = expt['plate']
            if plate != 'ES':
                plate_list.append(plate)
    _all_plates = np.unique(plate_list)

    alpha_list = {}
    alpha_list['alpha:ES'] = 1.
    for plate in _all_plates:       
        alpha_list[f'alpha:{plate}'] = uniform_prior(f'alpha:{plate}', lower=lower, upper=upper)
    return alpha_list


def _alpha_find_prior(experiment, alpha_list):
    """
    Parameters:
    ----------
    experiment  : dict of one experiment
        - One experiment may contain multiple experimental datasets, including data_rate, data_AUC, data_ICE,
        and other information like type of enzyme, index, list of plate for all datasets. 
        - Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        - Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).    
    alpha_list  : list of normalization factor for all plate across mutliple experiments
    ----------
    Return array of prior information for alpha
    """
    alphas = []
    for plate in experiment['plate']:
        if f'alpha:{plate}' in alpha_list.keys():
            alphas.append(alpha_list[f'alpha:{plate}'])
        elif plate == 'ES':
            alphas.append(1)
    return jnp.array(alphas)


def global_fitting(experiments, prior_infor=None,
                   logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1,
                   shared_params=None, multi_alpha=False, alpha_min=0., alpha_max=2.,
                   set_lognormal_dE=False, dE=0.1, log_sigmas=None, 
                   set_K_S_DS_equal_K_S_D=False, set_K_S_DI_equal_K_S_DS=False):
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
    shared_params   : dict of information for shared parameters
    multi_alpha     : boolean, normalization factor
                      If True, setting one alpha for each dataset.
                      If False, multiple datasets having the same plate share alpha
    alpha_min       : float, lower values of uniform distribution for prior of alpha
    alpha_max       : float, upper values of uniform distribution for prior of alpha
    set_lognormal_dE: boolean, using lognormal prior or uniform prior for enzyme concentration uncertainnty
    dE              : float, enzyme concentration uncertainty
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each enzyme
    """
    n_enzymes = len(experiments)

    # Define priors
    if prior_infor is None:
        init_prior_infor = define_uniform_prior_group(logKd_min, logKd_max, kcat_min, kcat_max)
        prior_infor = check_prior_group(init_prior_infor, n_enzymes)
    params_logK, params_kcat = prior_group_multi_enzyme(prior_infor, n_enzymes)

    if set_lognormal_dE and dE>0:
        E_list = _dE_priors(experiments, dE, 'lognormal')

    if not multi_alpha:
        alpha_list = _alpha_priors(experiments, lower=alpha_min, upper=alpha_max)

    for idx, expt in enumerate(experiments):
        try: idx_expt = expt['index']
        except: idx_expt = idx

        _params_logK = extract_logK_n_idx(params_logK, idx, shared_params, set_K_S_DS_equal_K_S_D=set_K_S_DS_equal_K_S_D, 
                                          set_K_S_DI_equal_K_S_DS=set_K_S_DI_equal_K_S_DS)
        _params_kcat = extract_kcat_n_idx(params_kcat, idx, shared_params)

        if type(expt['kinetics']) is dict:
            
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                
                if not multi_alpha:
                    plate = expt['plate'][n]
                    alpha = alpha_list[f'alpha:{plate}']
                else: alpha = None

                if set_lognormal_dE and dE>0:
                    Etot = _dE_find_prior(data_rate, E_list)
                else: 
                    Etot = None

                if data_rate is not None:
                    if log_sigmas is not None: 
                        log_sigma = log_sigmas[f'log_sigma:{idx_expt}:{n}']
                    else:
                        log_sigma = None
                    
                    fitting_each_dataset(type_expt='kinetics', data=data_rate, params=[*_params_logK, *_params_kcat],
                                         alpha=alpha, alpha_min=alpha_min, alpha_max=alpha_max, 
                                         Etot=Etot, log_sigma_rate=log_sigma, index=f'{idx_expt}:{n}')
        else:
            data_rate = expt['kinetics']
            
            if not multi_alpha:
                plate = expt['plate']
                alpha = alpha_list[f'alpha:{plate}']
            else: alpha = None

            if set_lognormal_dE and dE>0:
                Etot = _dE_find_prior(data_rate, E_list)
            else: 
                Etot = None

            if data_rate is not None:
                if log_sigmas is not None: 
                    log_sigma = log_sigmas[f'log_sigma:{idx_expt}']
                else:
                    log_sigma = None
                
                fitting_each_dataset(type_expt='kinetics', data=data_rate, params=[*_params_logK, *_params_kcat],
                                     alpha=alpha, alpha_min=alpha_min, alpha_max=alpha_max, 
                                     Etot=Etot, log_sigma_rate=log_sigma, index=f'{idx_expt}')


def EI_fitting(experiments, alpha_list, E_list, prior_infor=None,
               logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1,
               shared_params=None, multi_alpha=False,
               set_K_S_DS_equal_K_S_D=False, set_K_S_DI_equal_K_S_DS=False):
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
    shared_params : dict of information for shared parameters
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
        try: idx_expt = expt['index']
        except: idx_expt = idx

        _params_logK = extract_logK_n_idx(params_logK, idx, shared_params, set_K_S_DS_equal_K_S_D=set_K_S_DS_equal_K_S_D, 
                                          set_K_S_DI_equal_K_S_DS=set_K_S_DI_equal_K_S_DS)
        _params_kcat = extract_kcat_n_idx(params_kcat, idx, shared_params)

        if type(expt['kinetics']) is dict:
            
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]

                if len(E_list)>0: 
                    Etot = _dE_find_prior(data_rate, E_list)
                else: 
                    Etot = None
                
                if not multi_alpha:
                    plate = expt['plate'][n]
                    alpha = alpha_list[f'alpha:{plate}']
                else: 
                    alpha = alpha_list[f'alpha:{idx_expt}:{n}']
                
                if data_rate is not None:
                    fitting_each_dataset(type_expt='kinetics', data=data_rate, params=[*_params_logK, *_params_kcat],
                                         alpha=alpha, Etot=Etot, log_sigma_rate=None, index=f'{idx_expt}:{n}')
        else:
            data_rate = expt['kinetics']
            
            if not multi_alpha:
                plate = expt['plate']
                alpha = alpha_list[f'alpha:{plate}']
            else: alpha = alpha_list[f'alpha:{idx_expt}']

            if data_rate is not None:
                if len(E_list)>0: 
                    Etot = _dE_find_prior(data_rate, E_list)
                else: 
                    Etot = None
                
                fitting_each_dataset(type_expt='kinetics', data=data_rate, params=[*_params_logK, *_params_kcat],
                                     alpha=alpha, Etot=Etot, log_sigma_rate=None, index=f'{idx_expt}')


# def _extract_conc_percent_error(logConc, error):
#     """
#     Parameters
#     ----------
#     logConc     : numpy array, concenntration of a species
#     error       : float, the adjusted value of the highest concentration returned by the model

#     Return the array of adjusted concentration given the percentage of error
#     """
#     if error is not None:
#         return logConc+jnp.log(1-error) #percent error
#     else:
#         return logConc


# def _prior_conc_lognormal(logConc, error=0.1, name='error'):
#     """
#     Parameters:
#     ----------
#     logConc       : concentration of a species in natural log scale
#     error         : percentage of uncertainty for log normal distribution
#     name          : name of the prior
#     ----------
#     Return adjusted log concentration
#     """ 
#     if error != 0:
#         stated_value = jnp.exp(jnp.max(logConc))
#         uncertainty = stated_value*error
#         expt_value = lognormal_prior(name, stated_value, uncertainty)
#         ratio = expt_value/stated_value
#         return logConc + jnp.log(ratio)
#     else:
#         return logConc
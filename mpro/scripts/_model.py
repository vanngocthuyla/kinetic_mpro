import numpy as np

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform

from _kinetics import ReactionRate, MonomerConcentration, CatalyticEfficiency
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses, lognormal_prior
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _prior_check import prior_group_multi_enzyme


def _dE_priors(experiments, dE, prior_type=''):
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
    prior_type  : optional, this information is used to assign 'lognormal' or 'uniform' prior for enzyme concentration uncertainty
    ----------
    Return list of enzyme uncertainty
    """
    assert dE>=0 and dE<1, print("dE should be between 0 and 1.")

    error_E_list = {}
    _all_logMtot = []
    for idx, expt in enumerate(experiments):
        if 'CRC' in expt.keys():
            if type(expt['CRC']) is dict:
                for n in range(len(expt['CRC'])):
                    data_rate = expt['CRC'][n]
                    _all_logMtot.append(data_rate[1])
            else:
                data_rate = expt['CRC']
                if data_rate is not None:
                    _all_logMtot.append(data_rate[1])

    if len(_all_logMtot)>0:
        _all_logMtot = np.unique(np.concatenate(_all_logMtot))

        for _logConc in _all_logMtot:
            name = str(round(np.exp(_logConc)*1E9))
            if f'dE:{name}' not in error_E_list.keys():
                conc = jnp.exp(_logConc)*1E9
                if prior_type == 'lognormal':
                    error_E_list[f'dE:{name}'] = lognormal_prior(f'dE:{name}', conc, dE*conc)
                else:
                    error_E_list[f'dE:{name}'] = uniform_prior(f'dE:{name}', (1-dE)*conc, (1+dE)*conc)

    return error_E_list


def _dE_find_prior(data, error_E_list):
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
        if 'CRC' in expt.keys():
            if type(expt['CRC']) is dict and 'plate' in expt.keys():
                for plate in expt['plate']:
                    if plate is not None:
                        plate_list.append(plate)
            else:
                if 'plate' in expt.keys():
                    plate = expt['plate']
                    if plate is not None:
                        plate_list.append(plate)
    alpha_list = {}
    if len(plate_list)>0:
        _all_plates = np.unique(plate_list)

        for plate in _all_plates:
            alpha_list[f'alpha:{plate}'] = uniform_prior(f'alpha:{plate}', lower=lower, upper=upper)
    else:
        alpha_list = None
    return alpha_list


def _alpha_find_prior(plate, alpha_list):
    """
    Parameters:
    ----------
    plate       : information about plate of experiment
    alpha_list  : list of normalization factor for all plate across mutliple experiments
    ----------
    Return prior information for alpha
    """
    if plate is not None:
        if alpha_list is not None:
            alpha = alpha_list[f'alpha:{plate}']
        else:
            alpha = None
    else:
        alpha = 1
    return alpha


def fitting_each_dataset(type_expt, data, params, alpha=None, alpha_min=0., alpha_max=2.,
                         Etot=None, log_sigmas=None, index=''):
    """
    Parameters:
    ----------
    type_expt       : str, 'kinetics', 'AUC', 'ICE', or 'CRC'
    data            : list, each dataset contains response, logMtot, lotStot, logItot
    params          : list of kinetics parameters
    alpha           : float, normalization factor
    alpha_min       : float, lower values of uniform distribution for prior of alpha
    alpha_max       : float, upper values of uniform distribution for prior of alpha
    Etot            : float, enzyme concentration in nM
    log_sigma       : dict, measurement error of multiple experiments under log scale
    index           : str, index of fitting dataset
    ----------
    Return likelihood from data and run the Bayesian model using given prior information of parameters
    """
    assert type_expt in ['CRC', 'kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, ICE, or CRC."

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        rate_model = ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, *params)
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate)
        log_sigma_rate = uniform_prior(f'log_sigma_rate:{index}', lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma_rate = jnp.exp(log_sigma_rate)
        numpyro.sample(f'rate:{index}', dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)

    if type_expt == 'AUC':
        [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data
        auc_model = MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot, *params)
        log_sigma_auc_min, log_sigma_auc_max = logsigma_guesses(auc)
        log_sigma_auc = uniform_prior(f'log_sigma_AUC:{index}', lower=log_sigma_auc_min, upper=log_sigma_auc_max)
        sigma_auc = jnp.exp(log_sigma_auc)
        numpyro.sample(f'AUC:{index}', dist.Normal(loc=auc_model, scale=sigma_auc), obs=auc)

    if type_expt == 'ICE':
        [ice, ice_logMtot, ice_logStot, ice_logItot] = data
        ice_model = 1./CatalyticEfficiency(ice_logMtot, ice_logItot, *params)
        log_sigma_ice_min, log_sigma_ice_max = logsigma_guesses(ice)
        log_sigma_ice = uniform_prior(f'log_sigma_ICE:{index}', lower=log_sigma_ice_min, upper=log_sigma_ice_max)
        sigma_ice = jnp.exp(log_sigma_ice)
        numpyro.sample(f'ICE:{index}', dist.Normal(loc=ice_model, scale=sigma_ice), obs=ice)

    if type_expt == 'CRC':
        [crc, logMtot, logStot, logItot] = data

        if Etot is None:
            logE = jnp.array(logMtot)
        else:
            logE = jnp.log(Etot*1E-9)

        CRC_model = ReactionRate(logE, logStot, logItot, *params)

        if alpha is None:
            alpha = uniform_prior(f'alpha:{index}', lower=alpha_min, upper=alpha_max)

        if log_sigmas is not None and f'log_sigma_CRC:{index}' in log_sigmas.keys():
            log_sigma_crc = log_sigmas[f'log_sigma_CRC:{index}']
        else:
            log_sigma_crc_min, log_sigma_crc_max = logsigma_guesses(crc)
            log_sigma_crc = uniform_prior(f'log_sigma_CRC:{index}', lower=log_sigma_crc_min, upper=log_sigma_crc_max)

        sigma_CRC = jnp.exp(log_sigma_crc)
        numpyro.sample(f'CRC:{index}', dist.Normal(loc=CRC_model*alpha, scale=sigma_CRC), obs=crc)


def _fitting_each_expt(type_expt, expt, idx_expt, params):
    """
    Parameters:
    ----------
    type_expt       : str, 'kinetics', 'AUC', 'ICE'
    expt            : dict contains the information of all dataset within experiment
                      Each dataset contains response, logMtot, lotStot, logItot
    params          : list of kinetics parameters
    idx_expt        : str, index of each experiment
    ----------
    Run the Bayesian model for each experiment
    """
    if type(expt[type_expt]) is dict:
        for n in range(len(expt[type_expt])):
            data = expt[type_expt][n]
            if data is not None:
                fitting_each_dataset(type_expt=type_expt, data=data, params=params,
                                     index=f'{idx_expt}:{n}')
    else:
        data = expt[type_expt]
        if data is not None:
            fitting_each_dataset(type_expt=type_expt, data=data, params=params,
                                 index=f'{idx_expt}')


def global_fitting(experiments, prior_infor, shared_params, args):
    """
    Parameters:
    ----------
    experiments     : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    shared_params   : dict of information for shared parameters
    args            : class holding model arguments. For more information, check _define_model.py
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each enzyme
    """
    n_enzymes = len(experiments)

    params_logK, params_kcat = prior_group_multi_enzyme(prior_infor, n_enzymes, shared_params)

    # Define priors for normalization factor
    if not args.multi_alpha:
        alpha_list = _alpha_priors(experiments, lower=args.alpha_min, upper=args.alpha_max)
    else:
        alpha_list = None

    # Define priors for enzyme concentration uncertainty
    if args.set_lognormal_dE and args.dE>0:
        E_list = _dE_priors(experiments, args.dE, 'lognormal')
    elif args.dE>0:
        E_list = _dE_priors(experiments, args.dE)

    for idx, expt in enumerate(experiments):
        
        # Extract parameters by index
        try: idx_expt = expt['index']
        except: idx_expt = idx
        _params_logK = extract_logK_n_idx(params_logK, idx, shared_params, set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D,
                                          set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
        _params_kcat = extract_kcat_n_idx(params_kcat, idx, shared_params)

        # Fitting each dataset
        if 'kinetics' in expt.keys():
            _fitting_each_expt(type_expt='kinetics', expt=expt, idx_expt=idx_expt, params=[*_params_logK, *_params_kcat])

        if 'AUC' in expt.keys():
            _fitting_each_expt(type_expt='AUC', expt=expt, idx_expt=idx_expt, params=_params_logK)

        if 'ICE' in expt.keys():
            _fitting_each_expt(type_expt='ICE', expt=expt, idx_expt=idx_expt, params=[*_params_logK, *_params_kcat])
            
        if 'CRC' in expt.keys():
            if type(expt['CRC']) is dict:
                for n in range(len(expt['CRC'])):
                    data_rate = expt['CRC'][n]
                    plate = expt['plate'][n]
                    alpha = _alpha_find_prior(plate, alpha_list)

                    if args.dE>0: Etot = _dE_find_prior(data_rate, E_list)
                    else: Etot = None

                    if data_rate is not None:
                        fitting_each_dataset(type_expt='CRC', data=data_rate, params=[*_params_logK, *_params_kcat],
                                             alpha=alpha, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
                                             Etot=Etot, log_sigmas=args.log_sigmas, index=f'{idx_expt}:{n}')
            else:
                data_rate = expt['CRC']
                if data_rate is not None:
                    plate = expt['plate']
                    alpha = _alpha_find_prior(plate, alpha_list)

                    if args.dE>0: Etot = _dE_find_prior(data_rate, E_list)
                    else: Etot = None

                    fitting_each_dataset(type_expt='CRC', data=data_rate, params=[*_params_logK, *_params_kcat],
                                         alpha=alpha, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
                                         Etot=Etot, log_sigmas=args.log_sigmas, index=f'{idx_expt}')


def EI_fitting(experiments, prior_infor, shared_params, args):
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
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each enzyme
    """
    n_enzymes = len(experiments)
    E_list = args.E_list
    alpha_list = args.alpha_list
    
    params_logK, params_kcat = prior_group_multi_enzyme(prior_infor, n_enzymes, shared_params)

    # Define priors for normalization factor
    if not args.multi_alpha:
        alpha_list = _alpha_priors(experiments, lower=args.alpha_min, upper=args.alpha_max)
    else:
        alpha_list = None

    for idx, expt in enumerate(experiments):
        try: idx_expt = expt['index']
        except: idx_expt = idx

        _params_logK = extract_logK_n_idx(params_logK, idx, shared_params, set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, 
                                          set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
        _params_kcat = extract_kcat_n_idx(params_kcat, idx, shared_params)

        if type(expt['CRC']) is dict:
            
            for n in range(len(expt['CRC'])):
                data_rate = expt['CRC'][n]

                if len(E_list)>0: 
                    Etot = _dE_find_prior(data_rate, E_list)
                else: 
                    Etot = None
                
                if not args.multi_alpha:
                    plate = expt['plate'][n]
                    alpha = alpha_list[f'alpha:{plate}']
                else: 
                    alpha = alpha_list[f'alpha:{idx_expt}:{n}']
                
                if data_rate is not None:
                    fitting_each_dataset(type_expt='CRC', data=data_rate, params=[*_params_logK, *_params_kcat],
                                         alpha=alpha, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
                                         Etot=Etot, log_sigmas=None, index=f'{idx_expt}:{n}')
        else:
            data_rate = expt['CRC']
            
            if not multi_alpha:
                plate = expt['plate']
                alpha = alpha_list[f'alpha:{plate}']
            else: alpha = alpha_list[f'alpha:{idx_expt}']

            if data_rate is not None:
                if len(E_list)>0: 
                    Etot = _dE_find_prior(data_rate, E_list)
                else: 
                    Etot = None
                
                fitting_each_dataset(type_expt='CRC', data=data_rate, params=[*_params_logK, *_params_kcat],
                                     alpha=alpha, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
                                     Etot=Etot, log_sigmas=None, index=f'{idx_expt}')
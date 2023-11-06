import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
import numpy as np

from _kinetics_adjustable_constraints import Adjustable_ReactionRate, Adjustable_MonomerConcentration, Adjustable_CatalyticEfficiency
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _prior_check import check_prior_group, prior_group_multi_enzyme, define_uniform_prior_group


def _dE_priors(experiments, dE):
    """
    Parameters:
    ----------
    experiments : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).
    dE            : upper bound for the uniform prior of enzyme uncertainty
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
        if name not in error_E_list.keys():
            error_E_list[f'dE:{name}'] = uniform_prior(f'dE:{name}', 0, dE)
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


def _extract_conc_percent_error(logConc, error):
    """
    Parameters
    ----------
    logConc     : numpy array, concenntration of a species
    error       : float, the adjusted value of the highest concentration returned by the model

    Return the array of adjusted concentration given the percentage of error
    """
    if error is not None:
        return logConc+jnp.log(1-error) #percent error
    else:
        return logConc


def adjustable_global_fitting(experiments, prior_infor=None,
                              logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1,
                              shared_params=None):
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
        try:
            idx_expt = expt['index']
        except:
            idx_expt = idx

        _params_logK = extract_logK_n_idx(params_logK, idx, shared_params)
        _params_kcat = extract_kcat_n_idx(params_kcat, idx, shared_params)
        alpha = uniform_prior(f'alpha:{idx_expt}', lower=1, upper=2)

        if type(expt['kinetics']) is dict:
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None:
                    adjustable_fitting_each_dataset('kinetics', data_rate, [*_params_logK, *_params_kcat],
                                                    f'rate:{idx_expt}:{n}', f'log_sigma:{idx_expt}:{n}', alpha)
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                adjustable_fitting_each_dataset('kinetics', data_rate, [*_params_logK, *_params_kcat],
                                                f'rate:{idx_expt}', f'log_sigma:{idx_expt}', alpha)


def adjustable_fitting_each_dataset(type_expt, data, params, name_response, name_log_sigma, alpha):
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
        [rate, logMtot, logStot, logItot] = data
        rate_model = Adjustable_ReactionRate(logMtot, logStot, logItot, *params)
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
        log_sigma_rate = uniform_prior(name_log_sigma, lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma_rate = jnp.exp(log_sigma_rate)
        numpyro.sample(name_response, dist.Normal(loc=rate_model*alpha, scale=sigma_rate), obs=rate)
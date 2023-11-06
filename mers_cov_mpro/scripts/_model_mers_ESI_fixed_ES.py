# Fitting Bayesian model for Mpro given some constraints on parameters

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
import numpy as np

from _kinetics import ReactionRate
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses, lognormal_prior
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _prior_check import check_prior_group, prior_group_multi_enzyme, define_uniform_prior_group
from _model_mers import _dE_priors, _dE_find_prior, _extract_conc_percent_error


def global_fitting(experiments, prior_infor=None,
                   logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1,
                   shared_params=None, multi_alpha=False, set_error_E=False, dE=1,
                   multi_var=False):
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

    if set_error_E and dE>0:
        error_E_list = {}
        error_E_list['dE:100'] = 0.618
        error_E_list['dE:50'] = 0.314
        error_E_list['dE:25'] = 0.073

    alphas = jnp.array([0.727, 0.922, 0.831, 0.918])

    for idx, expt in enumerate(experiments):
        try: idx_expt = expt['index']
        except: idx_expt = idx

        _params_logK = extract_logK_n_idx(params_logK, idx, shared_params)
        _params_kcat = extract_kcat_n_idx(params_kcat, idx, shared_params)

        if type(expt['kinetics']) is dict:
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if set_error_E and dE>0:
                    percent_error_E = _dE_find_prior(data_rate, error_E_list)
                else:
                    percent_error_E = None
                alpha = alphas[n]
                if data_rate is not None:
                    fitting_each_dataset('kinetics', data_rate, [*_params_logK, *_params_kcat],
                                         alpha, percent_error_E, f'{idx_expt}:{n}')
        else:
            data_rate = expt['kinetics']
            if set_error_E and dE>0:
                percent_error_E = _dE_find_prior(data_rate, error_E_list)
            else:
                percent_error_E = None
            if data_rate is not None:
                fitting_each_dataset('kinetics', data_rate, [*_params_logK, *_params_kcat],
                                     alpha, percent_error_E, f'{idx_expt}')


def fitting_each_dataset(type_expt, data, params, alpha=None,
                         percent_error_E=None, index=''):
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

        if percent_error_E is None: logE = logMtot
        else: logE = _extract_conc_percent_error(logMtot, percent_error_E)

        rate_model = ReactionRate(logE, logStot, logItot, *params)

        if alpha is None:
            alpha = uniform_prior(f'alpha:{index}', lower=0, upper=2)
            
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
        log_sigma_rate = uniform_prior(f'log_sigma:{index}', lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma = jnp.exp(log_sigma_rate)

        numpyro.sample(f'rate:{index}', dist.Normal(loc=rate_model*alpha, scale=sigma), obs=rate)
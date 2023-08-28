## Fitting Bayesian model for Mpro given some constraints on parameters

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


def _sigma(response_model, alpha, sigma_model=0.):
    """
    Parameters:
    ----------
    response_model  : jax numpy array, data from the model
    alpha           : ratio for multiplicative noise of data given response
    sigma_model     : additive noise of data
    ----------
    Assuming that y_obs ~ N(y_model, sigma^2), this return function of sigma
    
    If multiplicative noise : sigma^2 = alpha*response_model
       or mixed noise       : sigma^2 = alpha*response_model+sigma_model**2
    """
    return jnp.sqrt(alpha)*response_model
    # return jnp.sqrt(alpha*(response_model**2)+sigma_model**2)


def adjustable_global_fitting(experiments, prior_infor=None,
                              logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1,
                              shared_params=None, set_K_I_M_equal_K_S_M=False,
                              set_K_S_DI_equal_K_S_DS=False, set_kcat_DSS_equal_kcat_DS=False, 
                              set_kcat_DSI_equal_kcat_DS=False, set_kcat_DSI_equal_kcat_DSS=False):
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
        # print(idx_expt)

        [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK_n_idx(params_logK, idx, shared_params,
                                                                                                              set_K_I_M_equal_K_S_M,
                                                                                                              set_K_S_DI_equal_K_S_DS)
        [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat_n_idx(params_kcat, idx, shared_params, set_kcat_DSS_equal_kcat_DS, 
                                                                    set_kcat_DSI_equal_kcat_DS, set_kcat_DSI_equal_kcat_DSS)
        alpha = uniform_prior(f'alpha:{idx_expt}', lower=0, upper=1)

        if type(expt['kinetics']) is dict:
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None:
                    adjustable_fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS],
                                                    f'rate:{idx_expt}:{n}', f'log_sigma_rate:{idx_expt}:{n}', alpha)
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                adjustable_fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS],
                                                f'rate:{idx_expt}', f'log_sigma_rate:{idx_expt}', alpha)


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
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        rate_model = Adjustable_ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, *params)
        # log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
        # log_sigma_rate = uniform_prior(name_log_sigma, lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        # sigma_rate = jnp.exp(log_sigma_rate)
        # sigma_rate = _sigma(rate_model, alpha, jnp.exp(log_sigma_rate))
        sigma_rate = _sigma(rate_model, alpha)
        numpyro.sample(name_response, dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)
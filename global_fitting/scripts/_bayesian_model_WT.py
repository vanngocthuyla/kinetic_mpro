import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

from _kinetics_WT import ReactionRate_WT
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses
from _params_extraction import extract_logK_n_idx_WT, extract_kcat_n_idx_WT
from _bayesian_model_multi_enzymes import prior_group_multi_enzyme


def fitting_each_dataset_WT(type_expt, data, params, name_response, name_log_sigma):
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
    assert type_expt in ['kinetics'], "Experiments type should be kinetics."

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        rate_model = ReactionRate_WT(kinetics_logMtot, kinetics_logStot, kinetics_logItot, *params)
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate)
        log_sigma_rate = uniform_prior(name_log_sigma, lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma_rate = jnp.exp(log_sigma_rate)
        numpyro.sample(name_response, dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)


def global_fitting_WT(experiments, prior_infor, logK_min=-20, logK_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each dataset (data_rate, data_AUC, data_ICE) contains response, logMtot, lotStot, logItot
        Note: for each data_rate, data_AUC, data_ICE, there may be one more datasets (to fit different variances).
    prior_infor : list of dict to assign prior distribution for kinetics parameters
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each enzyme
    """
    n_enzymes = len(experiments)
    params_logK, params_kcat = prior_group_multi_enzyme(prior_infor, n_enzymes)

    for idx, expt in enumerate(experiments):
        try:
            idx_expt = expt['index']
        except:
            idx_expt = idx
        [logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK_n_idx_WT(params_logK, idx)
        [kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat_n_idx_WT(params_kcat, idx)

        if type(expt['kinetics']) is dict: 
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None: 
                    fitting_each_dataset_WT('kinetics', data_rate, [logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, kcat_DS, kcat_DSI, kcat_DSS], 
                                            f'rate:{idx_expt}:{n}', f'log_sigma_rate:{idx_expt}:{n}')
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                fitting_each_dataset_WT('kinetics', data_rate, [logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, kcat_DS, kcat_DSI, kcat_DSS], 
                                        f'rate:{idx_expt}', f'log_sigma_rate:{idx_expt}')
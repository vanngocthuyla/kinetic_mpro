import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

from _kinetics_WT import ReactionRate_WT
from _bayesian_model_multi_enzymes import prior_group_multi_enzyme, fitting_each_dataset


def uniform_prior(name, lower, upper):
    """
    Parameters:
    ----------
    name: string, name of variable
    lower: float, lower value of uniform distribution
    upper: float, upper value of uniform distribution
    ----------
    return numpyro.Uniform
    """
    name = numpyro.sample(name, dist.Uniform(low=lower, high=upper))
    return name


def normal_prior(name, mu, sigma):
    """
    Parameters:
    ----------
    name : string, name of variable
    mu   : float, mean of distribution
    sigma: float, std of distribution
    ----------
    return numpyro.Normal
    """
    name = numpyro.sample(name, dist.Normal(loc=mu, scale=sigma))
    return name


def logsigma_guesses(response):
    """
    Parameters:
    ----------
    response: jnp.array, observed data of concentration-response dataset
    ----------
    return range of log of sigma
    """
    log_sigma_guess = jnp.log(response.std()) # jnp.log(response.std())
    log_sigma_min = log_sigma_guess - 10 #log_sigma_min.at[0].set(log_sigma_guess - 10)
    log_sigma_max = log_sigma_guess + 5 #log_sigma_max.at[0].set(log_sigma_guess + 5)
    return log_sigma_min, log_sigma_max


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
                    fitting_each_dataset('kinetics', data_rate, [logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, kcat_DS, kcat_DSI, kcat_DSS], 
                                         f'rate:{idx_expt}:{n}', f'log_sigma_rate:{idx_expt}:{n}')
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                fitting_each_dataset('kinetics', data_rate, [logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, kcat_DS, kcat_DSI, kcat_DSS], 
                                     f'rate:{idx_expt}', f'log_sigma_rate:{idx_expt}')
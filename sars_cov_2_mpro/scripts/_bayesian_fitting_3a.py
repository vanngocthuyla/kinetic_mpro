import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp

from _kinetics import ReactionRate
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses


def prior_logK(params_logK, logKd_min = -20, logKd_max = 0):

    params_logK_update = {}
    for name_logK in params_logK:
        if params_logK[name_logK] is None:
            name_logK_dist = uniform_prior(name_logK, logKd_min, logKd_max)
            params_logK_update[name_logK] = name_logK_dist
        else:
            params_logK_update[name_logK] = params_logK[name_logK]
    
    return params_logK_update


def fit_mut_3a_alone(data_rate, init_params_logK=None, 
                     logKd_min = -20, logKd_max = 0, kcat_min=0, kcat_max=1):

    if init_params_logK is None:
        init_params_logK = {}
        for i in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
            init_params_logK[i] = None
    
    params_logK = prior_logK(init_params_logK, logKd_min, logKd_max)
    [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK(params_logK)
    
    kcat_MS = uniform_prior('kcat_MS', kcat_min, kcat_max)
    kcat_DS = uniform_prior('kcat_DS', kcat_min, kcat_max)
    kcat_DSI = kcat_DS
    kcat_DSS = kcat_DS
    
    if data_rate is not None: 
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data_rate
        rate_model = ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                  logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                  kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate)
        log_sigma_rate = uniform_prior('log_sigma_rate', lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma_rate = jnp.exp(log_sigma_rate)
    
        numpyro.sample('rate', dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)
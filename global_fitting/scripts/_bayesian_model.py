import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

from _kinetics import ReactionRate, MonomerConcentration, CatalyticEfficiency
from _kinetics import analytical_ReactionRate, analytical_MonomerConcentration, analytical_CatalyticEfficiency


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


def prior_group(params_logK, params_kcat, 
                logKd_min = -20, logKd_max = 0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    params_logK : dict of all dissociation constants
    params_kcat : dict of all kcat 
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    return list of prior distribution for kinetics parameters
    """
    params_logK_update = {}
    params_kcat_update = {}
    for name_logK in params_logK:
        if params_logK[name_logK] is None:
            name_logK_dist = uniform_prior(name_logK, logKd_min, logKd_max)
            params_logK_update[name_logK] = name_logK_dist
        else:
            params_logK_update[name_logK] = params_logK[name_logK]

    for name_kcat in params_kcat:
        if params_kcat[name_kcat] is None:
            name_kcat_dist = uniform_prior(name_kcat, kcat_min, kcat_max)
            params_kcat_update[name_kcat] = name_kcat_dist
        else:
            params_kcat_update[name_kcat] = params_kcat[name_kcat]
    
    return params_logK_update, params_kcat_update


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


def global_fitting(experiments,
                   init_params_logK = None, init_params_kcat = None,
                   logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK : dict of all dissociation constants
    params_kcat : dict of all kcat 
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each dataset
    """
    
    # Define priors
    if init_params_logK is None:
        init_params_logK = {}
        for i in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
            init_params_logK[i] = None
    if init_params_kcat is None: 
        init_params_kcat = {}
        for i in ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']:
            init_params_kcat[i] = None

    params_logK, params_kcat = prior_group(init_params_logK, init_params_kcat, logKd_min, logKd_max, kcat_min, kcat_max)
    [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK(params_logK)
    [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat(params_kcat)

    for (n, experiment) in enumerate(experiments):
        name_tech = experiment['type']

        if name_tech == 'kinetics':
            rate = experiment['v']
            kinetics_logMtot = experiment['logMtot']
            kinetics_logStot = experiment['logStot']
            kinetics_logItot = experiment['logItot']
            rate_model = ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                      logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                      kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
            log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
            log_sigma_rate = uniform_prior(f'{name_tech}_log_sigma:{n}', lower=log_sigma_rate_min, upper=log_sigma_rate_max)
            sigma_rate = jnp.exp(log_sigma_rate)
        
            numpyro.sample(f'rate:{n}', dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)

        if name_tech == 'AUC':
            auc = experiment['M']
            AUC_logMtot = experiment['logMtot']
            AUC_logStot = experiment['logStot']
            AUC_logItot = experiment['logItot']
            auc_model = MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot, 
                                             logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                             logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
            log_sigma_auc_min, log_sigma_auc_max = logsigma_guesses(auc) 
            log_sigma_auc = uniform_prior(f'{name_tech}_log_sigma:{n}', lower=log_sigma_auc_min, upper=log_sigma_auc_max)
            sigma_auc = jnp.exp(log_sigma_auc)
            
            numpyro.sample(f'auc:{n}', dist.Normal(loc=auc_model, scale=sigma_auc), obs=auc)
        
        if name_tech == 'catalytic_efficiency':
            ice = experiment['Km_over_kcat']
            ice_logMtot = experiment['logMtot']
            ice_logStot = experiment['logStot']
            ice_logItot = experiment['logItot']
            ice_model = 1./CatalyticEfficiency(ice_logMtot, ice_logItot, 
                                               logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                               logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                               kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
            log_sigma_ice_min, log_sigma_ice_max = logsigma_guesses(ice) 
            log_sigma_ice = uniform_prior(f'{name_tech}_log_sigma:{n}', lower=log_sigma_ice_min, upper=log_sigma_ice_max)
            sigma_ice = jnp.exp(log_sigma_ice)
            numpyro.sample(f'ice:{n}', dist.Normal(loc=ice_model, scale=sigma_ice), obs=ice)


def analytical_global_fitting(experiments,
                              init_params_logK = None, init_params_kcat = None,
                              logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK : dict of all dissociation constants
    params_kcat : dict of all kcat 
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each dataset
    """
    
    # Define priors
    if init_params_logK is None:
        init_params_logK = {}
        for i in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
            init_params_logK[i] = None
    if init_params_kcat is None: 
        init_params_kcat = {}
        for i in ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']:
            init_params_kcat[i] = None

    params_logK, params_kcat = prior_group(init_params_logK, init_params_kcat, logKd_min, logKd_max, kcat_min, kcat_max)
    [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK(params_logK)
    [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat(params_kcat)

    for (n, experiment) in enumerate(experiments):
        name_tech = experiment['type']

        if name_tech == 'kinetics':
            rate = experiment['v']
            kinetics_logMtot = experiment['logMtot']
            kinetics_logStot = experiment['logStot']
            kinetics_logItot = experiment['logItot']
            rate_model = analytical_ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                 kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
            log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
            log_sigma_rate = uniform_prior(f'{name_tech}_log_sigma:{n}', lower=log_sigma_rate_min, upper=log_sigma_rate_max)
            sigma_rate = jnp.exp(log_sigma_rate)
        
            numpyro.sample(f'rate:{n}', dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)

        if name_tech == 'AUC':
            auc = experiment['M']
            AUC_logMtot = experiment['logMtot']
            AUC_logStot = experiment['logStot']
            AUC_logItot = experiment['logItot']
            auc_model = analytical_MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot, 
                                                        logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                        logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
            log_sigma_auc_min, log_sigma_auc_max = logsigma_guesses(auc) 
            log_sigma_auc = uniform_prior(f'{name_tech}_log_sigma:{n}', lower=log_sigma_auc_min, upper=log_sigma_auc_max)
            sigma_auc = jnp.exp(log_sigma_auc)
            
            numpyro.sample(f'auc:{n}', dist.Normal(loc=auc_model, scale=sigma_auc), obs=auc)
        
        if name_tech == 'catalytic_efficiency':
            ice = experiment['Km_over_kcat']
            ice_logMtot = experiment['logMtot']
            ice_logStot = experiment['logStot']
            ice_logItot = experiment['logItot']
            ice_model = 1./analytical_CatalyticEfficiency(ice_logMtot, ice_logItot, 
                                                          logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                          logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                                          kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
            log_sigma_ice_min, log_sigma_ice_max = logsigma_guesses(ice) 
            log_sigma_ice = uniform_prior(f'{name_tech}_log_sigma:{n}', lower=log_sigma_ice_min, upper=log_sigma_ice_max)
            sigma_ice = jnp.exp(log_sigma_ice)
            numpyro.sample(f'ice:{n}', dist.Normal(loc=ice_model, scale=sigma_ice), obs=ice)


def global_fitting_jit(data_rate, data_AUC = None, data_ice = None, 
                       init_params_logK = None, init_params_kcat = None,
                       logKd_min = -20, logKd_max = 0, kcat_min=0, kcat_max=1):
  
    """
    Parameters:
    ----------
    experiments : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK : dict of all dissociation constants
    params_kcat : dict of all kcat 
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each dataset
    """

    # Define priors
    if init_params_logK is None:
        init_params_logK = {}
        for i in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
            init_params_logK[i] = None
    if init_params_kcat is None: 
        init_params_kcat = {}
        for i in ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']:
            init_params_kcat[i] = None

    params_logK, params_kcat = prior_group(init_params_logK, init_params_kcat, logKd_min, logKd_max, kcat_min, kcat_max)
    [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK(params_logK)
    [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat(params_kcat)
    
    if data_rate is not None: 
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data_rate
        rate_model = ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                  logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                  kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
        log_sigma_rate = uniform_prior('log_sigma_rate', lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma_rate = jnp.exp(log_sigma_rate)
    
        numpyro.sample('rate', dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)
    
    if data_AUC is not None: 
        [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data_AUC
        auc_model = MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot, 
                                         logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                         logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
        log_sigma_auc_min, log_sigma_auc_max = logsigma_guesses(auc) 
        log_sigma_auc = uniform_prior('log_sigma_auc', lower=log_sigma_auc_min, upper=log_sigma_auc_max)
        sigma_auc = jnp.exp(log_sigma_auc)
        
        numpyro.sample('auc', dist.Normal(loc=auc_model, scale=sigma_auc), obs=auc)

    if data_ice is not None: 
        [ice, ice_logMtot, ice_logStot, ice_logItot] = data_ice
        ice_model = 1./CatalyticEfficiency(ice_logMtot, ice_logItot, 
                                           logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                           logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                           kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
        log_sigma_ice_min, log_sigma_ice_max = logsigma_guesses(ice) 
        log_sigma_ice = uniform_prior('log_sigma_ice', lower=log_sigma_ice_min, upper=log_sigma_ice_max)
        sigma_ice = jnp.exp(log_sigma_ice)
        numpyro.sample('ice', dist.Normal(loc=ice_model, scale=sigma_ice), obs=ice)
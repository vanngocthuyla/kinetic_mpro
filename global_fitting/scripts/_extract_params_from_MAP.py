import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import numpyro.distributions as dist

from _bayesian_model import logsigma_guesses
from _kinetics import ReactionRate, MonomerConcentration, CatalyticEfficiency

def _log_likelihood_normal(response_actual, response_model, sigma):
    """ 
    PDF of log likelihood of normal distribution
    
    Parameters
    ----------
    response_actual : jnp.array, response of data
    response_model  : jnp.array, predicted data
    sigma           : standard deviation
    ----------
    Return: 
        Sum of log PDF of response_actual given normal distribution N(response_model, sigma^2)
    """
    # zs = (response_model - response_actual)/sigma
    # norm_rv = stats.norm(loc=0, scale=1)
    # return np.sum(norm_rv.logpdf(zs))
    return jnp.sum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))


def _uniform_pdf(x, lower, upper):
    """ 
    PDF of uniform distribution
    
    Parameters
    ----------
    x               : float
    lower           : float, lower of uniform distribution
    upper           : float, upper of uniform distribution
    ----------
    Return: 
        PDF of x values given uniform distribution U(lower, upper)
    """
    # assert upper > lower, "upper must be greater than lower"
    # if (x < lower) or (x > upper):
    #     return 0.
    return 1./(upper - lower)


def _log_prior(mcmc_trace, experiments, params_logK_name=None, params_kcat_name=None, nsamples=None):
    """
    Sum of log prior of all parameters, assuming they follows uniform distribution

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK_name: list of all dissociation constant names
    params_kcat_name: list of all kcat names 
    nsamples        : int, number of samples to find MAP
    ----------
    Return: 
        An array which size equals to mcmc_trace[:samples], each position corresponds 
        to sum of log prior calculated by values of parameters from mcmc_trace
    """
    if params_logK_name is None: 
        params_logK_name = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
    if params_kcat_name is None: 
        params_kcat_name = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']

    if nsamples is None:
        nsamples = len(mcmc_trace[params_logK_name[0]])
    assert nsamples <= len(mcmc_trace[params_logK_name[0]]), "nsamples too big"

    log_priors = jnp.zeros(nsamples)
    f_log_prior_logK = vmap(lambda param: _uniform_pdf(param, -20, 0))
    for name in params_logK_name:
        param_trace = mcmc_trace[name][: nsamples]
        log_priors += jnp.log(f_log_prior_logK(param_trace))
    
    f_log_prior_kcat = vmap(lambda param: _uniform_pdf(param, 0, 1))
    for name in params_kcat_name:
        param_trace = mcmc_trace[name][: nsamples]
        log_priors += jnp.log(f_log_prior_kcat(param_trace))

    for (n, experiment) in enumerate(experiments):
        name_tech = experiment['type']

        if name_tech == 'kinetics':
            rate = experiment['v']
            log_sigma_min, log_sigma_max = logsigma_guesses(rate)

            f_log_prior_sigma = vmap(lambda sigma: _uniform_pdf(sigma, log_sigma_min, log_sigma_max))
            param_trace = mcmc_trace[f'{name_tech}_log_sigma:{n}'][: nsamples]
            log_priors += jnp.log(f_log_prior_sigma(param_trace))

        if name_tech == 'AUC':
            auc = experiment['M']
            log_sigma_min, log_sigma_max = logsigma_guesses(auc)

            f_log_prior_sigma = vmap(lambda sigma: _uniform_pdf(sigma, log_sigma_min, log_sigma_max))
            param_trace = mcmc_trace[f'{name_tech}_log_sigma:{n}'][: nsamples]
            log_priors += jnp.log(f_log_prior_sigma(param_trace))
        
        if name_tech == 'catalytic_efficiency':
            ice = experiment['Km_over_kcat']
            log_sigma_min, log_sigma_max = logsigma_guesses(ice)

            f_log_prior_sigma = vmap(lambda sigma: _uniform_pdf(sigma, log_sigma_min, log_sigma_max))
            param_trace = mcmc_trace[f'{name_tech}_log_sigma:{n}'][: nsamples]
            log_priors += jnp.log(f_log_prior_sigma(param_trace))
    
    return np.array(log_priors)


def _log_likelihood(mcmc_trace, experiments, params_logK_name=None, params_kcat_name=None, nsamples=None):
    """
    Sum of log likelihood of all parameters, assuming they follows normal distribution

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK_name: list of all dissociation constant names
    params_kcat_name: list of all kcat names 
    nsamples        : int, number of samples to find MAP
    ----------
    Return: 
        An array which size equals to mcmc_trace[:samples], each position corresponds 
        to sum of log likelihood calculated by values of parameters from mcmc_trace
    """
    if params_logK_name is None: 
        params_logK_name = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
    if params_kcat_name is None: 
        params_kcat_name = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']

    full_params_logK_name = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
    full_params_kcat_name = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']

    if nsamples is None:
        nsamples = len(mcmc_trace[params_name[0]])
    assert nsamples <= len(mcmc_trace[params_name[0]]), "nsamples too big"

    log_likelihoods = jnp.zeros(nsamples)

    for (n, experiment) in enumerate(experiments):
        print("Experiment", experiment['type'], n)
        name_tech = experiment['type']

        logK_trace = {}
        kcat_trace = {}
        for name in full_params_logK_name: 
            if name in params_logK_name: 
                logK_trace[name] = mcmc_trace[name][: nsamples]
            else:
                logK_trace[name] = jnp.zeros(nsamples)
        for name in full_params_kcat_name:
            if name in params_kcat_name:
                kcat_trace[name] = mcmc_trace[name][: nsamples]
            else:
                kcat_trace[name] = jnp.zeros(nsamples)

        if name_tech == 'kinetics':
            rate = experiment['v']
            kinetics_logMtot = experiment['logMtot']
            kinetics_logStot = experiment['logStot']
            kinetics_logItot = experiment['logItot']

            try: 
                log_sigma_trace = mcmc_trace[f'{name_tech}_log_sigma:{n}'][: nsamples]
            except:
                log_sigma_trace = mcmc_trace[experiment['log_sigma']][: nsamples]
            f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, log_sigma: _log_likelihood_normal(rate, 
                                                                                                                                                                                    ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, 
                                                                                                                                                                                                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                                 kcat_MS, kcat_DS, kcat_DSI, kcat_DSS), 
                                                                                                                                                                                    jnp.exp(log_sigma)) )
            log_likelihoods += f(logK_trace['logKd'], logK_trace['logK_S_M'], logK_trace['logK_S_D'], 
                                 logK_trace['logK_S_DS'], logK_trace['logK_I_M'], logK_trace['logK_I_D'], 
                                 logK_trace['logK_I_DI'], logK_trace['logK_S_DI'], kcat_trace['kcat_MS'], 
                                 kcat_trace['kcat_DS'], kcat_trace['kcat_DSI'], kcat_trace['kcat_DSS'], 
                                 log_sigma_trace)
            
        if name_tech == 'AUC':
            auc = experiment['M']
            AUC_logMtot = experiment['logMtot']
            AUC_logStot = experiment['logStot']
            AUC_logItot = experiment['logItot']
            try:
                log_sigma_trace = mcmc_trace[f'{name_tech}_log_sigma:{n}'][: nsamples]
            except:
                log_sigma_trace = mcmc_trace[experiment['log_sigma']][: nsamples]
            f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, log_sigma: _log_likelihood_normal(auc,
                                                                                                                                              MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot,
                                                                                                                                                                   logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI),
                                                                                                                                              jnp.exp(log_sigma)) )
            log_likelihoods += f(logK_trace['logKd'], logK_trace['logK_S_M'], logK_trace['logK_S_D'], 
                                 logK_trace['logK_S_DS'], logK_trace['logK_I_M'], logK_trace['logK_I_D'], 
                                 logK_trace['logK_I_DI'], logK_trace['logK_S_DI'],  
                                 log_sigma_trace)

        if name_tech == 'catalytic_efficiency':
            ice = experiment['Km_over_kcat']
            ice_logMtot = experiment['logMtot']
            ice_logStot = experiment['logStot']
            ice_logItot = experiment['logItot']
            try:
                log_sigma_trace = mcmc_trace[f'{name_tech}_log_sigma:{n}'][: nsamples]
            except:
                log_sigma_trace = mcmc_trace[experiment['log_sigma']][: nsamples]
            f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, log_sigma: _log_likelihood_normal(ice, 
                                                                                                                                                                                    1./CatalyticEfficiency(ice_logMtot, ice_logItot,
                                                                                                                                                                                                           logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                                           kcat_MS, kcat_DS, kcat_DSI, kcat_DSS), 
                                                                                                                                                                                    jnp.exp(log_sigma)) )
            log_likelihoods += f(logK_trace['logKd'], logK_trace['logK_S_M'], logK_trace['logK_S_D'], 
                                 logK_trace['logK_S_DS'], logK_trace['logK_I_M'], logK_trace['logK_I_D'], 
                                 logK_trace['logK_I_DI'], logK_trace['logK_S_DI'], kcat_trace['kcat_MS'], 
                                 kcat_trace['kcat_DS'], kcat_trace['kcat_DSI'], kcat_trace['kcat_DSS'], 
                                 log_sigma_trace)

    return np.array(log_likelihoods)


def _map_kinetics(mcmc_trace, experiments, params_logK_name=None, params_kcat_name=None, nsamples=None):
    """
    Finding MAP (maximum a posterior)

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK_name: list of all dissociation constant names
    params_kcat_name: list of all kcat names 
    nsamples        : int, number of samples to find MAP
    ----------
    Return:
        Values of parameters that maximize the posterior
    """
    if params_logK_name is None: 
        params_logK_name = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
    if params_kcat_name is None: 
        params_kcat_name = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']

    if nsamples is None:
        nsamples = len(mcmc_trace[params_logK_name[0]])
    assert nsamples <= len(mcmc_trace[params_logK_name[0]]), "nsamples too big"

    log_priors = _log_prior(mcmc_trace, experiments, params_logK_name, params_kcat_name, nsamples)
    log_likelihoods = _log_likelihood(mcmc_trace, experiments, params_logK_name, params_kcat_name, nsamples)
    log_probs = log_priors + log_likelihoods
    map_idx = np.argmax(log_probs)
    print("Map index: %d" % map_idx)

    map_params = {}
    for name in params_logK_name: 
        map_params[name] = mcmc_trace[name][map_idx]
    for name in params_kcat_name:
        map_params[name] = mcmc_trace[name][map_idx]

    return [map_params, map_idx]
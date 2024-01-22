import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import pandas as pd
import numpyro.distributions as dist

from _prior_distribution import logsigma_guesses
from _load_prior_csv import _prior_group_name


def _log_prior_sigma(mcmc_trace, data, sigma_name, nsamples):
    """
    Sum of log prior of all log_sigma, assuming they follows uniform distribution

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    response        : list of jnp.array, data from each experiment
    nsamples        : int, number of samples to find MAP
    ----------
    
    """ 
    [response, logMtot, logStot, logItot] = data
    log_sigma_min, log_sigma_max = logsigma_guesses(response)
    f_log_prior_sigma = vmap(lambda sigma: _uniform_pdf(sigma, log_sigma_min, log_sigma_max))
    param_trace = mcmc_trace[sigma_name][: nsamples]
    return jnp.log(f_log_prior_sigma(param_trace))


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
    return jnp.nansum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))


def _extract_logK_kcat(mcmc_trace, idx, nsamples, all_params_logK_name=None, all_params_kcat_name=None):
    """
    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    idx             : index of experiment
    ----------
    Return:
        mcmc_trace for experiment[idx]
    """
    params_name_logK = []
    params_name_kcat = []
    for name in mcmc_trace.keys():
        if name.startswith('logK'):
            params_name_logK.append(name)
        if name.startswith('kcat'):
            params_name_kcat.append(name)

    if all_params_logK_name is None:
        all_params_logK_name = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
    if all_params_kcat_name is None:
        all_params_kcat_name = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']

    trace_nth = {}
    in_axis_nth = []
    for name in all_params_logK_name:
        if f'{name}:{idx}' in params_name_logK:
            trace_nth[name] = mcmc_trace[f'{name}:{idx}'][: nsamples]
            in_axis_nth.append(0)
        elif name in params_name_logK:
            trace_nth[name] = mcmc_trace[name][: nsamples]
            in_axis_nth.append(0)
        else:
            trace_nth[name] = None
            in_axis_nth.append(None)

    for name in all_params_kcat_name:
        if f'{name}:{idx}' in params_name_kcat:
            trace_nth[name] = mcmc_trace[f'{name}:{idx}'][: nsamples]
            in_axis_nth.append(0)
        elif name in params_name_kcat:
            trace_nth[name] = mcmc_trace[name][: nsamples]
            in_axis_nth.append(0)
        else:
            trace_nth[name] = jnp.zeros(nsamples)
            in_axis_nth.append(0)

    return trace_nth, in_axis_nth


def _map_adjust_trace(mcmc_trace, experiments, prior_infor, set_K_I_M_equal_K_S_M=False,
                      set_K_S_DS_equal_K_S_D=False, set_K_S_DI_equal_K_S_DS=False,
                      set_kcat_DSS_equal_kcat_DS=False, set_kcat_DSI_equal_kcat_DS=False, 
                      set_kcat_DSI_equal_kcat_DSS=False):

    """
    Adjusting mcmc_trace based on constrains and prior information

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling trace (group_by_chain=False)
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    ----------
    Return          : adjusted mccm_trace
    """

    mcmc_trace_update = mcmc_trace.copy()
    n_enzymes = len(experiments)

    keys = list(mcmc_trace_update.keys())
    for prior in prior_infor:
        if prior['dist'] is None and not prior['name'] in keys:
            mcmc_trace_update[prior['name']] = jnp.repeat(prior['value'], len(mcmc_trace_update[keys[0]]))
    
    prior_infor_pd = pd.DataFrame(prior_infor)
    if set_K_I_M_equal_K_S_M:
        idx = np.where(prior_infor_pd.name=='logK_S_M')[0][0]
        if prior_infor[idx]['fit'] == 'global':
            mcmc_trace_update['logK_I_M'] = mcmc_trace_update['logK_S_M']
        elif prior_infor[idx]['fit'] == 'local':
            for n in range(n_enzymes):
                mcmc_trace_update[f'logK_I_M:{n}'] = mcmc_trace_update[f'logK_S_M:{n}']
    if set_K_S_DS_equal_K_S_D:
        idx = np.where(prior_infor_pd.name=='logK_S_D')[0][0]
        if prior_infor[idx]['fit'] == 'global':
            mcmc_trace_update['logK_S_DS'] = mcmc_trace_update['logK_S_D']
        elif prior_infor[idx]['fit'] == 'local':
            for n in range(n_enzymes):
                mcmc_trace_update[f'logK_S_DS:{n}'] = mcmc_trace_update[f'logK_S_D:{n}']
    if set_K_S_DS_equal_K_S_D and set_K_S_DI_equal_K_S_DS:
        idx = np.where(prior_infor_pd.name=='logK_S_D')[0][0]
        if prior_infor[idx]['fit'] == 'global':
            mcmc_trace_update['logK_S_DI'] = mcmc_trace_update['logK_S_D']
        elif prior_infor[idx]['fit'] == 'local':
            for n in range(n_enzymes):
                mcmc_trace_update[f'logK_S_DI:{n}'] = mcmc_trace_update[f'logK_S_D:{n}']
    elif set_K_S_DI_equal_K_S_DS:
        idx = np.where(prior_infor_pd.name=='logK_S_DS')[0][0]
        if prior_infor[idx]['fit'] == 'global':
            mcmc_trace_update['logK_S_DI'] = mcmc_trace_update['logK_S_DS']
        elif prior_infor[idx]['fit'] == 'local':
            for n in range(n_enzymes):
                mcmc_trace_update[f'logK_S_DI:{n}'] = mcmc_trace_update[f'logK_S_DS:{n}']
    if set_kcat_DSS_equal_kcat_DS:
        idx = np.where(prior_infor_pd.name=='kcat_DS')[0][0]
        if prior_infor[idx]['fit'] == 'global':
            mcmc_trace_update['kcat_DSS'] = mcmc_trace_update['kcat_DS']
        elif prior_infor[idx]['fit'] == 'local':
            for n in range(n_enzymes):
                mcmc_trace_update[f'kcat_DSS:{n}'] = mcmc_trace_update[f'kcat_DS:{n}']
    if set_kcat_DSI_equal_kcat_DS:
        idx = np.where(prior_infor_pd.name=='kcat_DS')[0][0]
        if prior_infor[idx]['fit'] == 'global':
            mcmc_trace_update['kcat_DSI'] = mcmc_trace_update['kcat_DS']
        elif prior_infor[idx]['fit'] == 'local':
            for n in range(n_enzymes):
                mcmc_trace_update[f'kcat_DSI:{n}'] = mcmc_trace_update[f'kcat_DS:{n}']
    elif set_kcat_DSI_equal_kcat_DSS:
        idx = np.where(prior_infor_pd.name=='kcat_DSS')[0][0]
        if prior_infor[idx]['fit'] == 'global':
            mcmc_trace_update['kcat_DSI'] = mcmc_trace_update['kcat_DSS']
        elif prior_infor[idx]['fit'] == 'local':
            for n in range(n_enzymes):
                mcmc_trace_update[f'kcat_DSI:{n}'] = mcmc_trace_update[f'kcat_DSS:{n}']
    
    if len(mcmc_trace_update.keys())>len(keys):
        print("Adjusted trace with keys: ", mcmc_trace_update.keys())

    return mcmc_trace_update


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


def _gaussian_pdf(x, mean, std):
    """ 
    PDF of gaussian distribution
    
    Parameters
    ----------
    x               : float
    mean            : float, mean/loc of gaussian distribution
    std             : float, standard deviation of gaussian distribution
    ----------
    Return: 
        PDF of x values given gaussian distribution N(loc, scale)
    """
    return jnp.exp(dist.Normal(loc=0, scale=1).log_prob((x-mean)/std))


def _lognormal_pdf(x, stated_center, uncertainty):
    """
    PDF of lognormal distribution
    Ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html

    Parameters
    ----------
    x               : float
    stated_center   : float, mean of normal distribution
    uncertainty     : float, scale of normal distribution
    ----------
    Return:
        PDF of x values given lognormal distribution from the normal(loc, scale)
    """

    # if x <= 0:
    #     return 0.
    # else:
    m = stated_center
    v = uncertainty**2

    mu = np.log(m / jnp.sqrt(1 + (v / (m ** 2))))
    sigma_2 = jnp.log(1 + (v / (m**2)))

    return 1 / x / jnp.sqrt(2 * jnp.pi * sigma_2) * jnp.exp(-0.5 / sigma_2 * (jnp.log(x) - mu)**2)


def _log_normal_likelihood(response_actual, response_model, sigma):
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
    return jnp.nansum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))
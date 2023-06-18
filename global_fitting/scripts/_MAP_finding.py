import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import numpyro.distributions as dist

from _prior_distribution import logsigma_guesses
from _kinetics_adjustable import Adjustable_ReactionRate, Adjustable_MonomerConcentration, Adjustable_CatalyticEfficiency
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


def _log_priors(mcmc_trace, experiments, prior_infor, nsamples=None):
    """
    Sum of log prior of all parameters, assuming they follows uniform distribution

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    params_logK_name: list of all dissociation constant names
    params_kcat_name: list of all kcat names 
    nsamples        : int, number of samples to find MAP
    ----------
    Return: 
        An array which size equals to mcmc_trace[:samples], each position corresponds 
        to sum of log prior calculated by values of parameters from mcmc_trace
    """ 
    params_name_logK = []
    params_name_kcat = []
    for name in mcmc_trace.keys():
        if name.startswith('logK'):
            params_name_logK.append(name)
        if name.startswith('kcat'):
            params_name_kcat.append(name)
    
    if nsamples is None:
        nsamples = len(mcmc_trace[params_name_logK[0]])
    assert nsamples <= len(mcmc_trace[params_name_logK[0]]), "nsamples too big"

    log_priors = jnp.zeros(nsamples)

    # log_prior of all logK and kcat
    infor = _prior_group_name(prior_infor, len(experiments), ['logKd', 'logK', 'kcat'])
    for name in params_name_logK+params_name_kcat:
        param_trace = mcmc_trace[name][: nsamples]
        param_infor = infor[name]
        if param_infor['dist'] == 'normal':
            f_prior_normal = vmap(lambda param: _gaussian_pdf(param, param_infor['loc'], param_infor['scale']))
            log_priors += jnp.log(f_prior_normal(param_trace))
        if param_infor['dist'] == 'uniform': 
            f_prior_uniform = vmap(lambda param: _uniform_pdf(param, param_infor['lower'], param_infor['upper']))
            log_priors += jnp.log(f_prior_uniform(param_trace))

    # log_prior of all sigma
    for idx, expt in enumerate(experiments):
        try: idx_expt = expt['index']
        except: idx_expt = idx

        if type(expt['kinetics']) is dict: 
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None:
                    log_priors += _log_prior_sigma(mcmc_trace, data_rate, f'log_sigma_rate:{idx_expt}:{n}', nsamples)
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                log_priors += _log_prior_sigma(mcmc_trace, data_rate, f'log_sigma_rate:{idx_expt}', nsamples)
        
        if type(expt['AUC']) is dict:
            for n in range(len(expt['AUC'])):
                data_AUC = expt['AUC'][n]
                if data_AUC is not None: 
                    log_priors += _log_prior_sigma(mcmc_trace, data_AUC, f'log_sigma_auc:{idx_expt}:{n}', nsamples)
        else:            
            data_AUC = expt['AUC']
            if data_AUC is not None: 
                log_priors += _log_prior_sigma(mcmc_trace, data_AUC, f'log_sigma_auc:{idx_expt}', nsamples)

        if type(expt['ICE']) is dict:
            for n in range(len(expt['ICE'])):
                data_ice = expt['ICE'][n]
                if data_ice is not None:
                    log_priors += _log_prior_sigma(mcmc_trace, data_ice, f'log_sigma_ice:{idx_expt}:{n}', nsamples)
        else:
            data_ice = expt['ICE']
            if data_ice is not None: 
                log_priors += _log_prior_sigma(mcmc_trace, data_ice, f'log_sigma_ice:{idx_expt}', nsamples)
    
    return np.array(log_priors)


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
    # return jnp.sum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))
    return jnp.nansum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))


def _log_likelihood_each_enzyme(type_expt, data, trace_logK, trace_kcat, trace_log_sigma, in_axes_nth, nsamples): 
    """
    Parameters:
    ----------
    type_expt     : str, 'kinetics', 'AUC', or 'ICE'
    data          : list, each dataset contains response, logMtot, lotStot, logItot
    trace_logK    : trace of all logK
    trace_logK    : trace of all kcat
    trace_sigma   : trace of log_sigma
    ----------
    Return log likelihood depending on the type of experiment and mcmc_trace
    """
    assert type_expt in ['kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, or ICE."
    log_likelihoods = jnp.zeros(nsamples)

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, log_sigma: _log_normal_likelihood(rate, 
                                                                                                                                                                                Adjustable_ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, 
                                                                                                                                                                                                        logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                                                                                                                                                                        logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                                        kcat_MS, kcat_DS, kcat_DSI, kcat_DSS),
                                                                                                                                                                                jnp.exp(log_sigma)),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'], 
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'], 
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], trace_kcat['kcat_MS'], 
                             trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'], 
                             trace_log_sigma)
    if type_expt == 'AUC':
        [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data

        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, log_sigma: _log_normal_likelihood(auc,
                                                                                                                                          Adjustable_MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot,
                                                                                                                                                                          logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                                                                                                                                          logK_I_M, logK_I_D, logK_I_DI, logK_S_DI),
                                                                                                                                          jnp.exp(log_sigma)),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'], 
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'], 
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], 
                             trace_log_sigma)
    
    if type_expt == 'ICE':
        [ice, ice_logMtot, ice_logStot, ice_logItot] = data
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, log_sigma: _log_normal_likelihood(ice, 
                                                                                                                                                                                1./Adjustable_CatalyticEfficiency(ice_logMtot, ice_logItot,
                                                                                                                                                                                                                  logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                                                                                                                                                                                  logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                                                  kcat_MS, kcat_DS, kcat_DSI, kcat_DSS), 
                                                                                                                                                                                jnp.exp(log_sigma)),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'], 
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'], 
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], trace_kcat['kcat_MS'], 
                             trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'], 
                             trace_log_sigma)
    return log_likelihoods


def _mcmc_trace_each_enzyme(mcmc_trace, idx, nsamples):
    """
    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
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
    
    full_params_logK_name = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
    full_params_kcat_name = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']

    trace_nth = {}
    in_axis_nth = []
    for name in full_params_logK_name:
        if f'{name}:{idx}' in params_name_logK:
            trace_nth[name] = mcmc_trace[f'{name}:{idx}'][: nsamples]
            in_axis_nth.append(0)
        elif name in params_name_logK:
            trace_nth[name] = mcmc_trace[name][: nsamples]
            in_axis_nth.append(0)
        else: 
            trace_nth[name] = None
            in_axis_nth.append(None)

    for name in full_params_kcat_name:
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


def _log_likelihoods(mcmc_trace, experiments, nsamples=None):
    """
    Sum of log likelihood of all parameters given their distribution information in params_dist

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    ----------
    Return: 
        Sum of log likelihood given experiments and mcmc_trace
    """    
    params_name_logK = []
    params_name_kcat = []
    for name in mcmc_trace.keys():
        if name.startswith('logK'):
            params_name_logK.append(name)
        if name.startswith('kcat'):
            params_name_kcat.append(name)

    if nsamples is None:
        nsamples = len(mcmc_trace[params_name_logK[0]])
    assert nsamples <= len(mcmc_trace[params_name_logK[0]]), "nsamples too big"

    log_likelihoods = jnp.zeros(nsamples)

    for idx, expt in enumerate(experiments):
        try: idx_expt = expt['index']
        except: idx_expt = idx

        trace_nth, in_axis_nth = _mcmc_trace_each_enzyme(mcmc_trace, idx, nsamples)

        in_axis_nth.append(0)
        if type(expt['kinetics']) is dict: 
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None:
                    print("Kinetics experiment", idx_expt)
                    trace_log_sigma = mcmc_trace[f'log_sigma_rate:{idx_expt}:{n}'][: nsamples]
                    log_likelihoods += _log_likelihood_each_enzyme('kinetics', data_rate, trace_nth, trace_nth, trace_log_sigma, in_axis_nth, nsamples)
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                print("Kinetics experiment", idx_expt)
                trace_log_sigma = mcmc_trace[f'log_sigma_rate:{idx_expt}'][: nsamples]
                log_likelihoods += _log_likelihood_each_enzyme('kinetics', data_rate, trace_nth, trace_nth, trace_log_sigma, in_axis_nth, nsamples)
        
        if type(expt['AUC']) is dict: 
            for n in range(len(expt['AUC'])):
                data_AUC = expt['AUC'][n]
                if data_AUC is not None: 
                    print("AUC experiment", idx_expt)
                    trace_log_sigma = mcmc_trace[f'log_sigma_auc:{idx_expt}:{n}'][: nsamples]
                    log_likelihoods += _log_likelihood_each_enzyme('AUC', data_AUC, trace_nth, trace_nth, trace_log_sigma, in_axis_nth[:9], nsamples)
        else:
            data_AUC = expt['AUC']
            if data_AUC is not None:
                print("AUC experiment", idx_expt) 
                trace_log_sigma = mcmc_trace[f'log_sigma_auc:{idx_expt}'][: nsamples]
                log_likelihoods += _log_likelihood_each_enzyme('AUC', data_AUC, trace_nth, trace_nth, trace_log_sigma, in_axis_nth[:9], nsamples)

        if type(expt['ICE']) is dict:
            for n in range(len(expt['ICE'])):
                data_ice = expt['ICE'][n]
                if data_ice is not None: 
                    print("ICE experiment", idx_expt)
                    trace_log_sigma = mcmc_trace[f'log_sigma_ice:{idx_expt}:{n}'][: nsamples]
                    log_likelihoods += _log_likelihood_each_enzyme('ICE', data_ice, trace_nth, trace_nth, trace_log_sigma, in_axis_nth, nsamples)
        else:
            data_ice = expt['ICE']
            if data_ice is not None:
                print("ICE experiment", idx_expt) 
                trace_log_sigma = mcmc_trace[f'log_sigma_ice:{idx_expt}'][: nsamples]
                log_likelihoods += _log_likelihood_each_enzyme('ICE', data_ice, trace_nth, trace_nth, trace_log_sigma, in_axis_nth, nsamples)

    return np.array(log_likelihoods)


def map_finding(mcmc_trace, experiments, prior_infor, nsamples=None):
    """
    Evaluate probability of a parameter set using posterior distribution
    Finding MAP (maximum a posterior) given prior distributions of parameters information

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling trace (group_by_chain=False)
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    ----------
    Return          : values of parameters that maximize the posterior
    """
    params_name_logK = []
    params_name_kcat = []
    for name in mcmc_trace.keys():
        if name.startswith('logK'):
            params_name_logK.append(name)
        if name.startswith('kcat'):
            params_name_kcat.append(name)

    if nsamples is None:
        nsamples = len(mcmc_trace[params_name_logK[0]])
    assert nsamples <= len(mcmc_trace[params_name_logK[0]]), "nsamples too big"       

    print("Calculing log of priors.")
    log_priors = _log_priors(mcmc_trace, experiments, prior_infor, nsamples)
    print("Calculing log likelihoods:")
    log_likelihoods = _log_likelihoods(mcmc_trace, experiments, nsamples)
    log_probs = log_priors + log_likelihoods
    # map_idx = np.argmax(log_probs)
    map_idx = np.nanargmax(log_probs)
    print("Map index: %d" % map_idx)

    map_params = {}
    for name in mcmc_trace.keys():
        map_params[name] = mcmc_trace[name][map_idx]

    return [map_idx, map_params, log_probs]


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
    # return stats.norm.pdf((x-mean)/std, 0, 1)
    return jnp.exp(dist.Normal(loc=0, scale=1).log_prob((x-mean)/std))


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
    # zs = (response_model - response_actual)/sigma
    # norm_rv = stats.norm(loc=0, scale=1)
    # return np.sum(norm_rv.logpdf(zs))
    # return jnp.sum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))
    return jnp.nansum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))
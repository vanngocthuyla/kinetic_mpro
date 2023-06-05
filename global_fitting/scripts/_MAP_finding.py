import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import numpyro.distributions as dist

from _prior_distribution import logsigma_guesses
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
    # return jnp.sum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))
    return jnp.nansum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))


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


def _log_prior_uniform(mcmc_trace, experiments, params_logK_name=None, params_kcat_name=None, nsamples=None):
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


def _log_prior(experiments, evaluated_params, params_dist):
    """
    Sum of log prior of all parameters given their distribution information in params_dist

    Parameters:
    ----------
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    evaluated_params: list of evaluated parameter set
    params_dist     : list of dict
        Each dataset contains type, name, distribution and its parameter. For example: 
        
        params_dist = []
        params_dist.append({'type':'logK', 'name': 'logKd', 'dist': 'normal', 'loc': -5, 'scale': 2}
        params_dist.append({'type':'kcat', 'name': 'kcat_MS', 'dist': 'uniform', 'lower': 0, 'upper': 1})
    ----------
    Return: 
        An array which size equals to length of evaluated_params, each position corresponds 
        to sum of log prior distributions
    """
    nsamples = len(evaluated_params[params_dist[0]['name']])
    log_priors = jnp.zeros(nsamples)

    normal_priors = {}
    uniform_priors = {}

    for params in params_dist:
        if params['dist'] == 'normal':
            normal_priors[params['name']] = {'loc': params['loc'], 'scale': params['scale']}
            
        if params['dist'] == 'uniform':
            uniform_priors[params['name']] = {'lower': params['lower'], 'upper': params['upper']}

    for name in normal_priors:
        param_list = evaluated_params[name]
        f_prior = vmap(lambda param: _gaussian_pdf(param, normal_priors[name]['loc'], normal_priors[name]['scale']))
        log_priors += jnp.log(f_prior(param_list))

    for name in uniform_priors:
        param_list = evaluated_params[name]
        f_prior = vmap(lambda param: _uniform_pdf(param, uniform_priors[name]['lower'], uniform_priors[name]['upper']))
        log_priors += jnp.log(f_prior(param_list))

    for (n, experiment) in enumerate(experiments):
        name_tech = experiment['type']

        if name_tech == 'kinetics':
            rate = experiment['v']
            log_sigma_min, log_sigma_max = logsigma_guesses(rate)

            f_prior_sigma = vmap(lambda sigma: _uniform_pdf(sigma, log_sigma_min, log_sigma_max))
            try: 
                param_list = evaluated_params['log_sigma_rate']
            except:
                param_list = evaluated_params[experiment['log_sigma']]
            log_priors += jnp.log(f_prior_sigma(param_list))

        if name_tech == 'AUC':
            auc = experiment['M']
            log_sigma_min, log_sigma_max = logsigma_guesses(auc)

            f_prior_sigma = vmap(lambda sigma: _uniform_pdf(sigma, log_sigma_min, log_sigma_max))
            try:
                param_list = evaluated_params['log_sigma_auc']
            except:
                param_list = evaluated_params[experiment['log_sigma']]
            log_priors += jnp.log(f_prior_sigma(param_list))
        
        if name_tech == 'catalytic_efficiency':
            ice = experiment['Km_over_kcat']
            log_sigma_min, log_sigma_max = logsigma_guesses(ice)

            f_prior_sigma = vmap(lambda sigma: _uniform_pdf(sigma, log_sigma_min, log_sigma_max))
            try:
                param_list = evaluated_params['log_sigma_ice']
            except:
                param_list = evaluated_params[experiment['log_sigma']]
            log_priors += jnp.log(f_prior_sigma(param_list))
    
    return np.array(log_priors)


def _log_likelihood(experiments, evaluated_params, params_dist):
    """
    Sum of log likelihood of all parameters given their distribution information in params_dist

    Parameters:
    ----------
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    evaluated_params: list of evaluated parameter set
    params_dist     : list of dict
        Each dataset contains type, name, distribution and its parameter. For example: 
        
        params_dist = []
        params_dist.append({'type':'logK', 'name': 'logKd', 'dist': 'normal', 'loc': -5, 'scale': 2}
        params_dist.append({'type':'kcat', 'name': 'kcat_MS', 'dist': 'uniform', 'lower': 0, 'upper': 1})
    ----------
    Return: 
        An array which size equals to length of evaluated_params, each position corresponds 
        to sum of log likelihood calculated by values of parameters from evaluated_params
    """
    params_logK_name = []
    params_kcat_name = []
    for param in params_dist:
        if param['type'] == 'logK':
            params_logK_name.append(param['name'])
        if param['type'] == 'kcat':
            params_kcat_name.append(param['name'])

    full_params_logK_name = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
    full_params_kcat_name = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']

    nsamples = len(evaluated_params[params_dist[0]['name']])
    log_likelihoods = jnp.zeros(nsamples)

    for (n, experiment) in enumerate(experiments):
        print("Experiment", experiment['type'], n)
        name_tech = experiment['type']

        logK_list = {}
        kcat_list = {}
        for name in full_params_logK_name: 
            if name in params_logK_name: 
                logK_list[name] = evaluated_params[name]
            else:
                logK_list[name] = jnp.zeros(nsamples)
        for name in full_params_kcat_name:
            if name in params_kcat_name:
                kcat_list[name] = evaluated_params[name]
            else:
                kcat_list[name] = jnp.zeros(nsamples)

        if name_tech == 'kinetics':
            rate = experiment['v']
            kinetics_logMtot = experiment['logMtot']
            kinetics_logStot = experiment['logStot']
            kinetics_logItot = experiment['logItot']
            try:
                log_sigma_trace = evaluated_params['log_sigma_rate'][: nsamples]
            except:
                log_sigma_trace = evaluated_params[experiment['log_sigma']][: nsamples]
            f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, log_sigma: _log_likelihood_normal(rate, 
                                                                                                                                                                                    ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, 
                                                                                                                                                                                                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                                 kcat_MS, kcat_DS, kcat_DSI, kcat_DSS), 
                                                                                                                                                                                    jnp.exp(log_sigma)) )
            log_likelihoods += f(logK_list['logKd'], logK_list['logK_S_M'], logK_list['logK_S_D'], 
                                 logK_list['logK_S_DS'], logK_list['logK_I_M'], logK_list['logK_I_D'], 
                                 logK_list['logK_I_DI'], logK_list['logK_S_DI'], kcat_list['kcat_MS'], 
                                 kcat_list['kcat_DS'], kcat_list['kcat_DSI'], kcat_list['kcat_DSS'], 
                                 log_sigma_trace)
            
        if name_tech == 'AUC':
            auc = experiment['M']
            AUC_logMtot = experiment['logMtot']
            AUC_logStot = experiment['logStot']
            AUC_logItot = experiment['logItot']
            try:
                log_sigma_trace = evaluated_params['log_sigma_auc']
            except:
                log_sigma_trace = evaluated_params[experiment['log_sigma']]
            f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, log_sigma: _log_likelihood_normal(auc,
                                                                                                                                              MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot,
                                                                                                                                                                   logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI),
                                                                                                                                              jnp.exp(log_sigma)) )
            log_likelihoods += f(logK_list['logKd'], logK_list['logK_S_M'], logK_list['logK_S_D'], 
                                 logK_list['logK_S_DS'], logK_list['logK_I_M'], logK_list['logK_I_D'], 
                                 logK_list['logK_I_DI'], logK_list['logK_S_DI'],
                                 log_sigma_trace)

        if name_tech == 'catalytic_efficiency':
            ice = experiment['Km_over_kcat']
            ice_logMtot = experiment['logMtot']
            ice_logStot = experiment['logStot']
            ice_logItot = experiment['logItot']
            try:
                log_sigma_trace = evaluated_params['log_sigma_ice']
            except:
                log_sigma_trace = evaluated_params[experiment['log_sigma']]
            f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, log_sigma: _log_likelihood_normal(ice, 
                                                                                                                                                                                    1./CatalyticEfficiency(ice_logMtot, ice_logItot,
                                                                                                                                                                                                           logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                                           kcat_MS, kcat_DS, kcat_DSI, kcat_DSS), 
                                                                                                                                                                                    jnp.exp(log_sigma)) )
            log_likelihoods += f(logK_list['logKd'], logK_list['logK_S_M'], logK_list['logK_S_D'], 
                                 logK_list['logK_S_DS'], logK_list['logK_I_M'], logK_list['logK_I_D'], 
                                 logK_list['logK_I_DI'], logK_list['logK_S_DI'], kcat_list['kcat_MS'], 
                                 kcat_list['kcat_DS'], kcat_list['kcat_DSI'], kcat_list['kcat_DSS'], 
                                 log_sigma_trace)

    return np.array(log_likelihoods)


def _map_uniform(mcmc_trace, experiments, params_logK_name=None, params_kcat_name=None, nsamples=None):
    """
    Finding MAP (maximum a posterior) assuming prior of parameters follow uniform distribution

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

    log_priors = _log_prior_uniform(mcmc_trace, experiments, params_logK_name, params_kcat_name, nsamples)
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


def _map_finding(mcmc_trace, experiments, params_to_evaluate=None, params_dist=None):
    """
    Evaluate probability of a parameter set using posterior distribution
    Finding MAP (maximum a posterior) given prior distributions of parameters in params_dist

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    evaluated_params: list of evaluated parameter set
    params_dist     : list of dict
        Each dataset contains type, name, distribution and its parameter. For example: 
        
        params_dist = []
        params_dist.append({'type':'logK', 'name': 'logKd', 'dist': 'normal', 'loc': -5, 'scale': 2}
        params_dist.append({'type':'kcat', 'name': 'kcat_MS', 'dist': 'uniform', 'lower': 0, 'upper': 1})
    ----------
    Return          : values of parameters that maximize the posterior
    """
    if params_dist is None:
        params_dist = []
        for name in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
            params_dist.append({'type':'logK', 'name': name, 'dist': 'uniform', 'lower': -20, 'upper': 0})
        for name in ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']:
            params_dist.append({'type':'kcat', 'name': name, 'dist': 'uniform', 'lower': 0, 'upper': 1})

    if params_to_evaluate is None:
        evaluated_params = mcmc_trace
    else:
        evaluated_params = params_to_evaluate        

    nsamples = len(evaluated_params[params_dist[0]['name']])

    log_priors = _log_prior(experiments, evaluated_params, params_dist)
    log_likelihoods = _log_likelihood(experiments, evaluated_params, params_dist)
    log_probs = log_priors + log_likelihoods
    # map_idx = np.argmax(log_probs)
    map_idx = np.nanargmax(log_probs)
    print("Map index: %d" % map_idx)

    map_params = {}
    for name in evaluated_params.keys():
        map_params[name] = evaluated_params[name][map_idx]

    return [log_probs, map_params, map_idx]
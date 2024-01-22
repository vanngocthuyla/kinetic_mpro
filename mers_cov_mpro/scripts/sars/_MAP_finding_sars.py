import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import pandas as pd
import numpyro.distributions as dist

from _prior_distribution import logsigma_guesses
from _kinetics_adjustable_constraints import Adjustable_ReactionRate, Adjustable_MonomerConcentration, Adjustable_CatalyticEfficiency
from _load_prior_csv import _prior_group_name

from _MAP_finding import _log_prior_sigma, _extract_logK_kcat, _gaussian_pdf, _uniform_pdf, _lognormal_pdf, _log_normal_likelihood, _map_adjust_trace


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
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, log_sigma: _log_normal_likelihood(rate, 
                                                                                                                                                                                        Adjustable_ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, 
                                                                                                                                                                                                                logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                                                                                                                                                                                logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                                                                                                                                                                                                kcat_MS, kcat_DS, kcat_DSI, kcat_DSS),
                                                                                                                                                                                        jnp.exp(log_sigma)),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'], 
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'], 
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], trace_logK['logKsp'], 
                             trace_kcat['kcat_MS'], trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'], 
                             trace_log_sigma)
    if type_expt == 'AUC':
        [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data

        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp, log_sigma: _log_normal_likelihood(auc,
                                                                                                                                                  Adjustable_MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot,
                                                                                                                                                                                  logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                                                                                                                                                  logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp),
                                                                                                                                                  jnp.exp(log_sigma)),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'], 
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'], 
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], trace_logK['logKsp'], 
                             trace_log_sigma)
    
    if type_expt == 'ICE':
        [ice, ice_logMtot, ice_logStot, ice_logItot] = data
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, log_sigma: _log_normal_likelihood(ice, 
                                                                                                                                                                                        1./Adjustable_CatalyticEfficiency(ice_logMtot, ice_logItot,
                                                                                                                                                                                                                          logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                                                                                                                                                                                          logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                                                                                                                                                                                                          kcat_MS, kcat_DS, kcat_DSI, kcat_DSS), 
                                                                                                                                                                                        jnp.exp(log_sigma)),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'], 
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'], 
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], trace_logK['logKsp'], 
                             trace_kcat['kcat_MS'], trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'], 
                             trace_log_sigma)
    return log_likelihoods


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

        trace_nth, in_axis_nth = _extract_logK_kcat(mcmc_trace, idx, nsamples)

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


def map_finding(mcmc_trace, experiments, prior_infor, set_K_I_M_equal_K_S_M=False,
                set_K_S_DI_equal_K_S_DS=False, set_kcat_DSS_equal_kcat_DS=False,
                set_kcat_DSI_equal_kcat_DS=False, set_kcat_DSI_equal_kcat_DSS=False, 
                nsamples=None):
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
    mcmc_trace_update = _map_adjust_trace(mcmc_trace, experiments, prior_infor, 
                                          set_K_I_M_equal_K_S_M, set_K_S_DI_equal_K_S_DS, 
                                          set_kcat_DSS_equal_kcat_DS, set_kcat_DSI_equal_kcat_DS,
                                          set_kcat_DSI_equal_kcat_DSS)
    log_likelihoods = _log_likelihoods(mcmc_trace_update, experiments, nsamples)
    
    log_probs = log_priors + log_likelihoods
    # map_idx = np.argmax(log_probs)
    map_idx = np.nanargmax(log_probs)
    print("Map index: %d" % map_idx)

    map_params = {}
    for name in mcmc_trace.keys():
        map_params[name] = mcmc_trace[name][map_idx]

    return [map_idx, map_params, log_probs]
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import pandas as pd
import numpyro.distributions as dist

from _prior_distribution import logsigma_guesses
from _kinetics import ReactionRate
from _load_prior_csv import _prior_group_name

from _MAP_finding import _log_prior_sigma, _extract_logK_kcat, _gaussian_pdf, _uniform_pdf, _lognormal_pdf, _log_normal_likelihood, _map_adjust_trace


def _log_priors(mcmc_trace, experiments, prior_infor, nsamples=None, 
                set_lognormal_dE=False, dE=0.1, dI=0.1):
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

    for idx, expt in enumerate(experiments):
        try: idx_expt = expt['index']
        except: idx_expt = idx

        if type(expt['kinetics']) is dict:
            _all_logItot = []
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                # log_prior of sigma
                if data_rate is not None and f'log_sigma:{idx_expt}:{n}' in mcmc_trace.keys():
                    log_priors += _log_prior_sigma(mcmc_trace, data_rate, f'log_sigma:{idx_expt}:{n}', nsamples)

                # Log_prior of uncertainty for inhibitor concentration if lognormal, calculating log_prior based on its logItot
                [rate, logMtot, logStot, logItot] = data_rate

                # If each dataset has its own inhibitor concentration error
                if f'I0:{idx_expt}:{n}' in mcmc_trace.keys():
<<<<<<< HEAD
                    log_priors += jnp.log(_lognormal_pdf(mcmc_trace[f'I0:{idx_expt}:{n}'][: nsamples], jnp.exp(jnp.max(logItot)), jnp.exp(jnp.max(logItot))*dI))
=======
                    log_priors += jnp.log(_lognormal_pdf(mcmc_trace[f'I0:{idx_expt}:{n}'][: nsamples], jnp.exp(jnp.max(logItot)), dI))
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
                
                # If all datasets shares one inhibitor concentration error, combining all logItot
                elif f'I0:{idx_expt}' in mcmc_trace.keys():
                    _all_logItot.append(logItot)

            # Calculating log_prior of combined logItot
            if len(_all_logItot)>0:
                max_logItot = jnp.max(np.concatenate(_all_logItot))
<<<<<<< HEAD
                log_priors += jnp.log(_lognormal_pdf(mcmc_trace[f'I0:{idx_expt}'][: nsamples], jnp.exp(max_logItot), jnp.exp(max_logItot)*dI))
=======
                log_priors += jnp.log(_lognormal_pdf(mcmc_trace[f'I0:{idx_expt}'][: nsamples], max_logItot, dI))
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
        else:
            data_rate = expt['kinetics']
            # log_prior of sigma
            if data_rate is not None and f'log_sigma:{idx_expt}' in mcmc_trace.keys():
                log_priors += _log_prior_sigma(mcmc_trace, data_rate, f'log_sigma:{idx_expt}', nsamples)

            # Log_prior of uncertainty for concentration if lognormal
            [rate, logMtot, logStot, logItot] = data_rate
            if f'I0:{idx_expt}' in mcmc_trace.keys():
<<<<<<< HEAD
                log_priors += jnp.log(_lognormal_pdf(mcmc_trace[f'I0:{idx_expt}'][: nsamples], jnp.exp(jnp.max(logItot)), jnp.exp(jnp.max(logItot))*dI))
=======
                log_priors += jnp.log(_lognormal_pdf(mcmc_trace[f'I0:{idx_expt}'][: nsamples], jnp.exp(jnp.max(logItot)), dI))
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0

    for key in mcmc_trace.keys():
        # log_prior of normalization factor:
        if key.startswith('alpha'):
            log_priors += jnp.log(_uniform_pdf(mcmc_trace[key][: nsamples], 0, 2))
        # log prior of enzyme concentration uncertainty
        if key.startswith('dE'): 
            # If dE follow lognormal
            if set_lognormal_dE and dE>0:
                conc = int(key[3:])
<<<<<<< HEAD
                log_priors = jnp.log(_lognormal_pdf(mcmc_trace[key][: nsamples], conc, dE*conc))
=======
                log_priors = jnp.log(_lognormal_pdf(mcmc_trace[key][: nsamples], conc, dE))
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
            # If dE follow uniform
            elif dE>0 and dE<1: 
                log_priors += jnp.log(_uniform_pdf(mcmc_trace[key][: nsamples], 0, dE))

    return np.array(log_priors)


def _log_likelihood_each_enzyme(type_expt, data, trace_logK, trace_kcat, trace_alpha, trace_sigma, in_axes_nth, nsamples):
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
    without enzyme/ligand concentration uncertainty.
    """
    assert type_expt in ['kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, or ICE."
    log_likelihoods = jnp.zeros(nsamples)

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, alpha, sigma: _log_normal_likelihood(rate,
                                                                                                                                                                                           ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                                                                                                                                                                                        logKd, logK_S_M, logK_S_D, logK_S_DS,
                                                                                                                                                                                                        logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                                                                                                                                                                                        kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)*alpha,
                                                                                                                                                                                           sigma),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'],
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'],
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], trace_logK['logKsp'],
                             trace_kcat['kcat_MS'], trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'],
                             trace_alpha, trace_sigma)
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
        Sum of log likelihood given experiments and mcmc_trace without enzyme/ligand concentration uncertainty.
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
        in_axis_nth = []
        try: idx_expt = expt['index']
        except: idx_expt = idx

        trace_nth, in_axis_nth = _extract_logK_kcat(mcmc_trace, idx, nsamples)

        # alpha
        if f'alpha:{idx_expt}' in mcmc_trace.keys():
            trace_alpha = mcmc_trace[f'alpha:{idx_expt}'][: nsamples]
            in_axis_nth.append(0)
        elif 'alpha' in mcmc_trace.keys():
            trace_alpha = mcmc_trace['alpha'][: nsamples]
            in_axis_nth.append(0)
        else: 
            trace_alpha = None
            in_axis_nth.append(None)

        in_axis_nth.append(0) #for sigma
        if type(expt['kinetics']) is dict:
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None:
                    print("Kinetics experiment", idx_expt)
                    if f'log_sigma:{idx_expt}:{n}' in mcmc_trace.keys():
                        trace_sigma = jnp.exp(mcmc_trace[f'log_sigma:{idx_expt}:{n}'][: nsamples])
                    else:
                        trace_sigma = jnp.zeros(nsamples)
                    log_likelihoods += _log_likelihood_each_enzyme('kinetics', data_rate, trace_nth, trace_nth, trace_alpha, trace_log_sigma, in_axis_nth, nsamples)
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                print("Kinetics experiment", idx_expt)
                if f'log_sigma:{idx_expt}' in mcmc_trace.keys():
                    trace_sigma = jnp.exp(mcmc_trace[f'log_sigma:{idx_expt}'][: nsamples])
                else:
                    trace_sigma = jnp.zeros(nsamples)
                log_likelihoods += _log_likelihood_each_enzyme('kinetics', data_rate, trace_nth, trace_nth, trace_alpha, trace_sigma, in_axis_nth, nsamples)

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
    map_idx = np.nanargmax(log_probs) #np.argmax(log_probs)
    print("Map index: %d" % map_idx)

    map_params = {}
    for name in mcmc_trace.keys():
        map_params[name] = mcmc_trace[name][map_idx]

    return [map_idx, map_params, log_probs]
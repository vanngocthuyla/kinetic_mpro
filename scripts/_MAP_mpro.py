import os
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import pandas as pd
import numpyro.distributions as dist
import pickle

from _prior_distribution import logsigma_guesses
from _load_prior_from_csv import _prior_group_name
from _kinetics import DimerBindingModel, Enzyme_Substrate, ReactionRate, MonomerConcentration, CatalyticEfficiency
from _kinetics import adjust_DimerBindingModel, adjust_ReactionRate, adjust_MonomerConcentration, adjust_CatalyticEfficiency

from _MAP import _extract_logK_kcat_trace, _uniform_pdf, _gaussian_pdf, _lognormal_pdf, _log_likelihood_normal, _map_adjust_trace, _log_prior_sigma
from _model import _dE_find_prior, _alpha_find_prior


def _map_finding(mcmc_trace, experiments, prior_infor, args, nsamples=None, 
                 adjust_fit=True, show_progress=True):
    """
    Evaluate probability of a parameter set using posterior distribution
    Finding MAP (maximum a posterior) given prior distributions of parameters information

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling trace (group_by_chain=False)
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    args            : class comprises other model arguments. For more information, check _define_model.py
    adjust_fit      : boolean, use adjustable fitting
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

    if show_progress:
        print("Calculing log of priors.")
    log_priors = _log_priors(mcmc_trace=mcmc_trace, experiments=experiments, prior_infor=prior_infor, nsamples=nsamples,
                             set_lognormal_dE=args.set_lognormal_dE, dE=args.dE, alpha_min=args.alpha_min, alpha_max=args.alpha_max)

    if show_progress:
        print("Calculing log likelihoods:")
    mcmc_trace_update = _map_adjust_trace(mcmc_trace=mcmc_trace.copy(), experiments=experiments, prior_infor=prior_infor,
                                          set_K_I_M_equal_K_S_M=args.set_K_I_M_equal_K_S_M, set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D,
                                          set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS, set_kcat_DSS_equal_kcat_DS=args.set_kcat_DSS_equal_kcat_DS,
                                          set_kcat_DSI_equal_kcat_DS=args.set_kcat_DSI_equal_kcat_DS, set_kcat_DSI_equal_kcat_DSS=args.set_kcat_DSI_equal_kcat_DSS,
                                          show_progress=show_progress)

    log_likelihoods = _log_likelihoods(mcmc_trace=mcmc_trace_update, experiments=experiments, alpha_list=args.alpha_list, E_list=args.E_list,
                                       nsamples=nsamples, adjust_fit=adjust_fit, show_progress=show_progress)

    log_probs = log_priors + log_likelihoods
    map_idx = np.nanargmax(log_probs)
    if show_progress:
        print("Map index: %d" % map_idx)

    map_params = {}
    for name in mcmc_trace.keys():
        map_params[name] = mcmc_trace[name][map_idx]

    return [map_idx, map_params, log_probs]


def _map_running(trace, expts, prior_infor, shared_params, args, adjust_fit=True):
    """
    Evaluate probability of a parameter set using posterior distribution
    Finding MAP (maximum a posterior) given prior distributions of parameters information

    Parameters:
    ----------
    trace           : list of dict, trace of Bayesian sampling trace (group_by_chain=False)
    expts           : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    shared_params   : dict, information for shared parameters
    adjust_fit      : boolean, use adjustable fitting
    args            : class comprises other model arguments. For more information, check _define_model.py
    ----------
    Return          : adjusted trace and map index
    """
    trace_map   = trace.copy()
    traces_name = args.traces_name

    if shared_params is not None and len(shared_params)>0:
        for name in shared_params.keys():
            param = shared_params[name]
            assigned_idx = param['assigned_idx']
            shared_idx = param['shared_idx']
            trace_map[f'{name}:{assigned_idx}'] = trace_map[f'{name}:{shared_idx}']
        pickle.dump(trace_map, open(os.path.join('MAP_'+traces_name+'.pickle'), "wb"))

    if not args.log_sigmas is None: 
        log_sigmas = args.log_sigmas
        for key in log_sigmas.keys():
            if key.startswith('log_sigma'):
                trace_map[key] = jnp.repeat(log_sigmas[key], args.niters*args.nchain)

    [map_index, map_params, log_probs] = _map_finding(mcmc_trace=trace_map, experiments=expts, prior_infor=prior_infor, 
                                                      args=args, nsamples=args.nsamples_MAP, adjust_fit=adjust_fit)

    with open("map.txt", "w") as f:
        print("MAP index:" + str(map_index), file=f)
        print("\nKinetics parameters:", file=f)
        for key in trace.keys():
            print(key, ': %.3f' %trace[key][map_index], file=f)

    pickle.dump(log_probs, open('log_probs.pickle', "wb"))

    map_values = {}
    for key in trace.keys():
        map_values[key] = trace[key][map_index]
    pickle.dump(map_values, open('map.pickle', "wb"))

    return [trace_map, map_index]


## Log Prior --------------------------------------------------------------------------------------

def _log_prior_sigma_each_expt(type_expt, expt, idx_expt, mcmc_trace, nsamples):
    """
    Sum of log prior of measurement error of each dataset

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    expt            : dict contains the information of all dataset within experiment
                      Each dataset contains response, logMtot, lotStot, logItot
    idx_expt        : str, index of experiment
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    mcmc_trace      : list of dict, trace of Bayesian sampling
    nsamples        : int, number of samples to find MAP
    ----------
    Return:
        Sum of log prior of measurement error of all dataset in given experiment
    """
    assert type_expt in ['CRC', 'kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, ICE, or CRC."
    
    log_priors_sigma = 0

    if type_expt == 'kinetics':
        prefix = 'rate'
    elif type_expt == 'AUC':
        prefix = 'AUC'
    elif type_expt == 'ICE':
        prefix = 'ICE'
    elif type_expt == 'CRC':
        prefix = 'CRC'

    if type(expt[type_expt]) is dict:
        for n in range(len(expt[type_expt])):
            data = expt[type_expt][n]
            if data is not None:
                log_priors_sigma += _log_prior_sigma(mcmc_trace=mcmc_trace, data=data, sigma_name=f'log_sigma_{prefix}:{idx_expt}:{n}', nsamples=nsamples)
    else:
        data = expt[type_expt]
        if data is not None:
            log_priors_sigma += _log_prior_sigma(mcmc_trace=mcmc_trace, data=data, sigma_name=f'log_sigma_{prefix}:{idx_expt}', nsamples=nsamples)

    return log_priors_sigma


def _log_priors(mcmc_trace, experiments, prior_infor, nsamples=None,
                set_lognormal_dE=False, dE=0.1, alpha_min=0., alpha_max=2.):
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

        for _type_expt in ['kinetics', 'AUC', 'ICE', 'CRC']:
            if _type_expt in expt.keys():
                log_priors += _log_prior_sigma_each_expt(_type_expt, expt, idx_expt, mcmc_trace, nsamples)

    for key in mcmc_trace.keys():
        # log_prior of normalization factor:
        if key.startswith('alpha'):
            log_priors += jnp.log(_uniform_pdf(mcmc_trace[key][: nsamples], alpha_min, alpha_max))
        
        # log prior of enzyme concentration uncertainty
        if key.startswith('dE'):
            conc = int(key[3:])
            # If dE follow lognormal
            if set_lognormal_dE and dE>0:
                log_priors = jnp.log(_lognormal_pdf(mcmc_trace[key][: nsamples], conc, dE*conc))
            # If dE follow uniform
            elif dE>0 and dE<1:
                log_priors += jnp.log(_uniform_pdf(mcmc_trace[key][: nsamples], (1-dE)*conc, (1+dE)*conc))

    return np.array(log_priors)


## Log Likelihood --------------------------------------------------------------------------------------

def _extract_conc_lognormal(logConc, expt_value):
    """
    Parameters
    ----------
    logConc     : numpy array, concenntration of a species
    expt_value  : float, percentage of error concentration degraded within the reaction

    Caculating the ratio between the adjusted and the theoretical values to return the array of concentration
    """
    stated_value = jnp.exp(jnp.max(logConc))
    ratio = expt_value/stated_value
    return logConc + jnp.log(ratio)


def _log_likelihoods(mcmc_trace, experiments, alpha_list=None, E_list=None, nsamples=None, 
                     adjust_fit=False, show_progress=True):
    """
    Sum of log likelihood of all parameters given their distribution information in params_dist

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    adjust_fit      : boolean, use adjustable fitting
    ----------
    Return:
        Sum of log likelihood given experiments, mcmc_trace, enzyme/ligand concentration uncertainty
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
        # in_axis_nth = []
        try: idx_expt = expt['index']
        except: idx_expt = idx

        trace_nth, in_axis_nth = _extract_logK_kcat_trace(mcmc_trace, idx, nsamples)

        if 'CRC' in expt.keys():
            func = _log_likelihood_each_dataset
            if type(expt['CRC']) is dict:
                for n in range(len(expt['CRC'])):
                    data_rate = expt['CRC'][n]
                    if data_rate is not None and 'plate' in expt.keys():
                        plate = expt['plate'][n]
                        if show_progress:
                            print("CRC experiment", idx_expt, ":", n)

                        if f'alpha:{idx_expt}:{n}' in mcmc_trace.keys(): #multiple alpha for each experiment
                            trace_alpha = mcmc_trace[f'alpha:{idx_expt}:{n}'][: nsamples]
                        elif f'alpha:{plate}' in mcmc_trace.keys(): #shared alpha among experiments with same plate
                            trace_alpha = mcmc_trace[f'alpha:{plate}'][: nsamples]
                        elif alpha_list is not None and f'alpha:{plate}' in alpha_list: #provided alpha list for multiple plates
                            trace_alpha = jnp.ones(nsamples)*alpha_list[f'alpha:{plate}']
                        else:
                            trace_alpha = jnp.ones(nsamples)
                        if n == 0: in_axis_nth.append(0) #index of vmap for alpha

                        if f'log_sigma_CRC:{idx_expt}:{n}' in mcmc_trace.keys():
                            trace_sigma = jnp.exp(mcmc_trace[f'log_sigma_CRC:{idx_expt}:{n}'][: nsamples])
                        else:
                            trace_sigma = jnp.ones(nsamples)
                        if n == 0: in_axis_nth.append(0) #index of vmap for sigma

                        if E_list is not None:
                            _trace_error_E = _dE_find_prior(data_rate, E_list)
                            _trace_error_E = np.reshape(np.repeat(_trace_error_E, len(mcmc_trace[params_name_logK[0]])), (len(_trace_error_E), len(mcmc_trace[params_name_logK[0]])))
                        else:
                            _trace_error_E = _dE_find_prior(data_rate, mcmc_trace)
                        if np.size(_trace_error_E)>0:
                            trace_error_E = _trace_error_E[:, : nsamples].T
                            if n==0: in_axis_nth.append(0)
                        else:
                            trace_error_E = None
                            if n==0: in_axis_nth.append(None)

                        log_likelihoods += func(type_expt='CRC', data=data_rate,
                                                trace_logK=trace_nth, trace_kcat=trace_nth,
                                                trace_alpha=trace_alpha, trace_sigma=trace_sigma,
                                                trace_error_E=trace_error_E,
                                                in_axes_nth=in_axis_nth, nsamples=nsamples,
                                                adjust_fit=adjust_fit)

            else:
                data_rate = expt['CRC']
                if data_rate is not None and 'plate' in expt.keys():
                    plate = expt['plate']
                    if show_progress:
                        print("CRC experiment", idx_expt)

                    if f'alpha:{idx_expt}' in mcmc_trace.keys(): #multiple alpha for each experiment
                        trace_alpha = mcmc_trace[f'alpha:{idx_expt}'][: nsamples]
                    elif f'alpha:{plate}' in mcmc_trace.keys(): #shared alpha among experiments with same plate
                        trace_alpha = mcmc_trace[f'alpha:{plate}'][: nsamples]
                    elif alpha_list is not None and f'alpha:{plate}' in alpha_list: #provided alpha list for multiple plates
                        trace_alpha = jnp.one(nsamples)*alpha_list[f'alpha:{plate}']
                    else:
                        trace_alpha = jnp.ones(nsamples)
                    in_axis_nth.append(0)

                    if f'log_sigma_CRC:{idx_expt}' in mcmc_trace.keys():
                        trace_sigma = jnp.exp(mcmc_trace[f'log_sigma_CRC:{idx_expt}'][: nsamples])
                    else:
                        trace_sigma = jnp.ones(nsamples)
                    in_axis_nth.append(0)

                    if E_list is not None:
                        _trace_error_E = _dE_find_prior(data_rate, E_list)
                        _trace_error_E = np.reshape(np.repeat(_trace_error_E, len(mcmc_trace[params_name_logK[0]])), (len(_trace_error_E), len(mcmc_trace[params_name_logK[0]])))
                    else:
                        _trace_error_E = _dE_find_prior(data_rate, mcmc_trace)
                    if np.size(_trace_error_E)>0:
                        trace_error_E = _trace_error_E[:, : nsamples].T
                        in_axis_nth.append(0)
                    else:
                        trace_error_E = None
                        in_axis_nth.append(None)

                    log_likelihoods += func(type_expt='CRC', data=data_rate,
                                            trace_logK=trace_nth, trace_kcat=trace_nth,
                                            trace_alpha=trace_alpha, trace_sigma=trace_sigma,
                                            trace_error_E=trace_error_E,
                                            in_axes_nth=in_axis_nth, nsamples=nsamples,
                                            adjust_fit=adjust_fit)

        for _type_expt in ['kinetics', 'AUC', 'ICE']:
            func = _log_likelihood_each_expt
            if _type_expt in expt.keys():
                log_likelihoods += _log_likelihood_each_expt(type_expt=_type_expt, expt=expt, 
                                                             idx_expt=idx_expt, mcmc_trace=mcmc_trace, 
                                                             idx=idx, nsamples=nsamples,
                                                             adjust_fit=adjust_fit)

    return np.array(log_likelihoods)


def _log_likelihood_each_expt(type_expt, expt, idx_expt, mcmc_trace, idx, nsamples=None,
                              adjust_fit=False):
    """
    Parameters:
    ----------
    type_expt       : str, 'kinetics', 'AUC', or 'ICE'
    expt            : dict contains the information of all dataset within experiment
                      Each dataset contains response, logMtot, lotStot, logItot
    idx_expt        : str, index of experiment that was assigned for name of sigma
    mcmc_trace      : list of dict, trace of Bayesian sampling
    idx             : int, ordered index of experiment
    nsamples        : int, number of samples to find MAP
    adjust_fit      : boolean, use adjustable fitting
    ----------
    Return log likelihood given type of experiment, experiment, mcmc_trace, and nsamples
    
    """
    assert type_expt in ['kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, ICE."
    log_likelihoods = 0.
    
    trace_nth, in_axis_nth = _extract_logK_kcat_trace(mcmc_trace, idx, nsamples)
    in_axis_nth.append(0) #adding one more in_axis for sigma

    if type_expt == 'kinetics':
        prefix = 'rate'
    elif type_expt == 'AUC':
        prefix = 'AUC'
        in_axis_nth = [*in_axis_nth[:8], 0] #no kcat in this experiment
    elif type_expt == 'ICE':
        prefix = 'ICE'

    func = _log_likelihood_each_dataset

    if type(expt[type_expt]) is dict:
        for n in range(len(expt[type_expt])):
            data = expt[type_expt][n]
            if data is not None:
                print(f"{type_expt} experiment {idx_expt}:{n}")
                if f'log_sigma_{prefix}:{idx_expt}:{n}' in mcmc_trace.keys():
                    trace_sigma = jnp.exp(mcmc_trace[f'log_sigma_{prefix}:{idx_expt}:{n}'][:nsamples])
                else:
                    trace_sigma = jnp.ones(nsamples)
                log_likelihoods += func(type_expt=type_expt, data=data,
                                        trace_logK=trace_nth, trace_kcat=trace_nth,
                                        trace_alpha=None, trace_sigma=trace_sigma, 
                                        trace_error_E=None, in_axes_nth=in_axis_nth, 
                                        nsamples=nsamples, adjust_fit=adjust_fit)
    else:
        data = expt[type_expt]
        if data is not None:
            print(f"{type_expt} experiment {idx_expt}")
            if f'log_sigma_{prefix}:{idx_expt}' in mcmc_trace.keys():
                trace_sigma = jnp.exp(mcmc_trace[f'log_sigma_{prefix}:{idx_expt}'][:nsamples])
            else:
                trace_sigma = jnp.ones(nsamples)
            log_likelihoods += func(type_expt=type_expt, data=data,
                                    trace_logK=trace_nth, trace_kcat=trace_nth,
                                    trace_alpha=None, trace_sigma=trace_sigma, 
                                    trace_error_E=None, in_axes_nth=in_axis_nth, 
                                    nsamples=nsamples, adjust_fit=adjust_fit)

    return log_likelihoods


def _log_likelihood_each_dataset(type_expt, data, trace_logK, trace_kcat, trace_alpha,
                                 trace_sigma, trace_error_E, in_axes_nth, nsamples=None,
                                 adjust_fit=False):
    """
    Parameters:
    ----------
    type_expt     : str, 'kinetics', 'AUC', or 'ICE'
    data          : list, each dataset contains response, logMtot, lotStot, logItot
    trace_logK    : trace of all logK
    trace_logK    : trace of all kcat
    trace_sigma   : trace of log_sigma
    trace_error_E : trace of enzyme concentration uncertainty
    in_axes_nth   : index to fun jax.vmap
    nsamples        : int, number of samples to find MAP
    adjust_fit    : boolean, use adjustable fitting
    ----------
    Return log likelihood given the experiment, mcmc_trace, enzyme/ligand concentration uncertainty
    """
    assert type_expt in ['CRC', 'kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, ICE, or CRC."
    log_likelihoods = jnp.zeros(nsamples, dtype=jnp.float64)

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        if adjust_fit:
            func = adjust_ReactionRate
        else:
            func = ReactionRate
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, sigma: _log_likelihood_normal(rate,
                                                                                                                                                                            func(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                                                                                                                                                                 logKd, logK_S_M, logK_S_D, logK_S_DS,
                                                                                                                                                                                 logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                 kcat_MS, kcat_DS, kcat_DSI, kcat_DSS),
                                                                                                                                                                            sigma),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'],
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'],
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], trace_kcat['kcat_MS'],
                             trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'],
                             trace_sigma)
    if type_expt == 'AUC':
        [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data
        if adjust_fit:
            func = adjust_MonomerConcentration
        else:
            func = MonomerConcentration
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, sigma: _log_likelihood_normal(auc,
                                                                                                                                      func(AUC_logMtot, AUC_logStot, AUC_logItot,
                                                                                                                                           logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                                                                                                           logK_I_M, logK_I_D, logK_I_DI, logK_S_DI),
                                                                                                                                      sigma),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'],
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'],
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'],
                             trace_sigma)

    if type_expt == 'ICE':
        [ice, ice_logMtot, ice_logStot, ice_logItot] = data
        if adjust_fit:
            func = adjust_CatalyticEfficiency
        else:
            func = CatalyticEfficiency
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, sigma: _log_likelihood_normal(ice,
                                                                                                                                                                            1./func(ice_logMtot, ice_logItot,
                                                                                                                                                                                    logKd, logK_S_M, logK_S_D, logK_S_DS,
                                                                                                                                                                                    logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                                                                                                                                                                    kcat_MS, kcat_DS, kcat_DSI, kcat_DSS,
                                                                                                                                                                                    ice_logItot),
                                                                                                                                                                            sigma),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'],
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'],
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'], trace_kcat['kcat_MS'],
                             trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'],
                             trace_sigma)

    if type_expt == 'CRC':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        if adjust_fit:
            func = _adjust_ReactionRate_uncertainty_conc
        else:
            func = _ReactionRate_uncertainty_conc
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, alpha, sigma, error_E: _log_likelihood_normal(rate,
                                                                                                                                                                                            func(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                                                                                                                                                                                logKd, logK_S_M, logK_S_D, logK_S_DS,
                                                                                                                                                                                                logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                                                                                                                                                                                kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, error_E)*alpha,
                                                                                                                                                                                            sigma),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'],
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'],
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'],
                             trace_kcat['kcat_MS'], trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'],
                             trace_alpha, trace_sigma, trace_error_E)

    return log_likelihoods


def _ReactionRate_uncertainty_conc(logMtot, logStot, logItot,
                                   logKd, logK_S_M, logK_S_D, logK_S_DS,
                                   logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                   kcat_MS=0., kcat_DS=0., kcat_DSI=0., kcat_DSS=0.,
                                   error_E=None):
    """
    Similar to function kinetics_.ReactionRate, this function return the reaction rate 
    given the parameters and adjusted enzyme/ligand concentration
        v = kcat_MS*[MS] + kcat_DS*[DS] + kcat_DSS*[DSS] + kcat_DSI*[DSI]
    
    Additional parameter
    ----------
    error_E  : float
        Adjusted enzyme concentration (nM)

    """
    if kcat_MS is None: kcat_MS = 0.
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.

    if error_E is None: logM = logMtot
    else: logM = jnp.log(error_E*1E-9)
    
    if logItot is None:
        log_concs = Enzyme_Substrate(logM, logStot, logKd, logK_S_M, logK_S_D, logK_S_DS)
        v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    else:
        log_concs = DimerBindingModel(logM, logStot, logItot,
                                      logKd, logK_S_M, logK_S_D, logK_S_DS,
                                      logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
        v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    
    return v


def _adjust_ReactionRate_uncertainty_conc(logMtot, logStot, logItot,
                                          logKd, logK_S_M, logK_S_D, logK_S_DS,
                                          logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                          kcat_MS=0., kcat_DS=0., kcat_DSI=0., kcat_DSS=0.,
                                          error_E=None):
    """
    Similar to function kinetics_.ReactionRate, this function return the reaction rate 
    given the parameters and adjusted enzyme/ligand concentration
        v = kcat_MS*[MS] + kcat_DS*[DS] + kcat_DSS*[DSS] + kcat_DSI*[DSI]
    
    Additional parameter
    ----------
    error_E  : float
        Adjusted enzyme concentration (nM) 
    """
    if kcat_MS is None: kcat_MS = 0.
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.

    if error_E is None: logM = logMtot
    else: logM = jnp.log(error_E*1E-9)
    
    func = adjust_DimerBindingModel
    log_concs = func(logM, logStot, logItot,
                     logKd, logK_S_M, logK_S_D, logK_S_DS,
                     logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
    v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    
    return v
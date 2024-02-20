<<<<<<< HEAD
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import pandas as pd
import numpyro.distributions as dist

from _prior_distribution import logsigma_guesses
from _load_prior_csv import _prior_group_name
from _kinetics import DimerBindingModel, Enzyme_Substrate

from _MAP_finding import _extract_logK_kcat, _lognormal_pdf, _log_normal_likelihood, _map_adjust_trace
from _MAP_finding_mers import _log_priors
from _model_mers import _dE_find_prior, _alpha_find_prior


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


def _ReactionRate_uncertainty_conc(logMtot, logStot, logItot,
                                   logKd, logK_S_M, logK_S_D, logK_S_DS,
                                   logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                   kcat_MS=0., kcat_DS=0., kcat_DSI=0., kcat_DSS=0.,
                                   error_E=None, I0=None):
    """
    Similar to function kinetics_.ReactionRate, this function return the reaction rate 
    given the parameters and adjusted enzyme/ligand concentration
        v = kcat_MS*[MS] + kcat_DS*[DS] + kcat_DSS*[DSS] + kcat_DSI*[DSI]
    
    Parameters
    ----------
    logMtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd : float
        Log of the dissociation constant of dimerization
    logKd_MS_M: float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M: float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS: float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI: float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    logKsp   : float
        Log of the dissociation constant for an aqueous inhibitor aggregating into solid state
    kcat_MS: float
        Rate constant of monomer-substrate complex
    kcat_DS  : float
        Rate constant of dimer-substrate complex
    kcat_DSS : float
        Rate constant of dimer-substrate-substrate complex
    kcat_DSI : float
        Rate constant of dimer-substrate-inhibitor complex
    error_E  : float
        Enzyme concentration uncertainty
    I0       : float
        The adjusted value of the highest concentration returned by the model in normal scale (M)

    All dissociation constants are in units of log molar 
    """
    if kcat_MS is None: kcat_MS = 0.
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.

    if error_E is None: logM = logMtot
    # else: logM = _extract_conc_percent_error(logMtot, error_E)
    else: logM = jnp.log(error_E*1E-9)
    
    if I0 is None: logI = logItot
    else: logI = _extract_conc_lognormal(logItot, I0)
    
    if logItot is None:
        log_concs = Enzyme_Substrate(logM, logStot, logKd, logK_S_M, logK_S_D, logK_S_DS)
        v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    else:
        log_concs = DimerBindingModel(logM, logStot, logI,
                                      logKd, logK_S_M, logK_S_D, logK_S_DS,
                                      logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
        v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v


def _log_likelihood_each_enzyme(type_expt, data, trace_logK, trace_kcat, trace_alpha, 
                                trace_sigma, trace_error_E, trace_I0, in_axes_nth, nsamples):
    """
    Parameters:
    ----------
    type_expt     : str, 'kinetics', 'AUC', or 'ICE'
    data          : list, each dataset contains response, logMtot, lotStot, logItot
    trace_logK    : trace of all logK
    trace_logK    : trace of all kcat
    trace_sigma   : trace of log_sigma
    ----------
    Return log likelihood given the experiment, mcmc_trace, enzyme/ligand concentration uncertainty
    """
    assert type_expt in ['kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, or ICE."
    log_likelihoods = jnp.zeros(nsamples, dtype=jnp.float64)

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, alpha, sigma, error_E, I0: _log_normal_likelihood(rate,
                                                                                                                                                                                                _ReactionRate_uncertainty_conc(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                                                                                                                                                                                                               logKd, logK_S_M, logK_S_D, logK_S_DS,
                                                                                                                                                                                                                               logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                                                               kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, error_E, I0)*alpha,
                                                                                                                                                                                                sigma),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'],
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'],
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'],
                             trace_kcat['kcat_MS'], trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'],
                             trace_alpha, trace_sigma, trace_error_E, trace_I0)
    return log_likelihoods


def _log_likelihoods(mcmc_trace, experiments, alpha_list=None, E_list=None, nsamples=None, show_progress=True):
    """
    Sum of log likelihood of all parameters given their distribution information in params_dist

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
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
        in_axis_nth = []
        try: idx_expt = expt['index']
        except: idx_expt = idx

        trace_nth, in_axis_nth = _extract_logK_kcat(mcmc_trace, idx, nsamples)

        if type(expt['kinetics']) is dict:
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                plate = expt['plate'][n]
                if data_rate is not None:
                    if show_progress: 
                        print("Kinetics experiment", idx_expt, ":", n)
                    
                    if f'alpha:{idx_expt}:{n}' in mcmc_trace.keys(): #multiple alpha for each experiment
                        trace_alpha = mcmc_trace[f'alpha:{idx_expt}:{n}'][: nsamples]
                    elif f'alpha:{plate}' in mcmc_trace.keys(): #shared alpha among experiments with same plate
                        trace_alpha = mcmc_trace[f'alpha:{plate}'][: nsamples]
                    elif alpha_list is not None and f'alpha:{plate}' in alpha_list: #provided alpha list for multiple plates
                        trace_alpha = jnp.ones(nsamples)*alpha_list[f'alpha:{plate}']
                    else:
                        trace_alpha = jnp.ones(nsamples)
                    if n == 0: in_axis_nth.append(0) #index of vmap for alpha

                    if f'log_sigma:{idx_expt}:{n}' in mcmc_trace.keys():
                        trace_sigma = jnp.exp(mcmc_trace[f'log_sigma:{idx_expt}:{n}'][: nsamples])
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

                    if f'I0:{idx_expt}:{n}' in mcmc_trace.keys():
                        trace_I0 = mcmc_trace[f'I0:{idx_expt}:{n}'][: nsamples]
                        if n == 0: in_axis_nth.append(0)
                    elif f'I0:{idx_expt}' in mcmc_trace.keys():
                        trace_I0 = mcmc_trace[f'I0:{idx_expt}'][: nsamples]
                        if n == 0: in_axis_nth.append(0)
                    else:
                        trace_I0 = None
                        if n == 0: in_axis_nth.append(None)
                    
                    log_likelihoods += _log_likelihood_each_enzyme('kinetics', data_rate, trace_nth, trace_nth, trace_alpha, trace_sigma, trace_error_E, trace_I0, in_axis_nth, nsamples)
        else:
            data_rate = expt['kinetics']
            plate = expt['plate']
            if data_rate is not None:
                if show_progress: 
                    print("Kinetics experiment", idx_expt)

                if f'alpha:{idx_expt}' in mcmc_trace.keys(): #multiple alpha for each experiment
                    trace_alpha = mcmc_trace[f'alpha:{idx_expt}'][: nsamples]
                elif f'alpha:{plate}' in mcmc_trace.keys(): #shared alpha among experiments with same plate
                    trace_alpha = mcmc_trace[f'alpha:{plate}'][: nsamples]
                elif alpha_list is not None and f'alpha:{plate}' in alpha_list: #provided alpha list for multiple plates
                    trace_alpha = jnp.one(nsamples)*alpha_list[f'alpha:{plate}'] 
                else:
                    trace_alpha = jnp.ones(nsamples)
                in_axis_nth.append(0)

                if f'log_sigma:{idx_expt}' in mcmc_trace.keys():
                    trace_sigma = jnp.exp(mcmc_trace[f'log_sigma:{idx_expt}'][: nsamples])
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

                if f'I0:{idx_expt}' in mcmc_trace.keys():
                    trace_I0 = mcmc_trace[f'I0:{idx_expt}'][: nsamples]
                    in_axis_nth.append(0)
                else:
                    trace_I0 = None
                    in_axis_nth.append(None)

                log_likelihoods += _log_likelihood_each_enzyme(type_expt='kinetics', data=data_rate, 
                                                               trace_logK=trace_nth, trace_kcat=trace_nth, 
                                                               trace_alpha=trace_alpha, trace_sigma=trace_sigma,
                                                               trace_error_E=trace_error_E, trace_I0=trace_I0, 
                                                               in_axes_nth=in_axis_nth, nsamples=nsamples)
    return np.array(log_likelihoods)


def map_finding(mcmc_trace, experiments, prior_infor, alpha_list=None, E_list=None,
                nsamples=None, set_lognormal_dE=False, dE=0.1, dI=0.1, set_K_S_DI_equal_K_S_DS=False, 
                set_K_S_DS_equal_K_S_D=False, show_progress=True):
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

    if show_progress:
        print("Calculing log of priors.")
    log_priors = _log_priors(mcmc_trace=mcmc_trace, experiments=experiments, prior_infor=prior_infor, nsamples=nsamples, 
                             set_lognormal_dE=set_lognormal_dE, dE=dE, dI=dI)

    if show_progress:
        print("Calculing log likelihoods:")
    mcmc_trace_update = _map_adjust_trace(mcmc_trace=mcmc_trace.copy(), experiments=experiments, prior_infor=prior_infor,
                                          set_K_I_M_equal_K_S_M=False, set_K_S_DS_equal_K_S_D=set_K_S_DS_equal_K_S_D,
                                          set_K_S_DI_equal_K_S_DS=set_K_S_DI_equal_K_S_DS, set_kcat_DSS_equal_kcat_DS=False, 
                                          set_kcat_DSI_equal_kcat_DS=False, set_kcat_DSI_equal_kcat_DSS=False, 
                                          show_progress=show_progress)
    log_likelihoods = _log_likelihoods(mcmc_trace=mcmc_trace_update, experiments=experiments, alpha_list=alpha_list, E_list=E_list, 
                                       nsamples=nsamples, show_progress=show_progress)
    
    log_probs = log_priors + log_likelihoods
    # map_idx = np.argmax(log_probs)
    map_idx = np.nanargmax(log_probs)
    if show_progress:
        print("Map index: %d" % map_idx)

    map_params = {}
    for name in mcmc_trace.keys():
        map_params[name] = mcmc_trace[name][map_idx]

=======
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy import stats
import pandas as pd
import numpyro.distributions as dist

from _prior_distribution import logsigma_guesses
from _load_prior_csv import _prior_group_name
from _kinetics import DimerBindingModel, Enzyme_Substrate

from _MAP_finding import _extract_logK_kcat, _lognormal_pdf, _log_normal_likelihood, _map_adjust_trace
from _MAP_finding_mers import _log_priors
from _model_mers import _dE_find_prior


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


def _ReactionRate_uncertainty_conc(logMtot, logStot, logItot,
                                   logKd, logK_S_M, logK_S_D, logK_S_DS,
                                   logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                   kcat_MS=0., kcat_DS=0., kcat_DSI=0., kcat_DSS=0.,
                                   error_E=None, I0=None):
    """
    Similar to function kinetics_.ReactionRate, this function return the reaction rate 
    given the parameters and adjusted enzyme/ligand concentration
        v = kcat_MS*[MS] + kcat_DS*[DS] + kcat_DSS*[DSS] + kcat_DSI*[DSI]
    
    Parameters
    ----------
    logMtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd : float
        Log of the dissociation constant of dimerization
    logKd_MS_M: float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M: float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS: float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI: float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    logKsp   : float
        Log of the dissociation constant for an aqueous inhibitor aggregating into solid state
    kcat_MS: float
        Rate constant of monomer-substrate complex
    kcat_DS  : float
        Rate constant of dimer-substrate complex
    kcat_DSS : float
        Rate constant of dimer-substrate-substrate complex
    kcat_DSI : float
        Rate constant of dimer-substrate-inhibitor complex
    error_E  : float
        Enzyme concentration uncertainty
    I0       : float
        The adjusted value of the highest concentration returned by the model in normal scale (M)

    All dissociation constants are in units of log molar 
    """
    if kcat_MS is None: kcat_MS = 0.
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.

    if error_E is None: logM = logMtot
    # else: logM = _extract_conc_percent_error(logMtot, error_E)
    else: logM = jnp.log(error_E*1E-9) 
    
    if I0 is None: logI = logItot
    else: logI = _extract_conc_lognormal(logItot, I0)
    
    if logItot is None:
        log_concs = Enzyme_Substrate(logM, logStot, logKd, logK_S_M, logK_S_D, logK_S_DS)
        v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    else:
        log_concs = DimerBindingModel(logM, logStot, logI,
                                      logKd, logK_S_M, logK_S_D, logK_S_DS,
                                      logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
        v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v


def _log_likelihood_each_enzyme(type_expt, data, trace_logK, trace_kcat, trace_alpha, 
                                trace_sigma, trace_error_E, trace_I0, in_axes_nth, nsamples):
    """
    Parameters:
    ----------
    type_expt     : str, 'kinetics', 'AUC', or 'ICE'
    data          : list, each dataset contains response, logMtot, lotStot, logItot
    trace_logK    : trace of all logK
    trace_logK    : trace of all kcat
    trace_sigma   : trace of log_sigma
    ----------
    Return log likelihood given the experiment, mcmc_trace, enzyme/ligand concentration uncertainty
    """
    assert type_expt in ['kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, or ICE."
    log_likelihoods = jnp.zeros(nsamples, dtype=jnp.float64)

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        f = vmap(lambda logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, alpha, sigma, error_E, I0: _log_normal_likelihood(rate,
                                                                                                                                                                                                _ReactionRate_uncertainty_conc(kinetics_logMtot, kinetics_logStot, kinetics_logItot,
                                                                                                                                                                                                                               logKd, logK_S_M, logK_S_D, logK_S_DS,
                                                                                                                                                                                                                               logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                                                                                                                                                                                                                               kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, error_E, I0)*alpha,
                                                                                                                                                                                                sigma),
                 in_axes=list(in_axes_nth))
        log_likelihoods += f(trace_logK['logKd'], trace_logK['logK_S_M'], trace_logK['logK_S_D'],
                             trace_logK['logK_S_DS'], trace_logK['logK_I_M'], trace_logK['logK_I_D'],
                             trace_logK['logK_I_DI'], trace_logK['logK_S_DI'],
                             trace_kcat['kcat_MS'], trace_kcat['kcat_DS'], trace_kcat['kcat_DSI'], trace_kcat['kcat_DSS'],
                             trace_alpha, trace_sigma, trace_error_E, trace_I0)
    return log_likelihoods


def _log_likelihoods(mcmc_trace, experiments, alpha_list=None, E_list=None, nsamples=None, show_progress=True):
    """
    Sum of log likelihood of all parameters given their distribution information in params_dist

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
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
        in_axis_nth = []
        try: idx_expt = expt['index']
        except: idx_expt = idx

        trace_nth, in_axis_nth = _extract_logK_kcat(mcmc_trace, idx, nsamples)

        if type(expt['kinetics']) is dict:
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None:
                    if show_progress: 
                        print("Kinetics experiment", idx_expt, ":", n)
                    
                    if f'alpha:{idx_expt}:{n}' in mcmc_trace.keys():
                        trace_alpha = mcmc_trace[f'alpha:{idx_expt}:{n}'][: nsamples]
                    elif f'alpha:{idx_expt}' in mcmc_trace.keys():
                        trace_alpha = mcmc_trace[f'alpha:{idx_expt}'][: nsamples]
                    elif alpha_list is not None and f'alpha:{idx_expt}:{n}' in alpha_list:
                        trace_alpha = jnp.one(nsamples)*alpha_list[f'alpha:{idx_expt}:{n}']
                    elif alpha_list is not None and f'alpha:{idx_expt}' in alpha_list:
                        trace_alpha = jnp.one(nsamples)*alpha_list[f'alpha:{idx_expt}']
                    else:
                        trace_alpha = jnp.ones(nsamples)
                    if n == 0: in_axis_nth.append(0) #index of vmap for alpha

                    if f'log_sigma:{idx_expt}:{n}' in mcmc_trace.keys():
                        trace_sigma = jnp.exp(mcmc_trace[f'log_sigma:{idx_expt}:{n}'][: nsamples])
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

                    if f'I0:{idx_expt}:{n}' in mcmc_trace.keys():
                        trace_I0 = mcmc_trace[f'I0:{idx_expt}:{n}'][: nsamples]
                        if n == 0: in_axis_nth.append(0)
                    elif f'I0:{idx_expt}' in mcmc_trace.keys():
                        trace_I0 = mcmc_trace[f'I0:{idx_expt}'][: nsamples]
                        if n == 0: in_axis_nth.append(0)
                    else:
                        trace_I0 = None
                        if n == 0: in_axis_nth.append(None)
                    
                    log_likelihoods += _log_likelihood_each_enzyme('kinetics', data_rate, trace_nth, trace_nth, trace_alpha, trace_sigma, trace_error_E, trace_I0, in_axis_nth, nsamples)
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                if show_progress: 
                    print("Kinetics experiment", idx_expt)

                if f'alpha:{idx_expt}' in mcmc_trace.keys():
                    trace_alpha = mcmc_trace[f'alpha:{idx_expt}'][: nsamples]
                elif alpha_list is not None and f'alpha:{idx_expt}' in alpha_list:
                    trace_alpha = jnp.one(nsamples)*alpha_list[f'alpha:{idx_expt}']
                else:
                    trace_alpha = jnp.ones(nsamples)
                in_axis_nth.append(0)

                if f'log_sigma:{idx_expt}' in mcmc_trace.keys():
                    trace_sigma = jnp.exp(mcmc_trace[f'log_sigma:{idx_expt}'][: nsamples])
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

                if f'I0:{idx_expt}' in mcmc_trace.keys():
                    trace_I0 = mcmc_trace[f'I0:{idx_expt}'][: nsamples]
                    in_axis_nth.append(0)
                else:
                    trace_I0 = None
                    in_axis_nth.append(None)

                log_likelihoods += _log_likelihood_each_enzyme(type_expt='kinetics', data=data_rate, 
                                                               trace_logK=trace_nth, trace_kcat=trace_nth, 
                                                               trace_alpha=trace_alpha, trace_sigma=trace_sigma,
                                                               trace_error_E=trace_error_E, trace_I0=trace_I0, 
                                                               in_axes_nth=in_axis_nth, nsamples=nsamples)

    return np.array(log_likelihoods)


def map_finding(mcmc_trace, experiments, prior_infor, alpha_list=None, E_list=None,
                nsamples=None, set_lognormal_dE=False, dE=0.1, dI=0.1, set_K_S_DI_equal_K_S_DS=False, 
                set_K_S_DS_equal_K_S_D=False, show_progress=True):
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
    log_priors = _log_priors(mcmc_trace=mcmc_trace, experiments=experiments, prior_infor=prior_infor, nsamples=nsamples, 
                             set_lognormal_dE=set_lognormal_dE, dE=dE, dI=dI)

    print("Calculing log likelihoods:")
    mcmc_trace_update = _map_adjust_trace(mcmc_trace=mcmc_trace.copy(), experiments=experiments, prior_infor=prior_infor,
                                          set_K_I_M_equal_K_S_M=False, set_K_S_DS_equal_K_S_D=set_K_S_DS_equal_K_S_D,
                                          set_K_S_DI_equal_K_S_DS=set_K_S_DI_equal_K_S_DS, set_kcat_DSS_equal_kcat_DS=False, 
                                          set_kcat_DSI_equal_kcat_DS=False, set_kcat_DSI_equal_kcat_DSS=False)
    log_likelihoods = _log_likelihoods(mcmc_trace=mcmc_trace_update, experiments=experiments, alpha_list=alpha_list, E_list=E_list, 
                                       nsamples=nsamples, show_progress=show_progress)
    
    log_probs = log_priors + log_likelihoods
    # map_idx = np.argmax(log_probs)
    map_idx = np.nanargmax(log_probs)
    print("Map index: %d" % map_idx)

    map_params = {}
    for name in mcmc_trace.keys():
        map_params[name] = mcmc_trace[name][map_idx]

>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
    return [map_idx, map_params, log_probs]
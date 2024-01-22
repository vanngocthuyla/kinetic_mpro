import jax
import jax.numpy as jnp
import numpy as np

from _prior_distribution import uniform_prior, lognormal_prior


def _dE_priors(experiments, dE, prior_type='lognormal'):
    """
    Parameters:
    ----------
    experiments : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).
    dE            : upper bound for the uniform prior of enzyme uncertainty if uniform distrubtion
                    or uncertainty of enzyme concentration of lognormal prior
    ----------
    Return list of enzyme uncertainty
    """
    error_E_list = {}
    _all_logMtot = []
    for idx, expt in enumerate(experiments):
        if type(expt['kinetics']) is dict:
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                _all_logMtot.append(data_rate[1])
        else:
            data_rate = expt['kinetics']
            _all_logMtot.append(data_rate[1])
    _all_logMtot = np.unique(np.concatenate(_all_logMtot))

    for _logConc in _all_logMtot:
        name = str(round(np.exp(_logConc)*1E9))
        if name not in error_E_list.keys():
            if prior_type == 'lognormal':
                error_E_list[f'dE:{name}'] = lognormal_prior(f'dE:{name}', jnp.exp(_logConc)*1E9, dE*jnp.exp(_logConc)*1E9)
            else:
                error_E_list[f'dE:{name}'] = uniform_prior(f'dE:{name}', 0, 1.)
    return error_E_list


def _dE_find_prior(data, error_E_list, prior_type='lognormal'):
    """
    Parameters:
    ----------
    data          : list, each dataset contains response, logMtot, lotStot, logItot
    error_E_list  : list of percentage error for enzyme concentration
    ----------
    Return array of prior information for enzyme concentration
    """
    [rate, logMtot, logStot, logItot] = data
    error_E = []
    for _logConc in logMtot:
        name = str(round(np.exp(_logConc)*1E9))
        if f'dE:{name}' in error_E_list.keys():
            error_E.append(error_E_list[f'dE:{name}'])
    return jnp.array(error_E)


# def _extract_conc_percent_error(logConc, error):
#     """
#     Parameters
#     ----------
#     logConc     : numpy array, concenntration of a species
#     error       : float, the adjusted value of the highest concentration returned by the model

#     Return the array of adjusted concentration given the percentage of error
#     """
#     if error is not None:
#         return logConc+jnp.log(1-error) #percent error
#     else:
#         return logConc

# def _prior_conc_lognormal(logConc, error=0.1, name='error'):
#     """
#     Parameters:
#     ----------
#     logConc       : concentration of a species in natural log scale
#     error         : percentage of uncertainty for log normal distribution
#     name          : name of the prior
#     ----------
#     Return adjusted log concentration
#     """ 
#     if error != 0:
#         stated_value = jnp.exp(jnp.max(logConc))
#         uncertainty = stated_value*error
#         expt_value = lognormal_prior(name, stated_value, uncertainty)
#         ratio = expt_value/stated_value
#         return logConc + jnp.log(ratio)
#     else:
#         return logConc
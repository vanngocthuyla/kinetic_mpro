import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
import numpy as np

from _kinetics_adjustable import Adjustable_ReactionRate, Adjustable_MonomerConcentration, Adjustable_CatalyticEfficiency
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _prior_check import check_prior_group, prior_group_multi_enzyme, define_uniform_prior_group


def adjustable_global_fitting(experiments, prior_infor=None, 
                              logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).
    prior_infor : list of dict to assign prior distribution for kinetics parameters
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each enzyme
    """
    n_enzymes = len(experiments)

    # Define priors
    if prior_infor is None:
        init_prior_infor = define_uniform_prior_group(logKd_min, logKd_max, kcat_min, kcat_max)
        prior_infor = check_prior_group(init_prior_infor, n_enzymes)
    params_logK, params_kcat = prior_group_multi_enzyme(prior_infor, n_enzymes)

    for idx, expt in enumerate(experiments):
        try:
            idx_expt = expt['index']
        except:
            idx_expt = idx

        [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI] = extract_logK_n_idx(params_logK, idx)
        [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS] = extract_kcat_n_idx(params_kcat, idx)
    
        if type(expt['kinetics']) is dict: 
            for n in range(len(expt['kinetics'])):
                data_rate = expt['kinetics'][n]
                if data_rate is not None:
                    adjustable_fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                                    f'rate:{idx_expt}:{n}', f'log_sigma_rate:{idx_expt}:{n}')
        else:
            data_rate = expt['kinetics']
            if data_rate is not None:
                adjustable_fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                                f'rate:{idx_expt}', f'log_sigma_rate:{idx_expt}')
        
        if type(expt['AUC']) is dict: 
            for n in range(len(expt['AUC'])):
                data_AUC = expt['AUC'][n]
                if data_AUC is not None: 
                    adjustable_fitting_each_dataset('AUC', data_AUC, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI], 
                                                    f'auc:{idx_expt}:{n}', f'log_sigma_auc:{idx_expt}:{n}')
        else:            
            data_AUC = expt['AUC']
            if data_AUC is not None: 
                adjustable_fitting_each_dataset('AUC', data_AUC, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI], 
                                                f'auc:{idx_expt}', f'log_sigma_auc:{idx_expt}')

        if type(expt['ICE']) is dict:
            for n in range(len(expt['ICE'])):
                data_ice = expt['ICE'][n]
                if data_ice is not None: 
                    adjustable_fitting_each_dataset('ICE', data_ice, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                                    f'ice:{idx_expt}:{n}', f'log_sigma_ice:{idx_expt}:{n}')
        else:
            data_ice = expt['ICE']
            if data_ice is not None: 
                adjustable_fitting_each_dataset('ICE', data_ice, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                                                f'ice:{idx_expt}', f'log_sigma_ice:{idx_expt}')


def adjustable_fitting_each_dataset(type_expt, data, params, name_response, name_log_sigma):
    """
    Parameters:
    ----------
    type_expt     : str, 'kinetics', 'AUC', or 'ICE'
    data          : list, each dataset contains response, logMtot, lotStot, logItot
    params        : list of kinetics parameters
    name_reponse  : str, name of posterior
    name_log_sigma: str, name of log_sigma for each dataset
    ----------
    Return likelihood from data and run the Bayesian model using given prior information of parameters
    """
    assert type_expt in ['kinetics', 'AUC', 'ICE'], "Experiments type should be kinetics, AUC, or ICE."

    if type_expt == 'kinetics':
        [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot] = data
        rate_model = Adjustable_ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, *params)
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
        log_sigma_rate = uniform_prior(name_log_sigma, lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma_rate = jnp.exp(log_sigma_rate)
        numpyro.sample(name_response, dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)

    if type_expt == 'AUC':
        [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data
        auc_model = Adjustable_MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot, *params)
        log_sigma_auc_min, log_sigma_auc_max = logsigma_guesses(auc) 
        log_sigma_auc = uniform_prior(name_log_sigma, lower=log_sigma_auc_min, upper=log_sigma_auc_max)
        sigma_auc = jnp.exp(log_sigma_auc)
        
        numpyro.sample(name_response, dist.Normal(loc=auc_model, scale=sigma_auc), obs=auc)

    if type_expt == 'ICE':
        [ice, ice_logMtot, ice_logStot, ice_logItot] = data
        ice_model = 1./Adjustable_CatalyticEfficiency(ice_logMtot, ice_logItot, *params)
        log_sigma_ice_min, log_sigma_ice_max = logsigma_guesses(ice) 
        log_sigma_ice = uniform_prior(name_log_sigma, lower=log_sigma_ice_min, upper=log_sigma_ice_max)
        sigma_ice = jnp.exp(log_sigma_ice)
        numpyro.sample(name_response, dist.Normal(loc=ice_model, scale=sigma_ice), obs=ice)
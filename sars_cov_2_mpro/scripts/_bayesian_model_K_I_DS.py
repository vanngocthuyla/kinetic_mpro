import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform

import jax
import jax.numpy as jnp
from jax import random, vmap
import jax.random as random

from scipy.optimize import fsolve, least_squares
from _chemical_reactions import ChemicalReactions
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses


@jax.jit
def DimerBindingModel(logMtot, logStot, logItot,
                      logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS):
    """
    Compute equilibrium concentrations for a binding model in which a ligand and substrate 
    competitively binds to a monomer, dimer, or dimer complexed with a ligand.
    
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
    
    All dissociation constants are in units of log molar 
    """
    dtype = jnp.float64
    logMtot = logMtot.astype(dtype)  # promote to dtype
    logStot = logStot.astype(dtype)  # promote to dtype
    logItot = logItot.astype(dtype)  # promote to dtype

    species = ['M','D','I','S','MS','MI','DI','DII','DS','DSI','DSS']
    # relationships between dissociation constants due to loops 
    logKd_MS_M = logKd + logK_S_D - logK_S_M
    logKd_MI_M = logKd + logK_I_D - logK_I_M
    logK_S_DI = logK_I_DS + logK_S_D - logK_I_D

    reactions = [{'M':2, 'D':-1},               # Dimerization
                 {'MS':1, 'M':1, 'DS': -1},
                 {'M': 1, 'S':1, 'MS': -1},     # Substrate binding
                 {'D':1, 'S':1, 'DS': -1},                 
                 {'DS':1, 'S':1, 'DSS':-1},
                 {'MI':1, 'M':1, 'DI': -1},
                 {'M':1, 'I':1, 'MI': -1},      # Inhibitor binding
                 {'D':1, 'I':1, 'DI':-1},                
                 {'DI':1, 'I':1, 'DII':-1},             
                 {'DS':1, 'I':1, 'DSI':-1},     # Substrate and inhibitor binding                  
                 {'DI':1, 'S':1, 'DSI':-1}
                 ]
    conservation_equations = [{'M':+1,'D':+2,'MS':+1,'MI':+1,'DI':+2,'DII':+2,'DS':+2,'DSI': +2,'DSS':+2}, # Total protein
                              {'S':+1, 'MS':+1, 'DS':+1, 'DSI':+1, 'DSS':+2},  # Total substrate
                              {'I':+1, 'MI':+1, 'DI':+1, 'DII':+2, 'DSI':+1} # Total ligand
                             ]
    binding_model = ChemicalReactions(reactions, conservation_equations)
    f_log_c = vmap(lambda logM, logS, logI: binding_model.logceq(jnp.array([logKd, logKd_MS_M, logK_S_M, logK_S_D, logK_S_DS, logKd_MI_M, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS, logK_S_DI]), 
                                                                 jnp.array([logM, logS, logI])))
    log_c = f_log_c(logMtot, logStot, logItot).T
    sorted_species = sorted(['M','D','I','S','MS','MI','DI','DII','DS','DSI','DSS'])
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])
    return log_concs


@jax.jit
def Enzyme_Substrate(logMtot, logStot, logKd, logK_S_M, logK_S_D, logK_S_DS):
    """
    Compute equilibrium concentrations of species for a binding model of an enzyme and a substrate

    Parameters:
    ----------
    logMtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot : numpy array
        Log of the total substrate concentation summed over bound and unbound species
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
    
    All dissociation constants are in units of log molar 
    ----------

    """
    logMtot = logMtot.astype(jnp.float64)  # promote to dtype
    logStot = logStot.astype(jnp.float64)  # promote to dtype

    # relationships between dissociation constants due to loops 
    logKd_MS_M = logKd + logK_S_D - logK_S_M

    reactions = [{'M':2, 'D':-1},               # Dimerization
                 {'MS':1, 'M':1, 'DS': -1},
                 {'M': 1, 'S':1, 'MS': -1},     # Substrate binding
                 {'D':1, 'S':1, 'DS': -1},                 
                 {'DS':1, 'S':1, 'DSS':-1}
                 ]
    conservation_equations = [{'M':+1,'D':+2,'MS':+1,'DS':+2,'DSS':+2}, # Total protein
                              {'S':+1, 'MS':+1, 'DS':+1, 'DSS':+2},  # Total substrate
                             ]
    binding_model = ChemicalReactions(reactions, conservation_equations)
    f_log_c = vmap(lambda logM, logS: binding_model.logceq(jnp.array([logKd, logKd_MS_M, logK_S_M, logK_S_D, logK_S_DS]), 
                                                           jnp.array([logM, logS])))
    log_c = f_log_c(logMtot, logStot).T
    sorted_species = sorted(['M','D','S','MS','DS','DSS'])
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])

    species_full = sorted(['M','D','I','S','MS','MI','DI','DII','DS','DSI','DSS'])
    log_concs_full = dict([(key, jnp.log(jnp.ones(logMtot.shape[0], jnp.float64)*1E-30)) for key in species_full])
    for key in sorted_species:
        log_concs_full[key] = log_concs[key]
         
    return log_concs_full

@jax.jit
def Enzyme_Inhibitor(logMtot, logItot, logKd, logK_I_M, logK_I_D, logK_I_DI):
    """
    Compute equilibrium concentrations of species for a binding model of an enzyme and a inhibitor

    Parameters:
    ----------
    logMtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logItot : numpy array
        Log of the total inhibitor concentation summed over bound and unbound species
    logKd : float
        Log of the dissociation constant of dimerization
    logKd_MI_M: float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    
    All dissociation constants are in units of log molar 
    ----------

    """
    logMtot = logMtot.astype(jnp.float64)  # promote to dtype
    logItot = logItot.astype(jnp.float64)  # promote to dtype

    # relationships between dissociation constants due to loops 
    logKd_MI_M = logKd + logK_I_D - logK_I_M

    reactions = [{'M':2, 'D':-1},               # Dimerization
                 {'MI':1, 'M':1, 'DI': -1},
                 {'M': 1, 'I':1, 'MI': -1},     # Substrate binding
                 {'D':1, 'I':1, 'DI': -1},                 
                 {'DI':1, 'I':1, 'DII':-1}
                 ]
    conservation_equations = [{'M':+1,'D':+2, 'MI':+1, 'DI':+2,'DII':+2},   # Total protein
                              {'I':+1,'MI':+1,'DI':+1,'DII':+2},       # Total substrate
                             ]
    binding_model = ChemicalReactions(reactions, conservation_equations)
    f_log_c = vmap(lambda logM, logI: binding_model.logceq(jnp.array([logKd, logKd_MI_M, logK_I_M, logK_I_D, logK_I_DI]), 
                                                           jnp.array([logM, logI])))
    log_c = f_log_c(logMtot, logItot).T
    sorted_species = sorted(['M','D','I','MI','DI','DII'])
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])

    species_full = sorted(['M','D','I','S','MS','MI','DI','DII','DS','DSI','DSS'])
    log_concs_full = dict([(key, jnp.log(jnp.ones(logMtot.shape[0], jnp.float64)*1E-30)) for key in species_full])
    for key in sorted_species:
        log_concs_full[key] = log_concs[key]
         
    return log_concs_full


def ReactionRate(logMtot, logStot, logItot, 
                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS,
                 kcat_MS=0., kcat_DS=0., kcat_DSI=1., kcat_DSS=1.):
    """
    Reaction Rate
      v = kcat_MS*[MS] + kcat_+DS*[DS] + kcat_DSI*[DSI] + kcat_DSS*[DSS]
    
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
    kcat_MS: float
        Rate constant of monomer-substrate complex
    kcat_DS: float
        Rate constant of dimer-substrate complex
    kcat_DSI: float
        Rate constant of dimer-substrate-inhibitor complex
    kcat_DSS: float
        Rate constant of dimer-substrate-substrate complex
    All dissociation constants are in units of log molar 
    """
    if logItot is None: 
        log_concs = Enzyme_Substrate(logMtot, logStot, logKd, logK_S_M, logK_S_D, logK_S_DS)
    elif logStot is None:
        log_concs = Enzyme_Inhibitor(logMtot, logItot, logKd, logK_I_M, logK_I_D, logK_I_DI)
    else: 
        log_concs = DimerBindingModel(logMtot, logStot, logItot, 
                                      logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                      logK_I_M, logK_I_D, logK_I_DI, logK_I_DS)
    v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v


def MonomerConcentration(logMtot, logStot, logItot, logKd, logK_S_M, logK_S_D, logK_S_DS, 
                         logK_I_M, logK_I_D, logK_I_DI, logK_I_DS):
    """
    Response of MonomerConcentration ~ [M] + [MI] + [MI]
    
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
    
    All dissociation constants are in units of log molar 
    """
    if logItot is None: 
        log_concs = Enzyme_Substrate(logMtot, logStot, logKd, logK_S_M, logK_S_D, logK_S_DS)
    elif logStot is None:
        log_concs = Enzyme_Inhibitor(logMtot, logItot, logKd, logK_I_M, logK_I_D, logK_I_DI)
    else: 
        log_concs = DimerBindingModel(logMtot, logStot, logItot, 
                                      logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                      logK_I_M, logK_I_D, logK_I_DI, logK_I_DS)
    M = jnp.exp(log_concs['M']) + jnp.exp(log_concs['MI']) + jnp.exp(log_concs['MS'])
    return M


def CatalyticEfficiency(logMtot, logItot,
                        logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS,
                        kcat_MS=0., kcat_DS=0., kcat_DSI=1., kcat_DSS=1., 
                        logStot = None):
    """
    kcat/Km, based on the finite difference derivative
    
    Parameters
    ----------
    logMtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
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
    kcat_MS: float
        Rate constant of monomer-substrate complex
    kcat_DS: float
        Rate constant of dimer-substrate complex
    kcat_DSI: float
        Rate constant of dimer-substrate-inhibitor complex
    kcat_DSS: float
        Rate constant of dimer-substrate-substrate complex

    All dissociation constants are in units of log molar 
    """
    if logStot is None:
        logStot = jnp.log(jnp.array([1, 2])*1E-6)
    DeltaS = (jnp.exp(logStot[1])-jnp.exp(logStot[0]))

    catalytic_efficiency = jnp.zeros(logItot.shape, jnp.float32)
    v1 = ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[0], logItot,
                      logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS, 
                      kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
    v2 = ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[1], logItot,
                      logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS, 
                      kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
    catalytic_efficiency = (v2-v1)/DeltaS
    return catalytic_efficiency


def prior_group_informative(logKd_min = -20, logKd_max = 0, kcat_min=0, kcat_max=1):

    logKd = normal_prior('logKd', -5, 3)
    # Substrate binding
    logK_S_M = uniform_prior('logK_S_M', logKd_min, logKd_max)
    logK_S_D = uniform_prior('logK_S_D', logKd_min, logKd_max)
    logK_S_DS = uniform_prior('logK_S_DS', logKd_min, logKd_max)
    # Inhibitor binding
    logK_I_M = uniform_prior('logK_I_M', logKd_min, logKd_max)
    logK_I_D = normal_prior('logK_I_D', -13, 3)
    logK_I_DI = uniform_prior('logK_I_DI', logKd_min, logKd_max)
    # Binding of substrate and inhibitor
    logK_I_DS = normal_prior('logK_I_DS', -9, 3)
    # kcat
    kcat_MS = 0
    kcat_DS = 0
    kcat_DSI = uniform_prior('kcat_DSI', kcat_min, kcat_max)
    kcat_DSS = uniform_prior('kcat_DSS', kcat_min, kcat_max)
    
    return logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS


def fitting_each_dataset(type_expt, data, params, name_response, name_log_sigma):
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
        rate_model = ReactionRate(kinetics_logMtot, kinetics_logStot, kinetics_logItot, *params)
        log_sigma_rate_min, log_sigma_rate_max = logsigma_guesses(rate) 
        log_sigma_rate = uniform_prior(name_log_sigma, lower=log_sigma_rate_min, upper=log_sigma_rate_max)
        sigma_rate = jnp.exp(log_sigma_rate)
        numpyro.sample(name_response, dist.Normal(loc=rate_model, scale=sigma_rate), obs=rate)

    if type_expt == 'AUC':
        [auc, AUC_logMtot, AUC_logStot, AUC_logItot] = data
        auc_model = MonomerConcentration(AUC_logMtot, AUC_logStot, AUC_logItot, *params)
        log_sigma_auc_min, log_sigma_auc_max = logsigma_guesses(auc) 
        log_sigma_auc = uniform_prior(name_log_sigma, lower=log_sigma_auc_min, upper=log_sigma_auc_max)
        sigma_auc = jnp.exp(log_sigma_auc)
        
        numpyro.sample(name_response, dist.Normal(loc=auc_model, scale=sigma_auc), obs=auc)

    if type_expt == 'ICE':
        [ice, ice_logMtot, ice_logStot, ice_logItot] = data
        ice_model = 1./CatalyticEfficiency(ice_logMtot, ice_logItot, *params)
        log_sigma_ice_min, log_sigma_ice_max = logsigma_guesses(ice) 
        log_sigma_ice = uniform_prior(name_log_sigma, lower=log_sigma_ice_min, upper=log_sigma_ice_max)
        sigma_ice = jnp.exp(log_sigma_ice)
        numpyro.sample(name_response, dist.Normal(loc=ice_model, scale=sigma_ice), obs=ice)


def global_fitting(data_rate, data_AUC=None, data_ice=None, logKd_min=-20, logKd_max=0, kcat_min=0, kcat_max=1):
    """
    Parameters:
    ----------
    experiments : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    logKd_min   : float, lower values of uniform distribution for prior of dissociation constants
    logKd_max   : float, upper values of uniform distribution for prior of dissociation constants
    kcat_min    : float, lower values of uniform distribution for prior of kcat
    kcat_max    : float, upper values of uniform distribution for prior of kcat
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of each dataset
    """
    # Define priors
    logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS = prior_group_informative(logKd_min, logKd_max, kcat_min, kcat_max)

    if data_rate is not None: 
        fitting_each_dataset('kinetics', data_rate, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                             'rate', 'log_sigma_rate')
    
    if data_AUC is not None: 
        fitting_each_dataset('AUC', data_AUC, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS], 
                             'auc', 'log_sigma_auc')

    if data_ice is not None: 
        fitting_each_dataset('ICE', data_ice, [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_I_DS, kcat_MS, kcat_DS, kcat_DSI, kcat_DSS], 
                             'ice', 'log_sigma_ice')
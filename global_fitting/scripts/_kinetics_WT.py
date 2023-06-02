
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from scipy.optimize import fsolve, least_squares

from _chemical_reactions import ChemicalReactions


@jax.jit
def Enzyme_Substrate_WT(logDtot, logStot, logK_S_D, logK_S_DS):
    """
    Compute equilibrium concentrations of species for a binding model of an enzyme (dimer) and a substrate

    Parameters:
    ----------
    logDtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logK_S_D : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    
    All dissociation constants are in units of log molar 
    ----------

    """
    dtype = jnp.float64
    logDtot = logDtot.astype(dtype)  # promote to dtype
    logStot = logStot.astype(dtype)  # promote to dtype

    species = ['D','S','DS','DSS']
    reactions = [{'D':1, 'S':1, 'DS': -1},
                 {'S':1, 'DS':1, 'DSS':-1},
                ]
    conservation_equations = [{'D':+1, 'DS':+1, 'DSS':+1}, # Total protein
                              {'S':+1, 'DS':+1, 'DSS':+2}, # Total substrate
                             ]
    binding_model = ChemicalReactions(reactions, conservation_equations)
    f_log_c = vmap(lambda logD, logS: binding_model.logceq(jnp.array([logK_S_D, logK_S_DS]), 
                                                           jnp.array([logD, logS])))
    log_c = f_log_c(logDtot, logStot).T
    sorted_species = sorted(['D','S','DS','DSS'])
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])

    species_full = sorted(['D','I','S','DI','DII','DS','DSI','DSS'])
    log_concs_full = dict([(key, jnp.log(jnp.ones(logDtot.shape[0], jnp.float64)*1E-30)) for key in species_full])
    for key in sorted_species:
        log_concs_full[key] = log_concs[key]

    return log_concs_full


@jax.jit
def DimerBindingModel_WT(logDtot, logStot, logItot,
                         logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI):
    """
    Compute equilibrium concentrations for a binding model in which a ligand and substrate 
    competitively binds to a monomer, dimer, or dimer complexed with a ligand.
    
    Parameters
    ----------
    logDtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logK_S_D : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
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
    logDtot = logDtot.astype(dtype)  # promote to dtype
    logStot = logStot.astype(dtype)  # promote to dtype
    logItot = logItot.astype(dtype)  # promote to dtype

    species = ['D','I','S','DI','DII','DS','DSI','DSS']
    # relationships between dissociation constants due to loops 
    logK_I_DS = logK_I_D + logK_S_DI - logK_S_D

    reactions = [{'D':1, 'S':1, 'DS': -1},                 
                 {'DS':1, 'S':1, 'DSS':-1},
                 {'D':1, 'I':1, 'DI':-1},                
                 {'DI':1, 'I':1, 'DII':-1},             
                 {'DS':1, 'I':1, 'DSI':-1},     # Substrate and inhibitor binding                  
                 {'DI':1, 'S':1, 'DSI':-1}
                 ]
    conservation_equations = [{'D':+1,'DI':+1,'DII':+1, 'DS':+1,'DSI': +1,'DSS':+1}, # Total protein
                              {'S':+1,'DS':+1,'DSI':+1,'DSS':+2}, # Total substrate
                              {'I':+1,'DI':+1,'DII':+2,'DSI':+1} # Total ligand
                             ]
    binding_model = ChemicalReactions(reactions, conservation_equations)
    f_log_c = vmap(lambda logD, logS, logI: binding_model.logceq(jnp.array([logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_I_DS, logK_S_DI]), 
                                                                 jnp.array([logD, logS, logI])))
    log_c = f_log_c(logDtot, logStot, logItot).T
    sorted_species = sorted(['D','I','S','DI','DII','DS','DSI','DSS'])
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])
    return log_concs


def ReactionRate_WT(logDtot, logStot, logItot,
                    logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI, 
                    kcat_DS=0., kcat_DSS=1.):
    """
    Reaction Rate
      v = kcat_+DS*[DS] + kcat_DSS*[DSS]
    
    Parameters
    ----------
    logDtot : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logK_S_D : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    kcat_DS: float
        Rate constant of dimer-substrate complex
    kcat_DSS: float
        Rate constant of dimer-substrate-substrate complex
    
    All dissociation constants are in units of log molar 
    """
    if logItot is None:
        log_concs = Enzyme_Substrate_WT(logDtot, logStot, logK_S_D, logK_S_DS)
    else:
        log_concs = DimerBindingModel_WT(logDtot, logStot, logItot,
                                         logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI)
    v = kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v
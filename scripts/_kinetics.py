import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from scipy.optimize import fsolve, least_squares

from _chemical_reactions import ChemicalReactions


@jax.jit
def DimerBindingModel(logMtot, logStot, logItot,
                      logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI):
    """
    Compute equilibrium concentrations for a binding model in which a ligand and substrate 
    competitively binds to a monomer, dimer, or dimer complexed with a ligand.
    
    Parameters
    ----------
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
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
    logK_I_DS = logK_I_D + logK_S_DI - logK_S_D

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
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
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
                              {'S':+1, 'MS':+1, 'DS':+1, 'DSS':+2},     # Total substrate
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
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total inhibitor concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
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
                              {'I':+1,'MI':+1,'DI':+1,'DII':+2},            # Total inhibitor
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


@jax.jit
def ReactionRate(logMtot, logStot, logItot, 
                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                 kcat_MS=0., kcat_DS=0., kcat_DSI=1., kcat_DSS=1.):
    """
    Reaction Rate
      v = kcat_MS*[MS] + kcat_+DS*[DS] + kcat_DSI*[DSI] + kcat_DSS*[DSS]
    
    Parameters
    ----------
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    kcat_MS     : float
        Rate constant of monomer-substrate complex
    kcat_DS     : float
        Rate constant of dimer-substrate complex
    kcat_DSI    : float
        Rate constant of dimer-substrate-inhibitor complex
    kcat_DSS    : float
        Rate constant of dimer-substrate-substrate complex

    All dissociation constants are in units of log molar 
    """
    if kcat_MS is None: kcat_MS = 0.
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.
    if logItot is None: 
        print("Fitting ES model.")
        log_concs = Enzyme_Substrate(logMtot, logStot, logKd, logK_S_M, logK_S_D, logK_S_DS)
    elif logStot is None:
        print("Fitting EI model.")
        log_concs = Enzyme_Inhibitor(logMtot, logItot, logKd, logK_I_M, logK_I_D, logK_I_DI)
    else: 
        print("Fitting ESI model.")
        log_concs = DimerBindingModel(logMtot, logStot, logItot, 
                                      logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                      logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
    v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v


@jax.jit
def MonomerConcentration(logMtot, logStot, logItot, logKd, logK_S_M, logK_S_D, logK_S_DS, 
                         logK_I_M, logK_I_D, logK_I_DI, logK_S_DI):
    """
    Response of MonomerConcentration ~ [M] + [MI] + [MI]
    
    Parameters
    ----------
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
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
                                      logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
    M = jnp.exp(log_concs['M']) + jnp.exp(log_concs['MI']) + jnp.exp(log_concs['MS'])
    return M


@jax.jit
def CatalyticEfficiency(logMtot, logItot,
                        logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                        kcat_MS=0., kcat_DS=0., kcat_DSI=1., kcat_DSS=1., 
                        logStot = None):
    """
    kcat/Km, based on the finite difference derivative
    
    Parameters
    ----------
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    kcat_MS     : float
        Rate constant of monomer-substrate complex
    kcat_DS     : float
        Rate constant of dimer-substrate complex
    kcat_DSI    : float
        Rate constant of dimer-substrate-inhibitor complex
    kcat_DSS    : float
        Rate constant of dimer-substrate-substrate complex

    All dissociation constants are in units of log molar 
    """
    if logStot is None:
        logStot = jnp.log(jnp.array([1, 2])*1E-6)
    DeltaS = (jnp.exp(logStot[1])-jnp.exp(logStot[0]))

    catalytic_efficiency = jnp.zeros(logItot.shape, jnp.float32)
    v1 = ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[0], logItot,
                      logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                      kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
    v2 = ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[1], logItot,
                      logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                      kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
    catalytic_efficiency = (v2-v1)/DeltaS
    return catalytic_efficiency


def Dimerization(logMtot, logKd):
    """
    Parameters:
    ----------
    logMtot : numpy array
        Log of the total protein concentation summed over bound and unbound species (units of log molar)
    logKd   : float
        Log of the dissociation constant of dimerization (units of log molar)
    ----------
    Return  : equilibrium concentrations of species for dimerization of protein
    """
    Mtot = jnp.exp(logMtot)
    Kd = jnp.exp(logKd)
    x = (4*Mtot + Kd - jnp.sqrt(Kd**2 + 8*Mtot*Kd))/8
    log_concs = {}
    log_concs['D'] = np.log(x)
    log_concs['M'] = np.log(Mtot - 2*x)

    return log_concs

## Dimer-only model -------------------------------------------------------------------------------------- ##

@jax.jit
def DimerOnlyModel(logDtot, logStot, logItot,
                   logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI):
    """
    Compute equilibrium concentrations for a binding model in which a ligand and substrate
    competitively binds to a dimer, or dimer complexed with a ligand.

    Parameters
    ----------
    logDtot     : numpy array
        Log of the total dimer concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
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
    conservation_equations = [{'D':+1,'DI':+1,'DII':+1, 'DS':+1,'DSI': +1,'DSS':+1}, # Total dimer
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


@jax.jit
def ReactionRate_DimerOnly(logDtot, logStot, logItot,
                           logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI,
                           kcat_DS=0., kcat_DSI=1., kcat_DSS=1.):
    """
    Reaction Rate
      v = kcat_+DS*[DS] + kcat_DSI*[DSI] + kcat_DSS*[DSS]
    
    Parameters
    ----------
    logDtot     : numpy array
        Log of the total dimer concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species      
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    kcat_DS     : float
        Rate constant of dimer-substrate complex
    kcat_DSI    : float
        Rate constant of dimer-substrate-inhibitor complex
    kcat_DSS    : float
        Rate constant of dimer-substrate-substrate complex
    
    All dissociation constants are in units of log molar 
    """
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.
    log_concs = DimerOnlyModel(logDtot, logStot, logItot,
                               logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI)
    v = kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v

## Adjustable model -------------------------------------------------------------------------------------- ##

@jax.jit
def adjust_DimerBindingModel(logMtot, logStot, logItot,
                             logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI):
    """
    Compute equilibrium concentrations for a binding model in which a ligand and substrate 
    competitively binds to a monomer, dimer, or dimer complexed with a ligand.
    
    Parameters
    ----------
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex

    All dissociation constants are in units of log molar 
    """
    dtype = jnp.float64

    species, reactions, params = define_species_reactions(logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                                          logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
    conservation_equations = define_conserved_equations(species)
    
    # If there is only enzyme
    if (conservation_equations[1] is None) and (conservation_equations[2] is None):
        logMtot = logMtot.astype(dtype)
        binding_model = ChemicalReactions(reactions, [conservation_equations[0]])
        f_log_c = vmap(lambda logM: binding_model.logceq(jnp.array(params), jnp.array([logM])))
        log_c = f_log_c(logMtot).T
    # If there are enzyme and substrate    
    elif conservation_equations[2] is None:
        logMtot = logMtot.astype(dtype)
        logStot = logStot.astype(dtype)
        binding_model = ChemicalReactions(reactions, [conservation_equations[0], conservation_equations[1]])
        f_log_c = vmap(lambda logM, logS: binding_model.logceq(jnp.array(params), jnp.array([logM, logS])))
        log_c = f_log_c(logMtot, logStot).T
    # If there are enzyme and inhibitor
    elif conservation_equations[1] is None:
        logMtot = logMtot.astype(dtype)
        logItot = logItot.astype(dtype)
        binding_model = ChemicalReactions(reactions, [conservation_equations[0], conservation_equations[2]])
        f_log_c = vmap(lambda logM, logI: binding_model.logceq(jnp.array(params), jnp.array([logM, logI])))
        log_c = f_log_c(logMtot, logItot).T
    else:
        logMtot = logMtot.astype(dtype)
        logStot = logStot.astype(dtype)
        logItot = logItot.astype(dtype)
        binding_model = ChemicalReactions(reactions, conservation_equations)
        f_log_c = vmap(lambda logM, logS, logI: binding_model.logceq(jnp.array(params), 
                                                                     jnp.array([logM, logS, logI])))
        log_c = f_log_c(logMtot, logStot, logItot).T
    
    sorted_species = sorted(species)
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])

    species_full = sorted(['M','D','S','I','MS','DS','DSS','MI','DI','DII','DSI'])
    log_concs_full = dict([(key, jnp.log(jnp.ones(logMtot.shape[0], jnp.float64)*1E-25)) for key in species_full])
    for key in sorted_species:
        log_concs_full[key] = log_concs[key]

    return log_concs_full


def adjust_ReactionRate(logMtot, logStot, logItot, 
                        logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                        kcat_MS=0., kcat_DS=0., kcat_DSI=0., kcat_DSS=0.):
    """
    Reaction Rate
      v = kcat_MS*[MS] + kcat_DS*[DS] + kcat_DSS*[DSS] + kcat_DSI*[DSI]
    
    Parameters
    ----------
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    kcat_MS     : float
        Rate constant of monomer-substrate complex
    kcat_DS     : float
        Rate constant of dimer-substrate complex
    kcat_DSI    : float
        Rate constant of dimer-substrate-inhibitor complex
    kcat_DSS    : float
        Rate constant of dimer-substrate-substrate complex

    All dissociation constants are in units of log molar 
    """
    if kcat_MS is None: kcat_MS = 0.
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.
    log_concs = adjust_DimerBindingModel(logMtot, logStot, logItot, 
                                         logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                         logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
    v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v


def adjust_MonomerConcentration(logMtot, logStot, logItot, 
                                logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI):
    """
    Response of MonomerConcentration ~ [M] + [MI] + [MI]
    
    Parameters
    ----------
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex

    All dissociation constants are in units of log molar 
    """
    log_concs = adjust_DimerBindingModel(logMtot, logStot, logItot, 
                                         logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                         logK_I_M, logK_I_D, logK_I_DI, logK_S_DI)
    M = jnp.exp(log_concs['M']) + jnp.exp(log_concs['MI']) + jnp.exp(log_concs['MS'])
    return M


def adjust_CatalyticEfficiency(logMtot, logItot, 
                               logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                               kcat_MS=0., kcat_DS=0., kcat_DSI=0., kcat_DSS=0., logStot = None):
    """
    kcat/Km, based on the finite difference derivative
    
    Parameters
    ----------
    logMtot     : numpy array
        Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array
        Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array
        Log of the total ligand concentation summed over bound and unbound species
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    kcat_MS     : float
        Rate constant of monomer-substrate complex
    kcat_DS     : float
        Rate constant of dimer-substrate complex
    kcat_DSI    : float
        Rate constant of dimer-substrate-inhibitor complex
    kcat_DSS    : float
        Rate constant of dimer-substrate-substrate complex

    All dissociation constants are in units of log molar 
    """
    if logStot is None:
        logStot = jnp.log(jnp.array([1, 2])*1E-6)
    DeltaS = (jnp.exp(logStot[1])-jnp.exp(logStot[0]))

    catalytic_efficiency = jnp.zeros(logItot.shape, jnp.float32)
    v1 = adjust_ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[0], logItot,
                             logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                             kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
    v2 = adjust_ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[1], logItot,
                             logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, 
                             kcat_MS, kcat_DS, kcat_DSI, kcat_DSS)
    catalytic_efficiency = (v2-v1)/DeltaS
    return catalytic_efficiency


def define_species_reactions(logKd=None, logK_S_M=None, logK_S_D=None, logK_S_DS=None, 
                             logK_I_M=None, logK_I_D=None, logK_I_DI=None, logK_S_DI=None):
    """
    Define the species and reactions of dimer binding model in which a ligand and substrate 
    competitively binds to a monomer, dimer, or dimer complexed with a ligand.
    
    Parameters
    ----------
    logKd       : float
        Log of the dissociation constant of dimerization
    logKd_MS_M  : float
        Log of the dissociation constant between the monomer and substrate-monomer complex
    logK_S_M    : float
        Log of the dissociation constant between the substrate and free monomer
    logK_S_D    : float
        Log of the dissociation constant between the substrate and free dimer
    logK_S_DS   : float
        Log of the dissociation constant between the substrate and ligand-dimer complex
    logKd_MI_M  : float
        Log of the dissociation constant between the monomer and ligand-monomer complex
    logK_I_M    : float 
        Log of the dissociation constant between the inhibitor and free monomer
    logK_I_D    : float
        Log of the dissociation constant between the inhibitor and free dimer
    logK_I_DI   : float
        Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
    logK_I_DS   : float
        Log of the dissociation constant between the inhibitor and substrate-dimer complex
    logK_S_DI   : float
        Log of the dissociation constant between the substrate and inhibitor-dimer complex
    
    All dissociation constants are in units of log molar 
    """
    params = []
    species = []
    reactions = []
    if logKd is not None: 
        params.append(logKd)
        species.append(['M', 'D'])
        reactions.append({'M':2, 'D':-1})
    if logK_S_M is not None: 
        params.append(logK_S_M)
        species.append(['S', 'M', 'MS'])
        reactions.append({'M': 1, 'S':1, 'MS': -1})
    if (logKd is not None) and (logK_S_M is not None) and (logK_S_D is not None):
        logKd_MS_M = logKd + logK_S_D - logK_S_M
        params.append(logKd_MS_M)
        reactions.append({'MS':1, 'M':1, 'DS': -1})
    if logK_S_D is not None:
        params.append(logK_S_D)
        species.append(['S', 'D', 'DS'])
        reactions.append({'D':1, 'S':1, 'DS': -1})
    if logK_S_DS is not None: 
        params.append(logK_S_DS)
        species.append(['S', 'DS', 'DSS'])
        reactions.append({'DS':1, 'S':1, 'DSS':-1})

    if logK_I_M is not None:
        params.append(logK_I_M) 
        species.append(['I', 'M', 'MI'])
        reactions.append({'M':1, 'I':1, 'MI': -1})
    if (logKd is not None) and (logK_I_M is not None) and (logK_I_D is not None):
        logKd_MI_M = logKd + logK_I_D - logK_I_M
        params.append(logKd_MI_M)
        reactions.append({'MI':1, 'M':1, 'DI': -1})
    if logK_I_D is not None: 
        params.append(logK_I_D)
        species.append(['I', 'D', 'DI'])
        reactions.append({'D':1, 'I':1, 'DI':-1})
    if logK_I_DI is not None:
        params.append(logK_I_DI)
        species.append(['I', 'DI', 'DII'])
        reactions.append({'DI':1, 'I':1, 'DII':-1})

    if logK_S_DI is not None: 
        params.append(logK_S_DI)
        species.append(['S', 'DI', 'DSI'])
        reactions.append({'DI':1, 'S':1, 'DSI':-1})

    if (logK_I_D is not None) and (logK_S_DI is not None) and (logK_S_D is not None):
        logK_I_DS = logK_I_D + logK_S_DI - logK_S_D
        params.append(logK_I_DS)
        reactions.append({'DS':1, 'I':1, 'DSI':-1})

    species = np.unique(np.concatenate(species))

    return species, reactions, params


def define_conserved_equations(species):
    """
    Define the conservation equations for dimer binding model given the species
    """
    conservation_equations = []
    M = {}
    S = {}
    I = {}
    for _species in species:
        if _species=='M': 
            M['M'] = 1
        if _species=='S': 
            S['S'] = 1
        if _species=='I': 
            I['I'] = 1
        if _species=='D': 
            M['D'] = 2
        if _species=='MS': 
            M['MS'] = 1
            S['MS'] = 1
        if _species=='MI': 
            M['MI'] = 1
            I['MI'] = 1
        if _species=='DS': 
            M['DS'] = 2
            S['DS'] = 1
        if _species=='DI': 
            M['DI'] = 2
            I['DI'] = 1
        if _species=='DSS': 
            M['DSS'] = 2
            S['DSS'] = 2
        if _species=='DII': 
            M['DII'] = 2
            I['DII'] = 2
        if _species=='DSI': 
            M['DSI'] = 2
            S['DSI'] = 1
            I['DSI'] = 1

    conservation_equations.append(M)
    if len(S)>0:
        conservation_equations.append(S)
    else:
        conservation_equations.append(None)
    if len(I)>0:
        conservation_equations.append(I)
    else:
        conservation_equations.append(None)
    return conservation_equations
<<<<<<< HEAD
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial

from _chemical_reactions import ChemicalReactions


@partial(jax.jit, static_argnames=['constraint_logK_S_DS', 'constraint_logK_I_M'])
def Adjustable_DimerBindingModel(logMtot, logStot, logItot,
                                 logKd, logK_S_M, logK_S_D, logK_S_DS,
                                 logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                 constraint_logK_S_DS=False, constraint_logK_I_M=False):
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
    logKsp   : float
        Log of the dissociation constant for an aqueous inhibitor aggregating into solid state

    All dissociation constants are in units of log molar
    """
    dtype = jnp.float64
    logMtot = logMtot.astype(dtype)  # promote to dtype
    logStot = logStot.astype(dtype)  # promote to dtype
    logItot = logItot.astype(dtype)  # promote to dtype

    species, reactions, params = define_species_reactions(logKd, logK_S_M, logK_S_D, logK_S_DS,
                                                          logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                                          constraint_logK_S_DS, constraint_logK_I_M)
    conservation_equations = define_conserved_equations(species)

    n_species = []
    for equations in conservation_equations:
        n_species.append(len(equations))
    n_species = np.array(n_species)
    index = np.where(n_species==0)[0]

    if len(index)==0:
        binding_model = ChemicalReactions(reactions, conservation_equations)
        f_log_c = vmap(lambda logM, logS, logI: binding_model.logceq(jnp.array(params),
                                                                     jnp.array([logM, logS, logI])))
        log_c = f_log_c(logMtot, logStot, logItot).T
    elif len(index) == 1:
        if index[0]==1:
            binding_model = ChemicalReactions(reactions, [conservation_equations[0], conservation_equations[2]])
            f_log_c = vmap(lambda logM, logI: binding_model.logceq(jnp.array(params), jnp.array([logM, logI])))
            log_c = f_log_c(logMtot, logItot).T
        else:
            binding_model = ChemicalReactions(reactions, [conservation_equations[0], conservation_equations[1]])
            f_log_c = vmap(lambda logM, logS: binding_model.logceq(jnp.array(params), jnp.array([logM, logS])))
            log_c = f_log_c(logMtot, logStot).T
    else:
        binding_model = ChemicalReactions(reactions, [conservation_equations[0]])
        f_log_c = vmap(lambda logM: binding_model.logceq(jnp.array(params), jnp.array([logM])))
        log_c = f_log_c(logMtot).T

    sorted_species = sorted(species)
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])

    species_full = sorted(['M','D','I','S','MS','MI','DI','DII','DS','DSI','DSS'])
    log_concs_full = dict([(key, jnp.log(jnp.ones(logMtot.shape[0], jnp.float64)*1E-30)) for key in species_full])
    for key in sorted_species:
        log_concs_full[key] = log_concs[key]

    return log_concs_full


def Adjustable_ReactionRate(logMtot, logStot, logItot,
                            logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                            kcat_MS=0., kcat_DS=0., kcat_DSI=0., kcat_DSS=0.,
                            constraint_logK_S_DS=False, constraint_logK_I_M=False):
    """
    Reaction Rate
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

    All dissociation constants are in units of log molar 
    """
    if kcat_MS is None: kcat_MS = 0.
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.
    log_concs = Adjustable_DimerBindingModel(logMtot, logStot, logItot,
                                             logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                             logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                             constraint_logK_S_DS, constraint_logK_I_M)
    v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v


def Adjustable_MonomerConcentration(logMtot, logStot, logItot, 
                                    logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                    logKsp, constraint_logK_S_DS=False, constraint_logK_I_M=False):
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
    logKsp   : float
        Log of the dissociation constant for an aqueous inhibitor aggregating into solid state

    All dissociation constants are in units of log molar 
    """
    log_concs = Adjustable_DimerBindingModel(logMtot, logStot, logItot, 
                                             logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                             logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                             logKsp, constraint_logK_S_DS, constraint_logK_I_M)
    M = jnp.exp(log_concs['M']) + jnp.exp(log_concs['MI']) + jnp.exp(log_concs['MS'])
    return M


def Adjustable_CatalyticEfficiency(logMtot, logItot,
                                   logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                   kcat_MS=0., kcat_DS=0., kcat_DSI=1., kcat_DSS=1.,
                                   logStot=None, constraint_logK_S_DS=False, constraint_logK_I_M=False):
    """
    kcat/Km, based on the finite difference derivative
    
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

    All dissociation constants are in units of log molar 
    """
    if logStot is None:
        logStot = jnp.log(jnp.array([1, 2])*1E-6)
    DeltaS = (jnp.exp(logStot[1])-jnp.exp(logStot[0]))

    catalytic_efficiency = jnp.zeros(logItot.shape, jnp.float32)
    v1 = Adjustable_ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[0], logItot,
                                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                 kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, 
                                 constraint_logK_S_DS, constraint_logK_I_M)
    v2 = Adjustable_ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[1], logItot,
                                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                 kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, 
                                 constraint_logK_S_DS, constraint_logK_I_M)
    catalytic_efficiency = (v2-v1)/DeltaS
    return catalytic_efficiency


def define_species_reactions(logKd=None, logK_S_M=None, logK_S_D=None, logK_S_DS=None,
                             logK_I_M=None, logK_I_D=None, logK_I_DI=None, logK_S_DI=None, logKsp=None,
                             constraint_logK_S_DS=False, constraint_logK_I_M=False):
    """
    Define the species and reactions of dimer binding model in which a ligand and substrate
    competitively binds to a monomer, dimer, or dimer complexed with a ligand.

    Parameters
    ----------
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
    if logKd is not None and logK_S_M is not None and logK_S_D is not None:
        logKd_MS_M = logKd + logK_S_D - logK_S_M
        params.append(logKd_MS_M)
        reactions.append({'MS':1, 'M':1, 'DS': -1})
    if logK_S_D is not None:
        params.append(logK_S_D)
        species.append(['S', 'D', 'DS'])
        reactions.append({'D':1, 'S':1, 'DS': -1})

    if logK_I_M is not None:
        if constraint_logK_I_M and logK_I_D is not None and logK_S_M is not None and logK_S_D is not None:
            logK_I_M = logK_I_D + logK_S_M - logK_S_D
            print("Constraint on logK_I_M")
        params.append(logK_I_M)
        species.append(['I', 'M', 'MI'])
        reactions.append({'M':1, 'I':1, 'MI': -1})
    if logKd is not None and logK_I_M is not None and logK_I_D is not None:
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
    if logK_I_D is not None and logK_S_DI is not None and logK_S_D is not None:
        logK_I_DS = logK_I_D + logK_S_DI - logK_S_D
        params.append(logK_I_DS)
        reactions.append({'DS':1, 'I':1, 'DSI':-1})

    if logK_S_DS is not None:
        if constraint_logK_S_DS and logK_I_DS is not None and logK_I_D is not None and logK_S_DI is not None:
            logK_S_DS = logK_I_DS + logK_I_D - logK_S_DI
            print("Constraint on logK_S_DS")
        params.append(logK_S_DS)
        species.append(['S', 'DS', 'DSS'])
        reactions.append({'DS':1, 'S':1, 'DSS':-1})

    if logKsp is not None:
        params.append(logKsp)
        species.append(['I'])
        reactions.append({'I':1})

    species = np.unique(np.concatenate(species))
    return species, reactions, params


def define_conserved_equations(species):
    """
    Define the conservation equations for dimer binding model given the species
    """
    conservation_equations = []
    D = {}
    S = {}
    I = {}
    for _species in species:
        if _species=='M': 
            D['M'] = 1
        if _species=='S': 
            S['S'] = 1
        if _species=='I': 
            I['I'] = 1
        if _species=='D': 
            D['D'] = 2
        if _species=='MS': 
            D['MS'] = 1
            S['MS'] = 1
        if _species=='MI': 
            D['MI'] = 1
            I['MI'] = 1
        if _species=='DS': 
            D['DS'] = 2
            S['DS'] = 1
        if _species=='DI': 
            D['DI'] = 2
            I['DI'] = 1
        if _species=='DSS': 
            D['DSS'] = 2
            S['DSS'] = 2
        if _species=='DII': 
            D['DII'] = 2
            I['DII'] = 2
        if _species=='DSI': 
            D['DSI'] = 2
            S['DSI'] = 1
            I['DSI'] = 1

    conservation_equations.append(D)
    conservation_equations.append(S)
    conservation_equations.append(I)
=======
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial

from _chemical_reactions import ChemicalReactions


@partial(jax.jit, static_argnames=['constraint_logK_S_DS', 'constraint_logK_I_M'])
def Adjustable_DimerBindingModel(logMtot, logStot, logItot,
                                 logKd, logK_S_M, logK_S_D, logK_S_DS,
                                 logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                 constraint_logK_S_DS=False, constraint_logK_I_M=False):
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
    logKsp   : float
        Log of the dissociation constant for an aqueous inhibitor aggregating into solid state

    All dissociation constants are in units of log molar
    """
    dtype = jnp.float64
    logMtot = logMtot.astype(dtype)  # promote to dtype
    logStot = logStot.astype(dtype)  # promote to dtype
    logItot = logItot.astype(dtype)  # promote to dtype

    species, reactions, params = define_species_reactions(logKd, logK_S_M, logK_S_D, logK_S_DS,
                                                          logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                                          constraint_logK_S_DS, constraint_logK_I_M)
    conservation_equations = define_conserved_equations(species)

    n_species = []
    for equations in conservation_equations:
        n_species.append(len(equations))
    n_species = np.array(n_species)
    index = np.where(n_species==0)[0]

    if len(index)==0:
        binding_model = ChemicalReactions(reactions, conservation_equations)
        f_log_c = vmap(lambda logM, logS, logI: binding_model.logceq(jnp.array(params),
                                                                     jnp.array([logM, logS, logI])))
        log_c = f_log_c(logMtot, logStot, logItot).T
    elif len(index) == 1:
        if index[0]==1:
            binding_model = ChemicalReactions(reactions, [conservation_equations[0], conservation_equations[2]])
            f_log_c = vmap(lambda logM, logI: binding_model.logceq(jnp.array(params), jnp.array([logM, logI])))
            log_c = f_log_c(logMtot, logItot).T
        else:
            binding_model = ChemicalReactions(reactions, [conservation_equations[0], conservation_equations[1]])
            f_log_c = vmap(lambda logM, logS: binding_model.logceq(jnp.array(params), jnp.array([logM, logS])))
            log_c = f_log_c(logMtot, logStot).T
    else:
        binding_model = ChemicalReactions(reactions, [conservation_equations[0]])
        f_log_c = vmap(lambda logM: binding_model.logceq(jnp.array(params), jnp.array([logM])))
        log_c = f_log_c(logMtot).T

    sorted_species = sorted(species)
    log_concs = dict([(key, log_c[n]) for n, key in enumerate(sorted_species)])

    species_full = sorted(['M','D','I','S','MS','MI','DI','DII','DS','DSI','DSS'])
    log_concs_full = dict([(key, jnp.log(jnp.ones(logMtot.shape[0], jnp.float64)*1E-30)) for key in species_full])
    for key in sorted_species:
        log_concs_full[key] = log_concs[key]

    return log_concs_full


def Adjustable_ReactionRate(logMtot, logStot, logItot,
                            logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                            kcat_MS=0., kcat_DS=0., kcat_DSI=0., kcat_DSS=0.,
                            constraint_logK_S_DS=False, constraint_logK_I_M=False):
    """
    Reaction Rate
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

    All dissociation constants are in units of log molar 
    """
    if kcat_MS is None: kcat_MS = 0.
    if kcat_DS is None: kcat_DS = 0.
    if kcat_DSI is None: kcat_DSI = 0.
    if kcat_DSS is None: kcat_DSS = 0.
    log_concs = Adjustable_DimerBindingModel(logMtot, logStot, logItot,
                                             logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                             logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                             constraint_logK_S_DS, constraint_logK_I_M)
    v = kcat_MS*jnp.exp(log_concs['MS']) + kcat_DS*jnp.exp(log_concs['DS']) + kcat_DSI*jnp.exp(log_concs['DSI']) + kcat_DSS*jnp.exp(log_concs['DSS'])
    return v


def Adjustable_MonomerConcentration(logMtot, logStot, logItot, 
                                    logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                    logKsp, constraint_logK_S_DS=False, constraint_logK_I_M=False):
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
    logKsp   : float
        Log of the dissociation constant for an aqueous inhibitor aggregating into solid state

    All dissociation constants are in units of log molar 
    """
    log_concs = Adjustable_DimerBindingModel(logMtot, logStot, logItot, 
                                             logKd, logK_S_M, logK_S_D, logK_S_DS, 
                                             logK_I_M, logK_I_D, logK_I_DI, logK_S_DI,
                                             logKsp, constraint_logK_S_DS, constraint_logK_I_M)
    M = jnp.exp(log_concs['M']) + jnp.exp(log_concs['MI']) + jnp.exp(log_concs['MS'])
    return M


def Adjustable_CatalyticEfficiency(logMtot, logItot,
                                   logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                   kcat_MS=0., kcat_DS=0., kcat_DSI=1., kcat_DSS=1.,
                                   logStot=None, constraint_logK_S_DS=False, constraint_logK_I_M=False):
    """
    kcat/Km, based on the finite difference derivative
    
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

    All dissociation constants are in units of log molar 
    """
    if logStot is None:
        logStot = jnp.log(jnp.array([1, 2])*1E-6)
    DeltaS = (jnp.exp(logStot[1])-jnp.exp(logStot[0]))

    catalytic_efficiency = jnp.zeros(logItot.shape, jnp.float32)
    v1 = Adjustable_ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[0], logItot,
                                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                 kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, 
                                 constraint_logK_S_DS, constraint_logK_I_M)
    v2 = Adjustable_ReactionRate(logMtot, jnp.ones(logItot.shape[0])*logStot[1], logItot,
                                 logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI, logKsp,
                                 kcat_MS, kcat_DS, kcat_DSI, kcat_DSS, 
                                 constraint_logK_S_DS, constraint_logK_I_M)
    catalytic_efficiency = (v2-v1)/DeltaS
    return catalytic_efficiency


def define_species_reactions(logKd=None, logK_S_M=None, logK_S_D=None, logK_S_DS=None,
                             logK_I_M=None, logK_I_D=None, logK_I_DI=None, logK_S_DI=None, logKsp=None,
                             constraint_logK_S_DS=False, constraint_logK_I_M=False):
    """
    Define the species and reactions of dimer binding model in which a ligand and substrate
    competitively binds to a monomer, dimer, or dimer complexed with a ligand.

    Parameters
    ----------
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
    if logKd is not None and logK_S_M is not None and logK_S_D is not None:
        logKd_MS_M = logKd + logK_S_D - logK_S_M
        params.append(logKd_MS_M)
        reactions.append({'MS':1, 'M':1, 'DS': -1})
    if logK_S_D is not None:
        params.append(logK_S_D)
        species.append(['S', 'D', 'DS'])
        reactions.append({'D':1, 'S':1, 'DS': -1})

    if logK_I_M is not None:
        if constraint_logK_I_M and logK_I_D is not None and logK_S_M is not None and logK_S_D is not None:
            logK_I_M = logK_I_D + logK_S_M - logK_S_D
            print("Constraint on logK_I_M")
        params.append(logK_I_M)
        species.append(['I', 'M', 'MI'])
        reactions.append({'M':1, 'I':1, 'MI': -1})
    if logKd is not None and logK_I_M is not None and logK_I_D is not None:
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
    if logK_I_D is not None and logK_S_DI is not None and logK_S_D is not None:
        logK_I_DS = logK_I_D + logK_S_DI - logK_S_D
        params.append(logK_I_DS)
        reactions.append({'DS':1, 'I':1, 'DSI':-1})

    if logK_S_DS is not None:
        if constraint_logK_S_DS and logK_I_DS is not None and logK_I_D is not None and logK_S_DI is not None:
            logK_S_DS = logK_I_DS + logK_I_D - logK_S_DI
            print("Constraint on logK_S_DS")
        params.append(logK_S_DS)
        species.append(['S', 'DS', 'DSS'])
        reactions.append({'DS':1, 'S':1, 'DSS':-1})

    if logKsp is not None:
        params.append(logKsp)
        species.append(['I'])
        reactions.append({'I':1})

    species = np.unique(np.concatenate(species))
    return species, reactions, params


def define_conserved_equations(species):
    """
    Define the conservation equations for dimer binding model given the species
    """
    conservation_equations = []
    D = {}
    S = {}
    I = {}
    for _species in species:
        if _species=='M': 
            D['M'] = 1
        if _species=='S': 
            S['S'] = 1
        if _species=='I': 
            I['I'] = 1
        if _species=='D': 
            D['D'] = 2
        if _species=='MS': 
            D['MS'] = 1
            S['MS'] = 1
        if _species=='MI': 
            D['MI'] = 1
            I['MI'] = 1
        if _species=='DS': 
            D['DS'] = 2
            S['DS'] = 1
        if _species=='DI': 
            D['DI'] = 2
            I['DI'] = 1
        if _species=='DSS': 
            D['DSS'] = 2
            S['DSS'] = 2
        if _species=='DII': 
            D['DII'] = 2
            I['DII'] = 2
        if _species=='DSI': 
            D['DSI'] = 2
            S['DSI'] = 1
            I['DSI'] = 1

    conservation_equations.append(D)
    conservation_equations.append(S)
    conservation_equations.append(I)
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
    return conservation_equations
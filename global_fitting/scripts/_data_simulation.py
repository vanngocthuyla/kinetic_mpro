import numpy as np

def kinetics_data_simulation(logMtot, logStot, logItot, params_logK, params_kcat, sigma,
                             type_experiment='kinetics', seed=None):
    """
    Parameters:
    ----------
    logMtot     : numpy array, Log of the total protein concentation summed over bound and unbound species
    logStot     : numpy array, Log of the total substrate concentation summed over bound and unbound species
    logItot     : numpy array, Log of the total ligand concentation summed over bound and unbound species
    params_logK : dict of all dissociation constants
        logKd       : float, Log of the dissociation constant of dimerization
        logKd_MS_M  :  float, Log of the dissociation constant between the monomer and substrate-monomer complex
        logK_S_M    : float, Log of the dissociation constant between the substrate and free monomer
        logK_S_D    : float, Log of the dissociation constant between the substrate and free dimer
        logK_S_DS   : float, Log of the dissociation constant between the substrate and ligand-dimer complex
        logKd_MI_M  : float, Log of the dissociation constant between the monomer and ligand-monomer complex
        logK_I_M    : float, Log of the dissociation constant between the inhibitor and free monomer
        logK_I_D    : float, Log of the dissociation constant between the inhibitor and free dimer
        logK_I_DI   : float, Log of the dissociation constant between the inhibitor and inhibitor-dimer complex
        logK_I_DS   : float, Log of the dissociation constant between the inhibitor and substrate-dimer complex
        logK_S_DI   : float, Log of the dissociation constant between the substrate and inhibitor-dimer complex
        All dissociation constants are in units of log molar
    params_kcat : dict of all kcat
        kcat_MS : float, Rate constant of monomer-substrate complex
        kcat_DS : float, Rate constant of dimer-substrate complex
        kcat_DSI: float, Rate constant of dimer-substrate-inhibitor complex
        kcat_DSS: float, Rate constant of dimer-substrate-substrate complex
    sigma       : float, noise of experiment
    type_experiment : string, optional, 'kinetics', 'AUC', or 'catalytic_efficiency'
    seed            : integer, optional, used to for random.seed
    ----------
    return simulated experiment
    """
    if seed==None: 
        np.random.seed(0)
    else: 
        np.random.seed(seed)

    total_n = len(logMtot)
    noise = np.random.normal(loc=0, scale=sigma, size=total_n)

    if type_experiment == 'kinetics': 
        response = ReactionRate(logMtot, logStot, logItot, *params_logK, *params_kcat) + noise
        response_name = 'v'
    if type_experiment == 'AUC': 
        response = MonomerConcentration(logMtot, logStot, logItot, *params_logK) + noise
        response_name = 'M'
    if type_experiment == 'catalytic_efficiency': 
        response = CatalyticEfficiency(logMtot, logItot, *params_logK, *params_kcat, logStot) + noise
        response_name = 'Km_over_kcat'

    experiment = {'type': type_experiment, 'logMtot': logMtot, 'logStot': logStot, 
                  'logItot': logItot, response_name: response, 'x':'logItot'}
    return experiment

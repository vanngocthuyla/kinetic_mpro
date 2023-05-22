import matplotlib.pyplot as plt
import numpy as np
import os

def plot_kinetics_data(experiments, params_logK, params_kcat, 
					   figure_size=(6.4, 4.8), dpi=80, OURDIR=None): 
    """
    Parameters:
    ----------
    experiments : list of dict
        Each dataset contains response, logMtot, lotStot, logItot
    params_logK : dict of all dissociation constants
        logKd       : float, Log of the dissociation constant of dimerization
        logKd_MS_M  : float, Log of the dissociation constant between the monomer and substrate-monomer complex
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
    figure_size     : (width, height) size of plot
    dpi             : quality of plot
    OURDIR          : optional, string, output directory for saving plot
    ----------
    return plots of each experiments
    """
    for n, experiment in enumerate(experiments):
        plt.figure(figsize=figure_size, dpi=dpi)
        try: 
            plt.title(f"{experiment['type']} from Fig. {experiment['figure']}")
        except: 
            pass
        x = experiment[experiment['x']]

        # Plot data
        if experiment['type']=='kinetics':
            plt.plot(x, experiment['v'], '.')
            plt.xlabel(experiment['x'])
            plt.ylabel('Rate (M min$^{-1}$)')
        elif experiment['type']=='AUC':
            plt.plot(x, experiment['M'], '.')
            plt.xlabel(experiment['x'])
            plt.ylabel('Monomer concentration (M)')
        elif experiment['type']=='catalytic_efficiency':
            plt.plot(x, experiment['Km_over_kcat'], '.')
            plt.xlabel(experiment['x'])
            plt.ylabel('K$_m$/k$_{cat}$')

        # Plot fit
        if experiment['x']=='logMtot':
            logMtot = np.linspace(experiment['logMtot'][0], experiment['logMtot'][-1], 50)
            logStot = experiment['logStot'][0]*np.ones(50)
            logItot = experiment['logItot'][0]*np.ones(50)
            x = logMtot
        elif experiment['x']=='logStot':
            logMtot = experiment['logMtot'][0]*np.ones(50)
            logStot = np.linspace(experiment['logStot'][0], experiment['logStot'][-1], 50)
            logItot = experiment['logItot'][0]*np.ones(50)
            x = logStot
        elif experiment['x']=='logItot':
            logMtot = experiment['logMtot'][0]*np.ones(50)
            logStot = experiment['logStot'][0]*np.ones(50)
            logItot = np.linspace(experiment['logItot'][0], experiment['logItot'][-1], 50)
            x = logItot

        if experiment['type']=='kinetics':
            y_model = ReactionRate(logMtot, logStot, logItot, *params_logK, *params_kcat)
        elif experiment['type']=='AUC':
            y_model = MonomerConcentration(logMtot, logStot, logItot, *params_logK)
        elif experiment['type']=='catalytic_efficiency':
            y_model = 1./CatalyticEfficiency(logMtot, logItot, *params_logK, *params_kcat, logStot)
        plt.plot(x, y_model)
        plt.tight_layout();

        if OURDIR is not None: 
            plt.savefig(os.path.join(OURDIR,f"{experiment['figure']}"))
            plt.ioff()
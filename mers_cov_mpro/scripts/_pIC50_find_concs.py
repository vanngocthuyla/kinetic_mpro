import numpy as np
import jax
import jax.numpy as jnp

import os
import pandas as pd
import pickle
import arviz as az

from _pIC50 import _pIC50_hill, _adjust_trace


def _table_pIC50_hill_find_conc(inhibitor_list, mcmc_dir, logDtot, logStot, logItot, logK_dE_alpha=None):
    """
    For a set of inhibitors, dimer-only pIC50s can be simulated given the specified values of 
    dimer/substrate concentrations and kinetic parameters from mcmc trace. 

    Parameters:
    ----------
    inhibitor_list  : list of inhibitor
    mcmc_dir        : str, directory of traces
    logDtot         : vector of dimer concentration
    logStot         : vector of substrate concentration
    logItot         : vector of inhibitor concentration
    logK_dE_alpha   : dict of information about fixed logK, dE and alpha
    ----------
    return table of kinetic parameters, pIC50, and hill slope for the whole dataset of multiple inhibitors
    """
    def f_table(inhibitor):
        
        inhibitor_dir = inhibitor[7:12]
        inhibitor_name = inhibitor[:12]

        if not os.path.isfile(os.path.join(mcmc_dir, inhibitor_dir, 'traces.pickle')):
            return None

        trace = pickle.load(open(os.path.join(mcmc_dir, inhibitor_dir, 'traces.pickle'), "rb"))
        data = az.InferenceData.to_dataframe(az.convert_to_inference_data(trace))

        nthin = int(len(data)/100)
        if logK_dE_alpha is not None:
            df = _adjust_trace(data.iloc[::nthin, :].copy(), logK_dE_alpha)
        else:
            df = data.iloc[::nthin, :].copy()

        pIC50_list, hill_list = _pIC50_hill(df, logDtot, logStot, logItot)

        df.insert(len(df.columns), 'pIC50', pIC50_list)
        df.insert(len(df.columns), 'hill', hill_list)

        df_inhibitor = df[['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI', 'pIC50', 'hill']]
        df_inhibitor = df_inhibitor[df_inhibitor.hill>0]
        
        return np.median(df_inhibitor.pIC50), np.std(df_inhibitor.pIC50), np.median(df_inhibitor.hill), np.std(df_inhibitor.hill)

    list_median_std = np.array(list(map(f_table, inhibitor_list)))
    pIC50_median = list_median_std.T[0]
    pIC50_std = list_median_std.T[1]
    hill_median = list_median_std.T[2]
    hill_std = list_median_std.T[3]

    table_mean = pd.DataFrame([inhibitor_list, pIC50_median, hill_median], index=['ID', 'pIC50', 'hill'])
    table_std = pd.DataFrame([inhibitor_list, pIC50_std, hill_std], index=['ID', 'pIC50_std', 'hill_std'])        

    table = pd.merge(table_mean.T, table_std.T, on='ID')
    return table
"""
Before estimating the pIC50 for a list of inhibitor giving their traces.pickle files in mcmc_dir, 
the concentrations of dimer and substrate can be predicted to decrease the difference between the 
cellular pEC50 and the dimer-only pIC50.
"""

import warnings
import numpy as np
import sys
import os
import itertools
import argparse

import pickle
import arviz as az
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import jax
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

from _pIC50 import _pd_mean_std_pIC, _correct_ID, table_pIC_hill_multi_inhibitor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

parser = argparse.ArgumentParser()

parser.add_argument( "--inhibitor_file",                type=str,               default="")
parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--logK_dE_alpha_file",            type=str,               default="")

parser.add_argument( "--cellular_pIC50_file",           type=str,               default="")

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--exclude_experiments",           type=str,               default="")

args = parser.parse_args()

df_mers = pd.read_csv(args.inhibitor_file)
_inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

exclude_experiments = args.exclude_experiments.split()
inhibitor_list = [name for name in _inhibitor_list if name[:12] not in exclude_experiments and os.path.isfile(os.path.join(args.mcmc_dir, name[7:12], 'traces.pickle'))]

if len(args.logK_dE_alpha_file)>0 and os.path.isfile(args.logK_dE_alpha_file):
    logK_dE_alpha = pickle.load(open(args.logK_dE_alpha_file, "rb"))
    
    if args.set_K_S_DS_equal_K_S_D:
        logK_dE_alpha['logK_S_DS'] = logK_dE_alpha['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        logK_dE_alpha['logK_S_DI'] = logK_dE_alpha['logK_S_DS']

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'kcat_DS', 'kcat_DSS']:
        assert key in logK_dE_alpha.keys(), f"Please provide {key} in logK_dE_alpha_file."
else:
    logK_dE_alpha = None

## Extracting cellular pIC50
_df_cell = pd.read_csv(args.cellular_pIC50_file)
_df_cell = _df_cell.rename(columns={'Molecule Name': "ID", 'IC50 (uM)': 'IC50'})
_df_cell.insert(len(_df_cell.columns), 'cell_pIC50', -np.log10(_df_cell.IC50*1E-6))
df_cell = _correct_ID(_pd_mean_std_pIC(_df_cell, 'cell_pIC50'))


def f_find_conc(init_logconc, inhibitor_list, mcmc_dir, n_points = 50, min_conc_I=1E-12, max_conc_I=1E-3):
    """
    For a set of inhibitors, dimer-only pIC50s can be simulated given the specified values of 
    dimer/substrate concentrations and kinetic parameters from mcmc trace.
    
    This function calculated the deviation between dimer-only pIC50s and cellular pIC50s.
    
    Parameters:
    ----------
    init_logconc    : initial concentrations under ln scale.
    inhibitor_list  : list of inhibitor
    mcmc_dir        : str, directory of traces
    n_points        : numer of datapoints to simulate concentration-response curve
    min_conc_I      : float, minimum value of inhibitor concentration
    max_conc_I      : float, maximum value of inhibitor concentration
    ----------

    """
    logDtot = np.ones(n_points)*(init_logconc-np.log(2))
    logStot = np.ones(n_points)*(init_logconc+np.log(1E3))
    logItot = np.linspace(np.log(min_conc_I), np.log(max_conc_I), n_points)

    table = table_pIC_hill_multi_inhibitor(inhibitor_list=inhibitor_list, mcmc_dir=mcmc_dir,
                                           logDtot=logDtot, logStot=logStot, logItot=logItot,
                                           measure='median', logK_dE_alpha=logK_dE_alpha)
    if table is None:
        return 0
    else:
        dat = pd.merge(table[['ID', 'pIC50']], df_cell[['ID', 'cell_pIC50']], on='ID', how='inner')
        print(dat.head())
        RMSD = np.sqrt(np.mean((dat.pIC50 - dat.cell_pIC50)**2))
        print("RMSD =", round(RMSD, 4))
        return RMSD


res = minimize(f_find_conc, x0=np.log(1*1E-6), method='COBYLA', tol=1E-6,
               bounds=((-21, -3),), args=(inhibitor_list, args.mcmc_dir))
init_logconc = res.x[0]

init_logDtot = init_logconc-np.log(2)
init_logStot = init_logconc+np.log(1E3)

print('Enzyme (dimer) under ln scale: %.3f' %init_logDtot)
print('Substrate under ln scale: %.3f ' %init_logStot)

with open(os.path.join(args.out_dir, "concentrations.txt"), "w") as f:

    print('\nEnzyme (dimer) under ln scale: %.3f' %init_logDtot, file=f)
    print('Substrate under ln scale: %.3f ' %init_logStot, file=f)

    print('Enzyme (dimer): %.3f (uM)' %(np.exp(init_logDtot)*1E6), file=f)
    print('Substrate : %.3f (uM)' %(np.exp(init_logStot)*1E6), file=f)
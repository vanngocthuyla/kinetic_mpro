import warnings
import numpy as np
import sys
import os
from glob import glob
import argparse

import pickle
import arviz as az
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import jax
import jax.numpy as jnp

from _pIC50 import _table_pIC50_hill
from _pIC50_find_concs import _table_pIC50_hill_find_conc

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

mcmc_dir = glob(os.path.join(args.mcmc_dir, "*"), recursive = True)
mcmc_dir = [os.path.basename(f) for f in mcmc_dir if os.path.isdir(f)]

df_mers = pd.read_csv(args.inhibitor_file)
_inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

exclude_experiments = args.exclude_experiments.split()
inhibitor_list = [name for name in _inhibitor_list if name[:12] not in exclude_experiments and name[7:12] in mcmc_dir]

if len(args.logK_dE_alpha_file)>0 and os.path.isfile(args.logK_dE_alpha_file):
    logK_dE_alpha = pickle.load(open(args.logK_dE_alpha_file, "rb"))
    
    if args.set_K_S_DS_equal_K_S_D:
        logK_dE_alpha['logK_S_DS'] = logK_dE_alpha['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        logK_dE_alpha['logK_S_DI'] = logK_dE_alpha['logK_S_DS']

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'kcat_DS', 'kcat_DSS', 'dE:100', 'dE:50', 'dE:25']:
        assert key in logK_dE_alpha.keys(), f"Please provide {key} in logK_dE_alpha_file."
else:
    logK_dE_alpha = None

## Extracting cellular pIC50
def _pd_mean_std(df, name):
    ID = np.unique(df['ID'])
    mean = []
    std = []
    for _ID in ID:
        mean.append(df[name][df.ID == _ID].mean())
        std.append(df[name][df.ID == _ID].std())
    if 'Experiment' in df.columns:
        expt = np.repeat(np.unique(df.Experiment), len(ID))
        return pd.DataFrame([expt, ID, mean, std], index=['Experiment', 'ID', name, name+'_std']).T
    return pd.DataFrame([ID, mean, std], index=['ID', name, name+'_std']).T

_df_cell = pd.read_csv(args.cellular_pIC50_file)
_df_cell = _df_cell.rename(columns={'Molecule Name': "ID", 'Vero-76_MERS-CoV-antiviral-effect_usu: EC50 (Num) (uM)': 'IC50'})
_df_cell.insert(len(_df_cell.columns), 'cell_pIC50', -np.log10(_df_cell.IC50*1E-6))

_df_cell = _df_cell[['Experiment', 'ID', 'cell_pIC50']]

name = 'cell_pIC50'
df_cell = pd.DataFrame(index=['Experiment', 'ID', name, name+'_std']).T
for expt in np.unique(_df_cell.Experiment):
    _df = _pd_mean_std(_df_cell[_df_cell.Experiment==expt], name)
    df_cell = pd.concat([df_cell, _df])

## Finding concentrations that return the lowest deviation between cellular and dimer-only pIC50s

def f_find_conc(init_conc, inhibitor_list, mcmc_dir, n_points = 50, min_conc_I=1E-12, max_conc_I=1E-3):
    """
    For a set of inhibitors, dimer-only pIC50s can be simulated given the specified values of 
    dimer/substrate concentrations and kinetic parameters from mcmc trace.
    
    This function calculated the deviation between dimer-only pIC50s and cellular pIC50s.
    
    Parameters:
    ----------
    init_conc       : list of initial concentrations of dimer and substrate under ln scale.
    inhibitor_list  : list of inhibitor
    mcmc_dir        : str, directory of traces
    n_points        : numer of datapoints to simulate concentration-response curve
    min_conc_I      : float, minimum value of inhibitor concentration
    max_conc_I      : float, maximum value of inhibitor concentration
    ----------

    """
    [init_logDtot, init_logStot] = init_conc

    logDtot = np.ones(n_points)*init_logDtot
    logStot = np.ones(n_points)*init_logStot
    logItot = np.linspace(np.log(min_conc_I), np.log(max_conc_I), n_points)

    table = _table_pIC50_hill_find_conc(inhibitor_list, mcmc_dir, logDtot, logStot, logItot, 
                                        logK_dE_alpha=logK_dE_alpha)
    if table is None:
        return 0
    else:
        dat = pd.merge(table[['ID', 'pIC50']], df_cell[['ID', 'cell_pIC50']], on='ID', how='inner')
        return np.sum((dat.pIC50 - dat.cell_pIC50)**2)

res = minimize(f_find_conc, x0=(np.log(1*1E-6)-np.log(2), np.log(1*1E-6)), #method='COBYLA',
               bounds=((-15, 0), (-15, 0)), args=(inhibitor_list, args.mcmc_dir))
[init_logDtot, init_logStot] = res.x

with open(os.path.join(args.out_dir, "concentrations.txt"), "w") as f:
    print("LogD", ': %.3f' %init_logDtot, file=f)
    print("LogS", ': %.3f' %init_logStot, file=f)

n_points = 50
min_conc_I = 1E-12
max_conc_I = 1E-3

logDtot = np.ones(n_points)*init_logDtot
logStot = np.ones(n_points)*init_logStot
logItot = np.linspace(np.log(min_conc_I), np.log(max_conc_I), n_points)

table = _table_pIC50_hill(inhibitor_list, args.mcmc_dir, logDtot, logStot, logItot, 
                          logK_dE_alpha, OUTDIR=args.out_dir)
table.to_csv("pIC50_table.csv", index=True)
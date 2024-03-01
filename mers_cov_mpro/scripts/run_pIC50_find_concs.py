import warnings
import numpy as np
import sys
import os
import itertools
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

from _pIC50 import _table_pIC50_hill, _table_pIC50_hill_find_conc

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
def _pd_mean_std(df, name):
    ID = np.unique(df['ID'])
    mean = []
    std = []
    for _ID in ID:
        mean.append(df[name][df.ID == _ID].mean())
        std.append(df[name][df.ID == _ID].std())
    return pd.DataFrame([ID, mean, std], index=['ID', name, name+'_std']).T

_df_cell = pd.read_csv(args.cellular_pIC50_file)
_df_cell = _df_cell.rename(columns={'Molecule Name': "ID", 'IC50 (Num) (uM)': 'IC50', 'IC50 (Mod)': 'Mod', 'Hour_after': 'Hour'})
_df_cell = _df_cell[_df_cell['Mod']=='=']
_df_cell = _df_cell[_df_cell.Hour!=72.]
_df_cell['PGP'].fillna(2, inplace=True)
_df_cell.insert(len(_df_cell.columns), 'cell_pIC50', -np.log10(_df_cell.IC50*1E-6))

df_cell_full = pd.DataFrame(index=['Experiment', 'ID', 'Cell', 'cell_pIC50', 'cell_pIC50_std']).T

for g in itertools.product(np.unique(_df_cell['Experiment']), np.unique(_df_cell['Cell']), np.unique(_df_cell['PGP'])):
    _df = _df_cell[(_df_cell['Experiment']==g[0])*(_df_cell['Cell']==g[1])*(_df_cell['PGP']==g[2])]
    if len(_df)>0:
        _df_average = _pd_mean_std(_df, 'cell_pIC50')
        nrow_average = len(_df_average)
        _df_average.insert(len(_df_average.columns), "PGP", np.repeat(g[2], nrow_average))
        _df_average.insert(0, "Cell", np.repeat(g[1], nrow_average))
        _df_average.insert(0, "Experiment", np.repeat(g[0], nrow_average))
        df_cell_full = pd.concat([df_cell_full, _df_average])

expt = 'mavda'
cell = 'TMPRSS2'
PGP = 1

# df_cell = df_cell_full[(df_cell_full.Experiment==expt)*(df_cell_full.Cell==cell)*(df_cell_full.PGP==PGP)]
df_cell = df_cell_full.copy()

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

    table = _table_pIC50_hill_find_conc(inhibitor_list=inhibitor_list, mcmc_dir=mcmc_dir,
                                        logDtot=logDtot, logStot=logStot, logItot=logItot,
                                        logK_dE_alpha=logK_dE_alpha)
    if table is None:
        return 0
    else:
        dat = pd.merge(table[['ID', 'pIC50']], df_cell[['ID', 'cell_pIC50']], on='ID', how='inner')
        print("SSD=", round(np.sum((dat.pIC50 - dat.cell_pIC50)**2),2))
        return np.sum((dat.pIC50 - dat.cell_pIC50)**2) #+ (init_logDtot-init_logStot)


res = minimize(f_find_conc, x0=(np.log(1*1E-6)-np.log(2), np.log(1*1E-6)), method='COBYLA',
               bounds=((-21, 0), (-21, 0)), args=(inhibitor_list, args.mcmc_dir))
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

table = _table_pIC50_hill(inhibitor_list=inhibitor_list, mcmc_dir=args.mcmc_dir, 
                          logDtot=logDtot, logStot=logStot, logItot=logItot,
                          logK_dE_alpha=logK_dE_alpha, OUTDIR=args.out_dir)
table.to_csv("pIC50_table.csv", index=True)
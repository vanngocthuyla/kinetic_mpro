"""
Estimating the pIC50 for a list of inhibitor giving their traces.pickle files in mcmc_dir
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

import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from jax.config import config
config.update("jax_enable_x64", True)

from _pIC50 import table_pIC_hill_one_inhibitor, _pIC_hill_one_inhibitor
from _pIC50_correlation import _df_pIC50_pIC90, corr_leave_p_out

parser = argparse.ArgumentParser()

parser.add_argument( "--inhibitor_file",                type=str,               default="")
parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--logK_dE_alpha_file",            type=str,               default="")

parser.add_argument( "--cellular_pIC50_file",           type=str,               default="")
parser.add_argument( "--control_pIC50_file",            type=str,               default="")
parser.add_argument( "--inhibit_pIC50_file",            type=str,               default="")

parser.add_argument( "--enzyme_conc_nM",                type=int,               default=100)
parser.add_argument( "--substrate_conc_nM",             type=int,               default=1350)
parser.add_argument( "--conc_uncertainnty_log",         type=float,             default=0.1)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--include_experiments",           type=str,               default="")
parser.add_argument( "--exclude_experiments",           type=str,               default="")

parser.add_argument( "--nsim",                          type=int,               default=100)
parser.add_argument( "--correlation_method",            type=str,               default="")
parser.add_argument( "--p_out_CV",                      type=int,               default="20")

args = parser.parse_args()


if len(args.correlation_method)==0:
    methods = ['pearsonr', 'spearmanr', 'kendall', 'RMSD', 'aRMSD']
else:
    methods = args.correlation_method.split()
    for _method in methods:
        assert _method in ['pearsonr', 'spearmanr', 'kendall', 'RMSD', 'aRMSD'], print("The method should be \'pearsonr\', \'spearmanr\', \'kendall\', \'RMSD\', \'aRMSD\'.")

df_mers = pd.read_csv(args.inhibitor_file)
_inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

include_experiments = args.include_experiments.split()
exclude_experiments = args.exclude_experiments.split()
if len(include_experiments)>0:
    inhibitor_list = [name for name in _inhibitor_list if (name[:12] not in exclude_experiments) and (name[:12] in include_experiments)]
else:
    inhibitor_list = [name for name in _inhibitor_list if name[:12] not in exclude_experiments]
N = len(inhibitor_list)


if len(args.logK_dE_alpha_file)>0 and os.path.isfile(args.logK_dE_alpha_file):
    logK_dE_alpha = pickle.load(open(args.logK_dE_alpha_file, "rb"))
    print(f"Loading {args.logK_dE_alpha_file}.")
    
    if args.set_K_S_DS_equal_K_S_D: 
        logK_dE_alpha['logK_S_DS'] = logK_dE_alpha['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        logK_dE_alpha['logK_S_DI'] = logK_dE_alpha['logK_S_DS']

    for key in ['logKd', 'logK_S_D', 'logK_S_DS', 'kcat_DS', 'kcat_DSS']:
        assert key in logK_dE_alpha.keys(), f"Please provide {key} in logK_dE_alpha_file."
else:
    logK_dE_alpha = None


## Extracting cellular pIC50 -----------------------------------------------------------------------------------------
print(f"Extracting cellular pIC values from {args.cellular_pIC50_file}")
_df_cell = pd.read_csv(args.cellular_pIC50_file)
_df_cell = _df_cell.rename(columns={'Molecule Name': "ID", 'IC50 (uM)': 'IC50_uM', 'Hill': 'neg_hill'})
_df_cell.insert(len(_df_cell.columns), 'hill', -_df_cell['neg_hill'])
df_cell = _df_pIC50_pIC90(_df_cell)
df_cell = df_cell.rename(columns={'pIC50': 'cell_pIC50', 'pIC90': 'cell_pIC90'})


## Extracting control pIC50 --------------------------------------------------------------------------------------
print(f"Extracting cellular pIC values from {args.control_pIC50_file}")
_df_ctrl = pd.read_csv(args.control_pIC50_file)
_df_ctrl = _df_ctrl.rename(columns={'Molecule Name': "ID", 'IC50 (uM)': 'IC50_uM', 'Hill': 'hill'})
df_ctrl = _df_pIC50_pIC90(_df_ctrl)
df_ctrl = df_ctrl.rename(columns={'pIC50': 'control_pIC50', 'pIC90': 'control_pIC90'})


## Extracting inhibition pIC50 --------------------------------------------------------------------------------------
print(f"Extracting cellular pIC values from {args.inhibit_pIC50_file}")
_df_inhib = pd.read_csv(args.inhibit_pIC50_file)
_df_inhib = _df_inhib.rename(columns={'Molecule Name': "ID", 'Hill': 'hill'})
df_inhib = _df_pIC50_pIC90(_df_inhib)
df_inhib = df_inhib.rename(columns={'pIC50': 'inhibit_pIC50', 'pIC90': 'inhibit_pIC90'})


## Estimating dimer-only pIC50 ----------------------------------------------------------------------------------------
n_sim = args.nsim
os.chdir(args.out_dir)

if os.path.isfile('pIC_table.csv'):
    df_dimer = pd.read_csv("pIC_table.csv")
else:
    init_logMtot = np.log(args.enzyme_conc_nM*1E-9)
    init_logStot = np.log(args.substrate_conc_nM*1E-9)
    init_logDtot = init_logMtot-np.log(2)

    n_points = 50
    min_conc = 1E-12
    max_conc = 1E-3

    # Introduce random errors to enzyme and substrate concentrations
    adjusted_init_logDtot = init_logDtot + np.random.normal(loc=0, scale=args.conc_uncertainnty_log, size=n_sim) # uncertainty X% of ln[E] concentration
    adjusted_init_logStot = init_logStot + np.random.normal(loc=0, scale=args.conc_uncertainnty_log, size=n_sim) # uncertainty X% of ln[S] concentration
    logItot = np.linspace(np.log(min_conc), np.log(max_conc), n_points)

    df_dimer = pd.DataFrame([np.zeros(n_sim*N), np.zeros(n_sim*N), np.zeros(n_sim*N), np.zeros(n_sim*N), np.zeros(n_sim*N)], 
                            index=['ID', 'sim', 'pIC50', 'pIC90', 'hill']).T

    row_idx = 0
    for n, inhibitor in enumerate(inhibitor_list):

        inhibitor_dir = inhibitor[7:12]
        inhibitor_name = inhibitor[:12]

        print(f"Analyzing {inhibitor_name}")

        # # Loop over simulations
        # for _ in range(n_sim):
        #     logD = np.ones(n_points)*adjusted_init_logDtot[_]
        #     logStot = np.ones(n_points)*adjusted_init_logStot[_]

        #     # pIC50, hill slope, and pIC90 estimation
        #     df = table_pIC_hill_one_inhibitor(inhibitor, args.mcmc_dir, logD, logStot, logItot,
        #                                       logK_dE_alpha)

        #     pIC50_list = np.array(df.pIC50)
        #     pIC90_list = np.array(df.pIC90)
        #     hill_list = np.array(df.hill)
        #     print(pIC50_list)

        #     filtered_pIC50 = pIC50_list[hill_list>1]
        #     filtered_pIC90 = pIC90_list[hill_list>1]
        #     filtered_hill = hill_list[hill_list>1]

        #     df_dimer.at[row_idx, 'ID'] = inhibitor_name
        #     df_dimer.at[row_idx, 'sim'] = _
        #     df_dimer.at[row_idx, 'pIC50'] = np.median(filtered_pIC50)
        #     df_dimer.at[row_idx, 'pIC90'] = np.median(filtered_pIC90)
        #     df_dimer.at[row_idx, 'hill'] = np.median(filtered_hill)
        #     row_idx += 1

        f_table = _pIC_hill_one_inhibitor
        list_median_std = np.array(list(map(lambda i: f_table(inhibitor, args.mcmc_dir, 
                                                              np.ones(n_points)*adjusted_init_logDtot[i], np.ones(n_points)*adjusted_init_logStot[i], logItot, 
                                                              'median', logK_dE_alpha), range(n_sim))))
        for _ in range(n_sim):
            df_dimer.at[row_idx, 'ID'] = inhibitor_name
            df_dimer.at[row_idx, 'sim'] = _
            df_dimer.at[row_idx, 'pIC50'] = list_median_std[_, 0]
            df_dimer.at[row_idx, 'pIC90'] = list_median_std[_, 4]
            df_dimer.at[row_idx, 'hill'] = list_median_std[_, 2]
            row_idx += 1
    
    df_dimer.to_csv("pIC_table.csv", index=True)

df_dimer = df_dimer.rename(columns={'pIC50': "dimer_pIC50", 'pIC90': 'dimer_pIC90'})

## Combining data for the correlogram -------------------------------------------------------------------------------

assay_name_50 = ['Inhibition pIC50', 'Control pIC50', 'Dimer-only pIC50', 'Cellular pEC50']
assay_name_90 = ['Inhibition pIC90', 'Control pIC90', 'Dimer-only pIC90', 'Cellular pEC90']
IC50_keys = ['inhibit_pIC50', 'control_pIC50', 'dimer_pIC50', 'cell_pIC50']
IC90_keys = ['inhibit_pIC90', 'control_pIC90', 'dimer_pIC90', 'cell_pIC90']
change_names_50 = dict(([i, assay_name_50[i]]) for i in range(4))
change_names_90 = dict(([i, assay_name_90[i]]) for i in range(4))

for keys, change_names in zip([IC50_keys, IC90_keys], [change_names_50,change_names_90]):
    for _method in methods:
        table = pd.DataFrame(columns=keys, index=range(len(keys)))
        for i in range(1, len(keys)):
            for j in range(i):
                if i == 2 or j == 2:
                    corr_list = []
                    for _ in range(n_sim):
                        dat = pd.merge(df_inhib[['ID', 'inhibit_pIC50', 'inhibit_pIC90']], df_dimer[df_dimer.sim==_], on='ID', how='inner')
                        dat = pd.merge(dat, df_cell[['ID', 'cell_pIC50', 'cell_pIC90']], on='ID', how='inner')
                        dat = pd.merge(dat, df_ctrl[['ID', 'control_pIC50', 'control_pIC90']], on='ID', how='inner')
                        x = dat[keys[i]]
                        y = dat[keys[j]]
                        corr, std = corr_leave_p_out(x, y, args.p_out_CV, _method)
                        corr_list.append(corr)
                    mean_corr = np.mean(corr_list)
                    std_corr = np.std(corr_list)
                    text = str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
                else:
                    dat = pd.merge(df_inhib[['ID', 'inhibit_pIC50', 'inhibit_pIC90']], df_dimer[df_dimer.sim==0], on='ID', how='inner')
                    dat = pd.merge(dat, df_cell[['ID', 'cell_pIC50', 'cell_pIC90']], on='ID', how='inner')
                    dat = pd.merge(dat, df_ctrl[['ID', 'control_pIC50', 'control_pIC90']], on='ID', how='inner')
                    x = dat[keys[i]]
                    y = dat[keys[j]]
                    mean_corr, std_corr = corr_leave_p_out(x, y, args.p_out_CV, _method)
                    text = str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
                table.iloc[i][keys[j]] = text
        table = table.rename(index=change_names)
        table.to_csv(f'pIC{keys[i][-2:]}_{_method}.csv')
        del table
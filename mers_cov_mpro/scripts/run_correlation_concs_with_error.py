"""
Estimating the pIC50 for a list of inhibitor giving their traces.pickle files in mcmc_dir
"""

import warnings
import numpy as np
import sys
import os
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

from _pIC50 import ReactionRate_DimerOnly, scaling_data, parameter_estimation, _adjust_trace
from _pIC50_correlation import _pd_mean_std, _corr_coef

parser = argparse.ArgumentParser()

parser.add_argument( "--inhibitor_file",                type=str,               default="")
parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--logK_dE_alpha_file",            type=str,               default="")

parser.add_argument( "--cellular_pIC50_file",           type=str,               default="")
parser.add_argument( "--biochem_pIC50_file",            type=str,               default="")

parser.add_argument( "--enzyme_conc_nM",                type=int,               default="100")
parser.add_argument( "--substrate_conc_nM",             type=int,               default="1350")

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--include_experiments",           type=str,               default="")
parser.add_argument( "--exclude_experiments",           type=str,               default="")

parser.add_argument( "--nsim",                          type=int,               default=100)
parser.add_argument( "--correlation_method",            type=str,               default="")

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
    
    if args.set_K_S_DS_equal_K_S_D: 
        logK_dE_alpha['logK_S_DS'] = logK_dE_alpha['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        logK_dE_alpha['logK_S_DI'] = logK_dE_alpha['logK_S_DS']

    for key in ['logKd', 'logK_S_D', 'logK_S_DS', 'kcat_DS', 'kcat_DSS']:
        assert key in logK_dE_alpha.keys(), f"Please provide {key} in logK_dE_alpha_file."
else:
    logK_dE_alpha = None

## Extracting cellular pIC50
_df_cell = pd.read_csv(args.cellular_pIC50_file)
_df_cell = _df_cell.rename(columns={'Molecule Name': "ID", 'IC50 (Num) (uM)': 'IC50', 'IC50 (Mod)': 'Mod'})
_df_cell = _df_cell[_df_cell['Mod']=='=']
_df_cell.insert(len(_df_cell.columns), 'cell_pIC50', -np.log10(_df_cell.IC50*1E-6))
df_cell = _pd_mean_std(_df_cell, 'cell_pIC50')

## Extracting biochemical pIC50
_df_CDD = pd.read_csv(args.biochem_pIC50_file)
_df_CDD = _df_CDD[['Molecule Name', 'MERS-CoV-MPro_fluorescence-dose-response_weizmann: IC50 (uM)', 'MERS-CoV-MPro_fluorescence-dose-response_weizmann: Minh_Protease_MERS_Mpro_pIC50 (calc) (uM)']]
df_CDD = _df_CDD.rename(columns={'Molecule Name': "ID", 'MERS-CoV-MPro_fluorescence-dose-response_weizmann: IC50 (uM)': 'control_IC50_uM', "MERS-CoV-MPro_fluorescence-dose-response_weizmann: Minh_Protease_MERS_Mpro_pIC50 (calc) (uM)": 'biochem_pIC50'})

_df_ctrl = df_CDD[['ID', 'control_IC50_uM']]
_df_ctrl = _df_ctrl.dropna(axis=0)
_df_ctrl.insert(len(_df_ctrl.columns), 'control_pIC50', -np.log10(_df_ctrl['control_IC50_uM']*1E-6))

df_half = _pd_mean_std(df_CDD, 'biochem_pIC50')     # Inhibition pIC50
df_ctrl = _pd_mean_std(_df_ctrl, 'control_pIC50')   # Control pIC50

## Estimating pIC50
init_logMtot = np.log(args.enzyme_conc_nM*1E-9)
init_logStot = np.log(args.substrate_conc_nM*1E-9)
init_logDtot = init_logMtot-np.log(2)

n_points = 50
min_conc = 1E-12
max_conc = 1E-3

# Number of simulations
n_sim = args.nsim
# Introduce random errors to enzyme and substrate concentrations
adjusted_init_logDtot = init_logDtot + np.random.normal(loc=0, scale=0.1, size=n_sim)
adjusted_init_logStot = init_logStot + np.random.normal(loc=0, scale=0.1, size=n_sim)
logItot = np.linspace(np.log(min_conc), np.log(max_conc), n_points)

os.chdir(args.out_dir)

df_dimer = pd.DataFrame([np.zeros(n_sim*N), np.zeros(n_sim*N), np.zeros(n_sim*N), np.zeros(n_sim*N)], 
                        index=['ID', 'sim', 'pIC50', 'hill']).T

row_idx = 0
for n, inhibitor in enumerate(inhibitor_list):

    inhibitor_dir = inhibitor[7:12]
    inhibitor_name = inhibitor[:12]

    #Loading pickle file
    if not os.path.isfile(os.path.join(args.mcmc_dir, inhibitor_dir, 'traces.pickle')):
        continue
    else:
        print("Analyzing", inhibitor)

    trace = pickle.load(open(os.path.join(args.mcmc_dir, inhibitor_dir, 'traces.pickle'), "rb"))
    data = az.InferenceData.to_dataframe(az.convert_to_inference_data(trace))

    # Adjust trace with additional enzyme-substrate parameters
    nthin = int(len(data)/100)
    if logK_dE_alpha is not None:
        df = _adjust_trace(data.iloc[::nthin, :].copy(), logK_dE_alpha)
    else:
        df = data.iloc[::nthin, :].copy()

    # Loop over simulations
    for _ in range(n_sim):
        logD = np.ones(n_points)*adjusted_init_logDtot[_]
        logStot = np.ones(n_points)*adjusted_init_logStot[_]

        pIC50_list = []
        hill_list = []
        # Iterate through each row of the DataFrame and estimate the pIC50/hill for each inhibitor
        for index, row in df.iterrows():
            logKd = row.logKd
            # Substrate binding to enzyme
            logK_S_D = row.logK_S_D
            logK_S_DS = row.logK_S_DS
            # Inhibitor binding to enzyme
            logK_I_D = row.logK_I_D
            logK_I_DI = row.logK_I_DI

            logK_S_DI = row.logK_S_DI
            # rate parameters
            kcat_DS = row.kcat_DS
            kcat_DSI = row.kcat_DSI
            kcat_DSS = row.kcat_DSS

            # Simulate the CRC using the dimer-only kinetic model
            v_sim = ReactionRate_DimerOnly(logD, logStot, logItot, logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI,
                                           kcat_DS, kcat_DSI, kcat_DSS)
            data = {}
            data['x'] = np.log10(np.exp(logItot))
            data['y'] = scaling_data(v_sim, min(v_sim), max(v_sim))

            # CRC fitting to estimate pIC50 and hill
            try:
                [theta, ASE, var] = parameter_estimation(data)
                [R_b, R_t, x50, H] = theta
                pIC50_list.append(-x50)
                hill_list.append(H)
            except:
                pIC50_list.append(0)
                hill_list.append(0)

        pIC50_list = np.array(pIC50_list)
        hill_list = np.array(hill_list)

        filtered_pIC50 = pIC50_list[hill_list>0]
        filtered_hill = hill_list[hill_list>0]

        df_dimer.at[row_idx, 'ID'] = inhibitor_name
        df_dimer.at[row_idx, 'sim'] = _
        df_dimer.at[row_idx, 'pIC50'] = np.mean(filtered_pIC50)
        df_dimer.at[row_idx, 'hill'] = np.mean(filtered_hill)
        row_idx += 1

df_dimer.to_csv("pIC50_table.csv", index=True)
df_dimer = df_dimer.rename(columns={'pIC50': "dimer_pIC50"})

assay_name = ['Inhibition pIC50', 'Control pIC50', 'Dimer-only pIC50', 'Cellular pEC50']
assay_names = ['biochem_', 'control_', 'dimer_', 'cell_']
keys = []
for name in assay_names:
    keys.append(name + 'pIC50')
change_names = dict(([i, keys[i]]) for i in range(4))

for _method in methods:
    pIC50_table = pd.DataFrame(columns=keys, index=range(len(keys)))
    for i in range(1, len(keys)):
        for j in range(i):
            if i == 2 or j == 2:
                corr_list = []
                for _ in range(n_sim):
                    dat = pd.merge(df_half, df_dimer[df_dimer.sim==_], on='ID', how='inner')
                    dat = pd.merge(dat, df_cell, on='ID', how='inner')
                    dat = pd.merge(dat, df_ctrl, on='ID', how='inner')
                    x = dat[keys[i]]
                    y = dat[keys[j]]
                    corr, p = _corr_coef(x, y, method=_method)
                    corr_list.append(corr)
                mean_corr = np.mean(corr_list)
                std_corr = np.std(corr_list)
                text = r'n=' +str(len(dat)) +'; corr=' +str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
            else:
                dat = pd.merge(df_half, df_dimer[df_dimer.sim==0], on='ID', how='inner')
                dat = pd.merge(dat, df_cell, on='ID', how='inner')
                dat = pd.merge(dat, df_ctrl, on='ID', how='inner')
                x = dat[keys[i]]
                y = dat[keys[j]]
                corr, p = _corr_coef(x, y, method=_method)
                text = r'n=' +str(len(dat)) +'; corr=' +str('%5.3f' %corr)
            pIC50_table.iloc[i][keys[j]] = text
    pIC50_table = pIC50_table.rename(index=change_names)
    pIC50_table.to_csv(f'pIC50_{_method}.csv')
    del pIC50_table
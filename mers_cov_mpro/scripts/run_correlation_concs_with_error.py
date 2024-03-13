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

from _pIC50 import ReactionRate_DimerOnly, scaling_data, parameter_estimation, _adjust_trace, f_pIC90
from _pIC50_correlation import _pd_mean_std, _corr_coef, _df_biochem_pIC50_pIC90, _df_cell_pIC50_pIC90, corr_leave_p_out

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
parser.add_argument( "--p_out_CV",                      type=int,               default="2")

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

## Extracting cellular pIC50 -----------------------------------------------------------------------------------------
_df_cell = pd.read_csv(args.cellular_pIC50_file)
_df_cell = _df_cell.rename(columns={'Molecule Name': "ID", 'Hour_after': 'Hour', 'Hill (Num)': 'hill',
                                    'IC50 (Num) (uM)': 'IC50', 'IC50 (Mod)': 'IC50_Mod',
                                    'IC90 (Num) (uM)': 'IC90', 'IC90 (Mod)': 'IC90_Mod'})
_df_cell['PGP'].fillna(2, inplace=True)
_df_cell['Hour'].fillna(0, inplace=True)

_df_cell = _df_cell_pIC50_pIC90(_df_cell)
_df_cell = _df_cell.rename(columns={'pIC50': 'cell_pIC50', 'pIC90': 'cell_pIC90'})

df_cell_full = pd.DataFrame(index=['Experiment', 'ID', 'Cell', 'cell_pIC50', 'cell_pIC50_std', 'cell_pIC90', 'cell_pIC90_std']).T

for g in itertools.product(np.unique(_df_cell['Experiment']), np.unique(_df_cell['Cell']), np.unique(_df_cell['PGP']), np.unique(_df_cell['Hour'])):
    _df = _df_cell[(_df_cell['Experiment']==g[0])*(_df_cell['Cell']==g[1])*(_df_cell['PGP']==g[2])*(_df_cell['Hour']==g[3])]
    if len(_df)>0:
        _df_average_50 = _pd_mean_std(_df, f'cell_pIC50')
        _df_average_90 = _pd_mean_std(_df, f'cell_pIC90')
        _df_average = pd.merge(_df_average_50, _df_average_90, on='ID', how='inner')

        nrow_average = len(_df_average)
        _df_average.insert(0, "Experiment", np.repeat(g[0], nrow_average))
        _df_average.insert(0, "Cell", np.repeat(g[1], nrow_average))
        _df_average.insert(len(_df_average.columns), "PGP", np.repeat(g[2], nrow_average))
        _df_average.insert(0, "Hour", np.repeat(g[3], nrow_average))
        df_cell_full = pd.concat([df_cell_full, _df_average])
        print(g, len(_df))

expt = 'Mavda'
cell = 'TMPRSS2'
PGP = 1
hour = 48

df_cell = df_cell_full[(df_cell_full.Experiment==expt)*(df_cell_full.Cell==cell)*(df_cell_full.PGP==PGP)*(df_cell_full.Hour==hour)]

## Extracting biochemical pIC50 --------------------------------------------------------------------------------------
_df_CDD = pd.read_csv(args.biochem_pIC50_file)
df_CDD = _df_CDD.rename(columns={'Molecule Name': "ID", 'IC50 (Num)': 'control_IC50', 'IC50 (Mod)': 'control_IC50_Mod',
                                 "Minh_pIC50 (Num)": 'biochem_pIC50', 'Hill (Num)': 'hill'})
_df_ctrl = df_CDD[['ID', 'control_IC50', 'control_IC50_Mod', 'hill']]
_df_ctrl = _df_ctrl.dropna(axis=0)
_df_ctrl = _df_ctrl[_df_ctrl['control_IC50_Mod']=='=']
_df_ctrl = _df_ctrl.reset_index(drop=True)
_df_ctrl = _df_biochem_pIC50_pIC90(_df_ctrl, 'control_')

_df_half = df_CDD[['ID', 'biochem_pIC50', 'hill']]
pIC50 = _df_half.biochem_pIC50.to_numpy().ravel()
hill = _df_half.hill.to_numpy().ravel()
pIC90 = f_pIC90(pIC50, hill)
_df_half.insert(2, 'biochem_pIC90', pIC90)

df_ctrl_50 = _pd_mean_std(_df_ctrl, name='control_pIC50')
df_ctrl_90 = _pd_mean_std(_df_ctrl, name='control_pIC90')
df_ctrl = pd.merge(df_ctrl_50, df_ctrl_90, on='ID', how='outer')

df_half_50 = _pd_mean_std(_df_half, name='biochem_pIC50')
df_half_90 = _pd_mean_std(_df_half, name='biochem_pIC90')
df_half = pd.merge(df_half_50, df_half_90, on='ID', how='outer')

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
adjusted_init_logDtot = init_logDtot + np.random.normal(loc=0, scale=0.1, size=n_sim) # uncertainty 10% of E concentration
adjusted_init_logStot = init_logStot + np.random.normal(loc=0, scale=0.1, size=n_sim) # uncertainty 10% of S concentration
logItot = np.linspace(np.log(min_conc), np.log(max_conc), n_points)

os.chdir(args.out_dir)

df_dimer = pd.DataFrame([np.zeros(n_sim*N), np.zeros(n_sim*N), np.zeros(n_sim*N), np.zeros(n_sim*N), np.zeros(n_sim*N)], 
                        index=['ID', 'sim', 'pIC50', 'pIC90', 'hill']).T

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
        pIC90_list = []
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
                pIC90_list.append(f_pIC90(-x50, H))
            except:
                pIC50_list.append(0)
                pIC90_list.append(0)
                hill_list.append(0)

        pIC50_list = np.array(pIC50_list)
        pIC90_list = np.array(pIC90_list)
        hill_list = np.array(hill_list)

        filtered_pIC50 = pIC50_list[hill_list>1]
        filtered_pIC90 = pIC90_list[hill_list>1]
        filtered_hill = hill_list[hill_list>1]

        df_dimer.at[row_idx, 'ID'] = inhibitor_name
        df_dimer.at[row_idx, 'sim'] = _
        df_dimer.at[row_idx, 'pIC50'] = np.median(filtered_pIC50)
        df_dimer.at[row_idx, 'pIC90'] = np.median(filtered_pIC90)
        df_dimer.at[row_idx, 'hill'] = np.median(filtered_hill)
        row_idx += 1

df_dimer.to_csv("pIC_table.csv", index=True)
df_dimer = df_dimer.rename(columns={'pIC50': "dimer_pIC50", 'pIC90': 'dimer_pIC90'})

assay_name_50 = ['Inhibition pIC50', 'Control pIC50', 'Dimer-only pIC50', 'Cellular pEC50']
assay_name_90 = ['Inhibition pIC90', 'Control pIC90', 'Dimer-only pIC90', 'Cellular pEC90']
IC50_keys = ['biochem_pIC50', 'control_pIC50', 'dimer_pIC50', 'cell_pIC50']
IC90_keys = ['biochem_pIC90', 'control_pIC90', 'dimer_pIC90', 'cell_pIC90']
change_names_50 = dict(([i, assay_name_50[i]]) for i in range(4))
change_names_90 = dict(([i, assay_name_90[i]]) for i in range(4))

keys = IC50_keys
change_names = change_names_50
for _method in methods:
    table = pd.DataFrame(columns=keys, index=range(len(keys)))
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
                    corr, std = corr_leave_p_out(x, y, args.p_out_CV, _method)
                    corr_list.append(corr)
                mean_corr = np.mean(corr_list)
                std_corr = np.std(corr_list)
                text = str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
            else:
                dat = pd.merge(df_half, df_dimer[df_dimer.sim==0], on='ID', how='inner')
                dat = pd.merge(dat, df_cell, on='ID', how='inner')
                dat = pd.merge(dat, df_ctrl, on='ID', how='inner')
                x = dat[keys[i]]
                y = dat[keys[j]]
                mean_corr, std_corr = corr_leave_p_out(x, y, args.p_out_CV, _method)
                text = str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
            table.iloc[i][keys[j]] = text
    table = table.rename(index=change_names)
    table.to_csv(f'pIC50_{_method}.csv')
    del table

keys = IC90_keys
change_names = change_names_90
for _method in methods:
    table = pd.DataFrame(columns=keys, index=range(len(keys)))
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
                    corr, std = corr_leave_p_out(x, y, args.p_out_CV, _method)
                    corr_list.append(corr)
                mean_corr = np.mean(corr_list)
                std_corr = np.std(corr_list)
                text = str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
            else:
                dat = pd.merge(df_half, df_dimer[df_dimer.sim==0], on='ID', how='inner')
                dat = pd.merge(dat, df_cell, on='ID', how='inner')
                dat = pd.merge(dat, df_ctrl, on='ID', how='inner')
                x = dat[keys[i]]
                y = dat[keys[j]]
                mean_corr, std_corr = corr_leave_p_out(x, y, args.p_out_CV, _method)
                text = str('%5.3f' %mean_corr) +' ± ' +str('%5.3e' %std_corr)
            table.iloc[i][keys[j]] = text
    table = table.rename(index=change_names)
    table.to_csv(f'pIC90_{_method}.csv')
    del table
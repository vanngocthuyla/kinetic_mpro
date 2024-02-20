import warnings
import numpy as np
import sys
import os
import argparse

import pickle
import arviz as az
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _pIC50 import ReactionRate_DimerOnly, scaling_data, parameter_estimation, _adjust_trace
from _model_mers import _dE_find_prior, _extract_conc_percent_error

parser = argparse.ArgumentParser()

parser.add_argument( "--inhibitor_file",                type=str,               default="")
parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--logK_dE_alpha_file",            type=str,               default="")

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

args = parser.parse_args()

df_mers = pd.read_csv(args.inhibitor_file)
inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

if len(args.logK_dE_alpha_file)>0 and os.path.isfile(args.logK_dE_alpha_file):
    logK_dE_alpha = pickle.load(open(args.logK_dE_alpha_file, "rb"))
    
    if args.set_K_S_DS_equal_K_S_D: 
        logK_dE_alpha['logK_S_DS'] = logK_dE_alpha['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        logK_dE_alpha['logK_S_DI'] = logK_dE_alpha['logK_S_DS']

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'alpha:ESI:0', 'alpha:ESI:1', 'alpha:ESI:2', 'alpha:ESI:3',
                'kcat_DS', 'kcat_DSS', 'dE:100', 'dE:50', 'dE:25']:
        assert key in logK_dE_alpha.keys(), f"Please provide {key} in logK_dE_alpha_file."

E_list = {}
for key in ['dE:100', 'dE:50', 'dE:25']:
    E_list[key] = logK_dE_alpha[key]

no_expt = 4
alpha_list = {}
for i in range(no_expt):
    alpha_list[f'alpha:ESI:{i}'] = logK_dE_alpha[f'alpha:ESI:{i}']

init_logMtot = np.log(100*1E-9)
init_logStot = np.log(1350*1E-9)

Etot = _dE_find_prior([None, np.array([init_logMtot]), None, None], E_list)
init_logM = jnp.log(Etot[0]*1E-9)
init_logD = init_logM-np.log(2)

n_points = 50
min_conc = 1E-12
max_conc = 1E-3

logM = np.ones(n_points)*init_logM
logD = np.ones(n_points)*init_logD
logStot = np.ones(n_points)*init_logStot
logItot = np.linspace(np.log(min_conc), np.log(max_conc), n_points)

if not os.path.exists('Plot'):
    os.makedirs('Plot')
if not os.path.exists('Parameters'):
    os.makedirs('Parameters')

os.chdir(args.out_dir)
table_mean = pd.DataFrame()
table_std = pd.DataFrame()
for n, inhibitor in enumerate(inhibitor_list):
    print("Analyzing", inhibitor)

    inhibitor_dir = inhibitor[7:12]
    inhibitor_name = inhibitor[:12]

    trace = pickle.load(open(os.path.join(args.mcmc_dir, inhibitor_dir, 'traces.pickle'), "rb"))
    data = az.InferenceData.to_dataframe(az.convert_to_inference_data(trace))

    nthin = int(len(data)/100)
    df = _adjust_trace(data.iloc[::nthin, :].copy(), logK_dE_alpha)

    pIC50_list = []
    hill_list = []
    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        logKd = row.logKd
        # Substrate binding to enzyme
        logK_S_M = row.logK_S_M
        logK_S_D = row.logK_S_D
        logK_S_DS = row.logK_S_DS
        # Inhibitor binding to enzyme
        logK_I_M = row.logK_I_M
        logK_I_D = row.logK_I_D
        logK_I_DI = row.logK_I_DI

        logK_S_DI = row.logK_S_DI
        # rate parameters
        kcat_MS = row.kcat_MS
        kcat_DS = row.kcat_DS
        kcat_DSI = row.kcat_DSI
        kcat_DSS = row.kcat_DSS

        v_sim = ReactionRate_DimerOnly(logD, logStot, logItot, logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI,
                                       kcat_DS, kcat_DSI, kcat_DSS)*alpha_list['alpha:ESI:0']
        data = {}
        data['x'] = np.log10(np.exp(logItot))
        data['y'] = scaling_data(v_sim, min(v_sim), max(v_sim))

        # CRC fitting
        try:
            [theta, ASE, var] = parameter_estimation(data)
            [R_b, R_t, x50, H] = theta
            pIC50_list.append(-x50)
            hill_list.append(H)
            # temp = np.linspace(min(data['x']), max(data['x']), 50)
            # plt.plot(data['x'], data['y'], ".")
            # plt.plot(temp, f_curve_vec(temp, *theta), "-");
        except:
            print("Cannot estimate parameters")
            pIC50_list.append(0)
            hill_list.append(0)

    df.insert(len(df.columns), 'pIC50', pIC50_list)
    df.insert(len(df.columns), 'hill', hill_list)
    df.to_csv(os.path.join('Parameters', inhibitor_name+".csv"), index=False)

    plt.figure()
    sns.kdeplot(data=df, x='pIC50', shade=True, alpha=0.1);
    plt.savefig(os.path.join('Plot', inhibitor_name))

    df_inhibitor = df[['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI', 'pIC50', 'hill']]
    table_mean.insert(len(table_mean.columns), inhibitor_name, df_inhibitor.median())
    table_std.insert(len(table_std.columns), inhibitor_name, df_inhibitor.std())

table_mean = table_mean.T
table_std = table_std.T.rename(columns={'logK_I_M': 'logK_I_M_std', 'logK_I_D': 'logK_I_D_std', 
                                        'logK_I_DI': 'logK_I_DI_std', 'logK_S_DI': 'logK_S_DI_std', 
                                        'pIC50': 'pIC50_std', 'hill': 'hill_std'})
table = pd.concat([table_mean, table_std], axis=1)
table.to_csv("pIC50_table.csv", index=True)
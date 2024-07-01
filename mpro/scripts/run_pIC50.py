"""
Estimating the pIC50 for a list of inhibitor giving their traces.pickle files in mcmc_dir.
The results showed a table of enzyme-inhibitor parameters, pIC50, hill slope, and some message
about the curves (trending, noise).
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
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _pIC50 import ReactionRate_DimerOnly, scaling_data, parameter_estimation, _adjust_trace, f_curve_vec, f_pIC90
from _model_mers import _dE_find_prior
from _load_data_mers import load_data_one_inhibitor
from _CRC_fitting import _expt_check_noise_trend

parser = argparse.ArgumentParser()

parser.add_argument( "--inhibitor_file",                type=str,               default="")
parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--logK_dE_alpha_file",            type=str,               default="")

parser.add_argument( "--multi_var",                     action="store_true",    default=True)
parser.add_argument( "--enzyme_conc_nM",                type=int,               default="100")
parser.add_argument( "--substrate_conc_nM",             type=int,               default="1350")

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--include_experiments",           type=str,               default="")
parser.add_argument( "--exclude_experiments",           type=str,               default="")

args = parser.parse_args()

df_mers = pd.read_csv(args.inhibitor_file)
_inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

include_experiments = args.include_experiments.split()
exclude_experiments = args.exclude_experiments.split()
if len(include_experiments)>0:
    inhibitor_list = [name for name in _inhibitor_list if (name[:12] not in exclude_experiments) and (name[:12] in include_experiments)]
else:
    inhibitor_list = [name for name in _inhibitor_list if name[:12] not in exclude_experiments]

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

init_logMtot = np.log(args.enzyme_conc_nM*1E-9)
init_logStot = np.log(args.substrate_conc_nM*1E-9)
init_logDtot = init_logMtot-np.log(2)

n_points = 50
min_conc = 1E-12
max_conc = 1E-3

logM = np.ones(n_points)*init_logMtot
logD = np.ones(n_points)*init_logDtot
logStot = np.ones(n_points)*init_logStot
logItot = np.linspace(np.log(min_conc), np.log(max_conc), n_points)

if not os.path.exists('Plot'):
    os.makedirs('Plot')
if not os.path.exists('Parameters'):
    os.makedirs('Parameters')

os.chdir(args.out_dir)
table_mean = pd.DataFrame()
table_std = pd.DataFrame()
table_mes = pd.DataFrame(index=['Noise', 'Trend'])
for n, inhibitor in enumerate(inhibitor_list):

    inhibitor_dir = inhibitor[7:12]
    inhibitor_name = inhibitor[:12]

    if not os.path.isfile(os.path.join(args.mcmc_dir, inhibitor_dir, 'traces.pickle')):
        continue
    else:
        print("Analyzing", inhibitor)

    trace = pickle.load(open(os.path.join(args.mcmc_dir, inhibitor_dir, 'traces.pickle'), "rb"))
    data = az.InferenceData.to_dataframe(az.convert_to_inference_data(trace))

    nthin = int(len(data)/100)
    if logK_dE_alpha is not None:
        df = _adjust_trace(data.iloc[::nthin, :].copy(), logK_dE_alpha)
    else:
        df = data.iloc[::nthin, :].copy()

    pIC50_list = []
    pIC90_list = []
    hill_list = []
    # Iterate through each row of the DataFrame
    plt.figure()
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

        v_sim = ReactionRate_DimerOnly(logD, logStot, logItot, logK_S_D, logK_S_DS, logK_I_D, logK_I_DI, logK_S_DI,
                                       kcat_DS, kcat_DSI, kcat_DSS)
        data = {}
        data['x'] = np.log10(np.exp(logItot))
        data['y'] = scaling_data(v_sim, min(v_sim), max(v_sim))

        # CRC fitting
        try:
            [theta, ASE, var] = parameter_estimation(data)
            [R_b, R_t, x50, H] = theta
            pIC50_list.append(-x50)
            hill_list.append(H)
            pIC90_list.append(f_pIC90(-x50, H))
            if H > 0:
                temp = np.linspace(min(data['x']), max(data['x']), 50)
                # plt.plot(data['x'], scaling_data(data['y'], min(data['y']), max(data['y'])), ".")
                plt.plot(temp, f_curve_vec(temp, *theta), "-")
                plt.savefig(os.path.join('Plot', 'CRC_'+inhibitor_name))
        except:
            # print("Cannot estimate parameters")
            pIC50_list.append(0)
            hill_list.append(0)
            pIC90_list.append(0)

    df.insert(len(df.columns), 'pIC50', pIC50_list)
    df.insert(len(df.columns), 'pIC90', pIC90_list)
    df.insert(len(df.columns), 'hill', hill_list)
    df.to_csv(os.path.join('Parameters', inhibitor_name+".csv"), index=False)

    plt.figure()
    sns.kdeplot(data=df[df.hill>0], x='pIC50', shade=True, alpha=0.1);
    plt.savefig(os.path.join('Plot', inhibitor_name))

    df_inhibitor = df[['logK_I_D', 'logK_I_DI', 'logK_S_DI', 'pIC50', 'pIC90', 'hill']]
    df_inhibitor = df_inhibitor[df_inhibitor.hill>0]
    table_mean.insert(len(table_mean.columns), inhibitor_name, df_inhibitor.median())
    table_std.insert(len(table_std.columns), inhibitor_name, df_inhibitor.std())

    del df

    expts, expts_plot = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==inhibitor)*(df_mers['Drop']!=1.0)],
                                                multi_var=args.multi_var)

    ## Outlier detection and trend checking
    [_, _, mes_noise, mes_trend] = _expt_check_noise_trend(expts)

    ## Report if curve is increasing or decreasing
    table_mes.insert(len(table_mes.columns), inhibitor_name, [mes_noise, mes_trend])

table_mean = table_mean.T
table_std = table_std.T.rename(columns={'logK_I_D': 'logK_I_D_std', 'logK_I_DI': 'logK_I_DI_std', 'logK_S_DI': 'logK_S_DI_std', 
                                        'pIC50': 'pIC50_std', 'pIC90': 'pIC90_std', 'hill': 'hill_std'})
table_mes = table_mes.T
table = pd.concat([table_mean, table_std, table_mes], axis=1)
table.to_csv("pIC50_table.csv", index=True)
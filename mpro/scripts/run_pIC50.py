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

from _kinetics import ReactionRate_DimerOnly
from _model import _dE_find_prior
from _load_data_mers import load_data_one_inhibitor
from _CRC_fitting import _expt_check_noise_trend
from _pIC50 import table_pIC_hill_one_inhibitor

from jax.config import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()

parser.add_argument( "--inhibitor_file",                type=str,               default="")
parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--logK_dE_alpha_file",            type=str,               default="")

parser.add_argument( "--multi_var",                     action="store_true",    default=True)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--include_experiments",           type=str,               default="")
parser.add_argument( "--exclude_experiments",           type=str,               default="")

parser.add_argument( "--enzyme_conc_nM",                type=float,             default="100")
parser.add_argument( "--substrate_conc_nM",             type=float,             default="1350")

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

    # pIC50, hill slope, and pIC90 estimation
    df = table_pIC_hill_one_inhibitor(inhibitor, args.mcmc_dir, logD, logStot, logItot,
                                      logK_dE_alpha, args.out_dir)
    if df is not None:
        print(f"Analyzing {inhibitor_name}")
        table_mean.insert(len(table_mean.columns), inhibitor_name, df.median())
        table_std.insert(len(table_std.columns), inhibitor_name, df.std())
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
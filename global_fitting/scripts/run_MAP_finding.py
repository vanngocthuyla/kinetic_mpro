import warnings
import numpy as np
import sys
import os
import argparse

import pickle
import arviz as az
import pandas as pd

import jax
import jax.numpy as jnp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _MAP_finding import map_finding
from _trace_analysis import extract_params_from_map_and_prior
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _load_data import load_data_mut_wt
from _plotting import adjustable_plot_data

parser = argparse.ArgumentParser()

parser.add_argument( "--fit_mutant_kinetics",   action="store_true",    default=False)
parser.add_argument( "--fit_mutant_AUC",        action="store_true",    default=False)
parser.add_argument( "--fit_mutant_ICE",        action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Nashed",   action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Vuong",    action="store_true",    default=False)
parser.add_argument( "--fit_E_S",               action="store_true",    default=False)
parser.add_argument( "--fit_E_I",               action="store_true",    default=False)
parser.add_argument( "--multi_var_mut",         action="store_true",    default=False)
parser.add_argument( "--multi_var_wt",          action="store_true",    default=False)

parser.add_argument( "--mcmc_file",             type=str, 				default="")
parser.add_argument( "--prior_infor",           type=str,               default="")
# parser.add_argument( "--nsamples",              type=str, 				default=None)
parser.add_argument( "--out_dir",               type=str,               default="")

args = parser.parse_args()


expts, expts_mut, expts_wt, expts_wt_2 = load_data_mut_wt(args.fit_mutant_kinetics, args.fit_mutant_AUC, args.fit_mutant_ICE,
                                                          args.fit_wildtype_Nashed, args.fit_wildtype_Vuong,
                                                          args.fit_E_S, args.fit_E_I, args.multi_var_mut, args.multi_var_wt)

trace = pickle.load(open(args.mcmc_file, "rb"))
trace['log_sigma_rate:mut'] = trace['log_sigma_rate:0']
trace['log_sigma_auc:mut'] = trace['log_sigma_auc:0']
trace['log_sigma_ice:mut'] = trace['log_sigma_ice:0']
trace['log_sigma_rate:wt_1'] = trace['log_sigma_rate:1']
trace['log_sigma_rate:wt_2'] = trace['log_sigma_rate:1']
print(trace.keys())

shared_params = {}
shared_params['logKd'] = {'assigned_idx': 2, 'shared_idx': 1}
shared_params['kcat_DSI'] = {'assigned_idx': 2, 'shared_idx': 1}
shared_params['kcat_DSS'] = {'assigned_idx': 2, 'shared_idx': 1}

if shared_params is not None and len(shared_params)>0:
    for name in shared_params.keys():
        param = shared_params[name]
        assigned_idx = param['assigned_idx']
        shared_idx = param['shared_idx']
        trace[f'{name}:{assigned_idx}'] = trace[f'{name}:{shared_idx}']

df = pd.read_csv(args.prior_infor)
print(df)
prior_infor_update = []
for index, row in df.iterrows():
    prior_infor_update.append(row.to_dict())

# logKd_min = -20.
# logKd_max = 0.
# kcat_min = 0.
# kcat_max = 1.

# prior = {}
# prior['logKd'] = {'type':'logK', 'name': 'logKd', 'fit':'local', 'dist': 'normal', 'loc': [-5, -13.5, -13.5], 'scale': [1, 3, 3]}
# prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
# prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
# prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
# prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
# prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'global', 'dist': 'normal', 'loc': -13, 'scale': 3}
# prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'global', 'dist': 'normal', 'loc': -15, 'scale': 3}
# prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

# prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0}
# prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': None, 'value': 0}
# prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': [1, 100, 100]}
# prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': [1, 100, 100]}

# prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
# prior_infor_update = check_prior_group(prior_infor, len(expts))
# pd.DataFrame(prior_infor_update).to_csv("Prior_infor.csv", index=False)
# pd.DataFrame(prior_infor_update)

# Finding MAP
[map_index, map_params, log_probs] = map_finding(trace, expts, prior_infor_update)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

n = 0
for expt_plot in [expts_mut, expts_wt, expts_wt_2]:
    if len(expt_plot)>0:
        adjustable_plot_data(expt_plot, extract_logK_n_idx(params_logK, n), extract_kcat_n_idx(params_kcat, n),
                             OUTDIR=args.out_dir)
        n = n + 1

with open("map.txt", "w") as f:
    print("MAP index:" + str(map_index), file=f)
    print("\nKinetics parameters:", file=f)
    for key in trace.keys():
        print(key, ': %.3f' %trace[key][map_index], file=f)
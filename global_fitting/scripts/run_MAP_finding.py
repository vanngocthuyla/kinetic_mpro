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

parser.add_argument( "--set_K_I_M_equal_K_S_M",         action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DSS",   action="store_true",    default=False)

parser.add_argument( "--mcmc_file",             type=str, 				default="")
parser.add_argument( "--prior_infor",           type=str,               default="")
# parser.add_argument( "--nsamples",              type=str, 				default=None)
parser.add_argument( "--out_dir",               type=str,               default="")

args = parser.parse_args()


expts, expts_mut, expts_wt, expts_wt_2 = load_data_mut_wt(args.fit_mutant_kinetics, args.fit_mutant_AUC, args.fit_mutant_ICE,
                                                          args.fit_wildtype_Nashed, args.fit_wildtype_Vuong,
                                                          args.fit_E_S, args.fit_E_I, args.multi_var_mut, args.multi_var_wt)

trace = pickle.load(open(args.mcmc_file, "rb"))
print(trace.keys())

shared_params = {}
shared_params['logKd'] = {'assigned_idx': 2, 'shared_idx': 1}
# shared_params['kcat_DSI'] = {'assigned_idx': 2, 'shared_idx': 1}
# shared_params['kcat_DSS'] = {'assigned_idx': 2, 'shared_idx': 1}

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

# Finding MAP
[map_index, map_params, log_probs] = map_finding(trace, expts, prior_infor_update, 
                                                 args.set_K_I_M_equal_K_S_M, args.set_K_S_DI_equal_K_S_DS, 
                                                 args.set_kcat_DSI_equal_kcat_DSS)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

if args.set_K_I_M_equal_K_S_M:
    params_logK['logK_I_M'] = params_logK['logK_S_M']
if args.set_K_S_DI_equal_K_S_DS:
    params_logK['logK_S_DI'] = params_logK['logK_S_DS'] 
if args.set_kcat_DSI_equal_kcat_DSS:
    params_kcat['kcat_DSS'] = params_kcat['kcat_DSI']

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
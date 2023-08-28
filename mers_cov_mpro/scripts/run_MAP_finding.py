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

from _load_data_mers import load_data_one_inhibitor
from _plotting import adjustable_plot_data_mers
from _MAP_finding_mers import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior
from _trace_analysis import extract_params_from_trace_and_prior

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--mcmc_file",                     type=str,               default="")
parser.add_argument( "--prior_infor",                   type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)

parser.add_argument( "--set_K_I_M_equal_K_S_M",         action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSS_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DSS",   action="store_true",    default=False)

parser.add_argument( "--nsamples",                      type=str,               default=None)

args = parser.parse_args()

df_mers = pd.read_csv(args.input_file)
expts, expts_plot = load_data_one_inhibitor(df_mers, multi_var=args.multi_var)

logKd_min = -20.
logKd_max = 0.
kcat_min = 0.
kcat_max = 10

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': 'normal', 'loc': -8, 'scale': 2}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

# prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}

if args.set_K_I_M_equal_K_S_M: 
    del prior['logK_I_M']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']
if args.set_kcat_DSS_equal_kcat_DS:
    del prior['kcat_DSS']
if args.set_kcat_DSI_equal_kcat_DS or args.set_kcat_DSI_equal_kcat_DSS:
    del prior['kcat_DSI']

shared_params = None

prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
print("Prior information: \n", pd.DataFrame(prior_infor_update))

# df = pd.read_csv(args.prior_infor)
# print(df)
# prior_infor_update = []
# for index, row in df.iterrows():
#     prior_infor_update.append(row.to_dict())

trace = pickle.load(open(args.mcmc_file, "rb"))
print(trace.keys())

expts_map = []
count = 0
for i in range(len(expts_plot)):
    if i == 1:
        expts_map.append(expts[i])
    else:
        expts_map.append({'enzyme': 'mers',
                          'figure': '2023-03-24-3E_8S.xlsx',
                          'kinetics': [expts_plot[count]['v'], expts_plot[count]['logMtot'],
                                       expts_plot[count]['logStot'], expts_plot[count]['logItot']]})
        count = count + 1
trace['alpha:2'] = trace['alpha:0']
trace['alpha:3'] = trace['alpha:0']

[map_index, map_params, log_probs] = map_finding(trace, expts_map, prior_infor_update, 
                                                 args.set_K_I_M_equal_K_S_M, args.set_K_S_DI_equal_K_S_DS, 
                                                 args.set_kcat_DSS_equal_kcat_DS, args.set_kcat_DSI_equal_kcat_DS,
                                                 args.set_kcat_DSI_equal_kcat_DSS)

with open("map.txt", "w") as f:
    print("MAP index:" + str(map_index), file=f)
    print("\nKinetics parameters:", file=f)
    for key in trace.keys():
        print(key, ': %.3f' %trace[key][map_index], file=f)

pickle.dump(log_probs, open('log_probs.pickle', "wb"))

map_values = {}
for key in trace.keys():
    map_values[key] = trace[key][map_index]
pickle.dump(map_values, open('map.pickle', "wb"))

# ## Fitting plot
# params_logK, params_kcat = extract_params_from_trace_and_prior(trace, prior_infor_update)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

for n in range(len(expts)):
    if args.set_K_I_M_equal_K_S_M:
        try: params_logK[f'logK_I_M:{n}'] = params_logK[f'logK_S_M:{n}']
        except: params_logK['logK_I_M'] = params_logK['logK_S_M']
    if args.set_K_S_DI_equal_K_S_DS:
        try: params_logK[f'logK_S_DI:{n}'] = params_logK[f'logK_S_DS:{n}']
        except: params_logK['logK_S_DI'] = params_logK['logK_S_DS']
    if args.set_kcat_DSS_equal_kcat_DS: 
        try: params_kcat[f'kcat_DSS:{n}'] = params_kcat[f'kcat_DS:{n}']
        except: params_kcat['kcat_DSS'] = params_kcat['kcat_DS']
    if args.set_kcat_DSI_equal_kcat_DS: 
        try: params_kcat[f'kcat_DSI:{n}'] = params_kcat[f'kcat_DS:{n}']
        except: params_kcat['kcat_DSI'] = params_kcat['kcat_DS']
    elif args.set_kcat_DSI_equal_kcat_DSS:
        try: params_kcat[f'kcat_DSI:{n}'] = params_kcat[f'kcat_DSS:{n}']
        except: params_kcat['kcat_DSI'] = params_kcat['kcat_DSS']

n = 0
adjustable_plot_data_mers(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                          extract_kcat_n_idx(params_kcat, n, shared_params),
                          OUTDIR=args.out_dir)
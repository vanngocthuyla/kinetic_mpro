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

from _load_data_mers import load_data_no_inhibitor, load_data_one_inhibitor
from _plotting import adjustable_plot_data_mers
from _MAP_finding_mers_concs import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, extract_params_from_trace_and_prior

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="Inhibitor")
parser.add_argument( "--mcmc_file",                     type=str,               default="")
parser.add_argument( "--prior_infor",                   type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_error_E",                   action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.5)
parser.add_argument( "--set_error_I",                   action="store_true",    default=False)
parser.add_argument( "--dI",                            type=float,             default=0.1)

parser.add_argument( "--set_K_I_M_equal_K_S_M",         action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSS_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DSS",   action="store_true",    default=False)

parser.add_argument( "--nsamples",                      type=str,               default=None)

args = parser.parse_args()

df_mers = pd.read_csv(args.input_file)
expts_no_I, expts_plot_no_I = load_data_no_inhibitor(df_mers[df_mers['Inhibitor (nM)']==0.0], 
                                                     multi_var=args.multi_var)
# expts_I, expts_plot_I = load_data_one_inhibitor(df_mers[df_mers['Inhibitor (nM)']>0.0],
#                                                 multi_var=args.multi_var, name=args.name_inhibitor)

# expts = expts_no_I+expts_I
# expts_plot = expts_plot_no_I+expts_plot_I
inhibitor_name = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])
expts = expts_no_I
expts_plot = expts_plot_no_I
for i, name in enumerate(inhibitor_name):
    expts_, expts_plot_ = load_data_one_inhibitor(df_mers[df_mers['Inhibitor_ID']==name],
                                                    multi_var=args.multi_var)
    expts = expts + expts_
    expts_plot = expts_plot + expts_plot_


shared_params = None

df = pd.read_csv(args.prior_infor)
print(df)
prior_infor_update = []
for index, row in df.iterrows():
    prior_infor_update.append(row.to_dict())

trace = pickle.load(open(args.mcmc_file, "rb"))
print(trace.keys())

expts_map = expts
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
if args.set_error_E and args.dE>0: 
    _error_E =trace[f'percent_error_E:{n}'][map_index]
else: _error_E = None
adjustable_plot_data_conc(expts_plot_no_I, extract_logK_n_idx(params_logK, n, shared_params),
                          extract_kcat_n_idx(params_kcat, n, shared_params), 
                          error_E= _error_E, OUTDIR=args.out_dir)
n = 1
if args.set_error_E and args.dE>0: 
    if args.multi_var:
        _error_E = [trace[f'percent_error_E:{n}:{i}'][map_index] for i in range(len(expts_plot_I))]
    else: _error_E =trace[f'percent_error_E:{n}'][map_index]
else: _error_E = None
if args.set_error_I and args.dI>0:
    if args.multi_var: _I0 = [trace[f'I0:{n}:{i}'][map_index] for i in range(4)]
    else: _I0 = trace['I0:{n}'][map_index]
else: _I0 = None
adjustable_plot_data_conc(expts_plot_I, extract_logK_n_idx(params_logK, n, shared_params),
                          extract_kcat_n_idx(params_kcat, n, shared_params), 
                          error_E=_error_E, I0 = _I0, OUTDIR=args.out_dir)
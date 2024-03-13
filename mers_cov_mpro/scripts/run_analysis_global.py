import warnings
import numpy as np
import sys
import os
from glob import glob
import argparse

import pickle
import arviz as az
import pandas as pd

import jax
import jax.numpy as jnp

from _load_data_mers import load_data_one_inhibitor
from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, _trace_ln_to_log
from _plotting import plot_data_conc_log, _plot_1D_histogram, _plot_2D_histogram, _heat_map, _linear_corr

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

parser = argparse.ArgumentParser()

parser.add_argument( "--kinetic_file",                  type=str,               default="")
parser.add_argument( "--mcmc_file",                     type=str,               default="")
parser.add_argument( "--map_dir",                       type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="")

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--niters",                        type=int,               default=0)
parser.add_argument( "--extended_global",               type=int,               default=False)

args = parser.parse_args()

df_mers = pd.read_csv(args.kinetic_file)
if len(args.name_inhibitor)>0:
    inhibitor_list = np.array([args.name_inhibitor.split()+'-001'])
else:
    inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

shared_params = None

# Convert from ln to log10
trace = pickle.load(open(args.mcmc_file, "rb"))
trace_log = _trace_ln_to_log(trace, group_by_chain=False, nchain=args.nchain)
pickle.dump(trace_log, open(os.path.join(args.out_dir, "traces_log.pickle"), "wb"))

if args.niters == 0:
    key = list(trace.keys())
    niters = int(len(trace[key[0]])/args.nchain)
else:
    niters = args.niters

## --------------------------
#Plot of shared parameters
params_names = {'kcat_DS':              '$k_{cat,DS}$',
                'kcat_DSS':             '$k_{cat,DSS}$',
                'logKd':                '$logK_d$',
                'logK_S_M':             '$logK_{S,M}$',
                'logK_S_D':             '$logK_{S,D}$',
                'logK_S_DS':            '$logK_{S,DS}$'}

# Change parameter names for plotting
trace_log_plot = {}
for key in trace_log.keys():
    if key in params_names.keys():
        trace_log_plot[params_names[key]] = trace_log[key]
pickle.dump(trace_log_plot, open(os.path.join(args.out_dir, "traces_log_plot_logK_S.pickle"), "wb"))

param_list = ['$logK_d$', '$logK_{S,M}$', '$logK_{S,D}$', '$logK_{S,DS}$']
_plot_1D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, '1D_logK'))

param_list = ['$k_{cat,DS}$', '$k_{cat,DSS}$']
_plot_1D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, '1D_kcat'))
## --------------------------
#Plot of alpha and dE
params_names = {}
for _key in trace.keys():
    if _key.startswith("alpha"):
        _new_key = r'{'+_key[6:]+'}'
        _new_key =_new_key.replace("_", "-")
        params_names[_key] = f'$\\alpha^{_new_key}$'
    if _key.startswith("dE"):
        _new_key = r'{'+_key[3:]+'}'
        _new_key =_new_key.replace("_", "-")
        params_names[_key] = f'$[E]^{_new_key}$'

# Change parameter names for plotting
trace_log_plot = {}
for key in trace_log.keys():
    if key in params_names.keys():
        trace_log_plot[params_names[key]] = trace_log[key]
pickle.dump(trace_log_plot, open(os.path.join(args.out_dir, "traces_log_plot_alpha_dE.pickle"), "wb"))

if args.extended_global:
    param_name = '$\\alpha'
    param_list = [key for key in trace_log_plot.keys() if key.startswith(param_name)]
else:
    param_list = ['$\\alpha^{E100-S1350}$', '$\\alpha^{E100-S750}$', '$\\alpha^{E100-S50}$', '$\\alpha^{E50-S150}$']
_plot_1D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, '1D_alpha'))
_plot_2D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, '2D_alpha'), label_size=20, tick_size=18, rotation_x=90)
_heat_map(trace_log_plot, param_list, os.path.join(args.out_dir, 'heatmap_alpha'), dpi=200,
          fig_size=(4.25, 3.75), fontsize_tick=18, nchain=args.nchain, niters=niters)

param_list = ['$[E]^{100}$', '$[E]^{50}$', '$[E]^{25}$']
_plot_1D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, '1D_dE'))
_plot_2D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, '2D_dE'), label_size=24, tick_size=22, rotation_x=90)
_heat_map(trace_log_plot, param_list, os.path.join(args.out_dir, 'heatmap_dE'), dpi=200,
          fig_size=(3.25, 2.75), fontsize_tick=14, nchain=args.nchain, niters=niters)

## --------------------------
# MAP index extraction
with open(os.path.join(args.map_dir, 'map.txt')) as f:
    first_line = f.readline()
map_index = int(first_line.replace("\n", "").split(":")[1])

# Load prior file
prior = pickle.load(open(os.path.join(args.map_dir, 'MAP_prior.pickle'), "rb"))
prior_infor = convert_prior_from_dict_to_list(prior, True, True)
prior_infor_update = check_prior_group(prior_infor, len(inhibitor_list))

map_sampling = pickle.load(open(os.path.join(args.map_dir, 'map.pickle'), "rb"))
alpha_list = {}
for key in map_sampling.keys():
    if key.startswith('alpha'):
        alpha_list[key] = map_sampling[key]

if args.set_lognormal_dE and args.dE>0:
    E_list = {}
    for key in ['dE:100', 'dE:50', 'dE:25']:
        E_list[key] = map_sampling[key]
else: E_list = None

## --------------------------
for idx, inhibitor_name in enumerate(inhibitor_list):
    _dir = inhibitor_name[7:12]

    print("Woking with "+inhibitor_name)
    if not os.path.exists(os.path.join(args.out_dir, _dir)):
        os.mkdir(os.path.join(args.out_dir, _dir))
        print("Create", os.path.join(args.out_dir, _dir))

    expts, expts_plot = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==inhibitor_name)],
                                                multi_var=args.multi_var)

    params_names = {'kcat_DS':              '$k_{cat,DS}$',
                    f'kcat_DSI:{idx+1}':    '$k_{cat,DSI}$',
                    'kcat_DSS':             '$k_{cat,DSS}$',
                    'logK_S_M':             '$logK_{S,M}$',
                    'logK_S_D':             '$logK_{S,D}$',
                    'logK_S_DS':            '$logK_{S,DS}$',
                    f'logK_I_M:{idx+1}':    '$logK_{I,M}$',
                    f'logK_I_D:{idx+1}':    '$logK_{I,D}$',
                    f'logK_I_DI:{idx+1}':   '$logK_{I,DI}$',
                    f'logK_S_DI:{idx+1}':   '$logK_{S,DI}$',
                    'logKd':                '$logK_d$',
                    'log_sigma:ES'  :       'log$\sigma^{ES}$',
                    'log_sigma:ES:0':       'log$\sigma^{ES}$',
                    }
    
    for i in range(len(expts_plot)):
        log_sigma_name = expts_plot[i]['plate']
        log_sigma_name = log_sigma_name.replace("_", "-")
        params_names[f'log_sigma:{_dir}:{i}'] = f'log$\sigma^{log_sigma_name}$'

    # Change parameter names for plotting
    trace_plot = {}
    trace_log_plot = {}
    for key in trace_log.keys():
        if key in params_names.keys():
            trace_plot[key] = trace[key]
            trace_log_plot[params_names[key]] = trace_log[key]
    pickle.dump(trace_log_plot, open(os.path.join(args.out_dir, _dir, "traces_log_plot.pickle"), "wb"))
        
    ## --------------------------
    ## Fitting plot
    params_logK, params_kcat = extract_params_from_map_and_prior(trace_plot, map_index, prior_infor_update)

    n = idx+1
    if args.extended_global:
        if idx == 11:
            plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                              extract_kcat_n_idx(params_kcat, n, shared_params),
                              alpha_list=alpha_list, E_list=E_list, fontsize_tick=15, fontsize_label=20,
                              line_colors=['deeppink', 'blue', 'green', 'orange', 'purple'],
                              fig_size=(6, 4), plot_legend=False, dpi=100,
                              OUTFILE=os.path.join(args.out_dir, _dir, 'ESI'))
        elif idx == 13:
            plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                              extract_kcat_n_idx(params_kcat, n, shared_params),
                              alpha_list=alpha_list, E_list=E_list, fontsize_tick=15, fontsize_label=20,
                              line_colors=['olive'], fig_size=(6, 4), plot_legend=False, dpi=100,
                              OUTFILE=os.path.join(args.out_dir, _dir, 'ESI'))
        elif idx == 14:
            plot_data_conc_log(expts_plot[start:end], extract_logK_n_idx(params_logK, n, shared_params),
                              extract_kcat_n_idx(params_kcat, n, shared_params),
                              alpha_list=alpha_list, E_list=E_list, fontsize_tick=15, fontsize_label=20,
                              line_colors=['gray'],fig_size=(6, 4), plot_legend=False, dpi=100,
                              OUTFILE=os.path.join(args.out_dir, _dir, 'ESI'))
        else:
            plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                               extract_kcat_n_idx(params_kcat, n, shared_params),
                               alpha_list=alpha_list, E_list=E_list,
                               fig_size=(6, 4), plot_legend=False, dpi=100,
                               OUTFILE=os.path.join(args.out_dir, _dir, 'ESI'))
    else:
        plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                           extract_kcat_n_idx(params_kcat, n, shared_params),
                           alpha_list=alpha_list, E_list=E_list,
                           fig_size=(6, 4), plot_legend=False, dpi=100,
                           OUTFILE=os.path.join(args.out_dir, _dir, 'ESI'))

    ## --------------------------
    param_list = ['$logK_{I,M}$', '$logK_{I,D}$', '$logK_{I,DI}$', '$logK_{S,DI}$']
    _plot_1D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, _dir, '1D_logK_EI'))
    param_list = ['$logK_d$', '$logK_{S,M}$', '$logK_{S,D}$', '$logK_{S,DS}$',
                  '$logK_{I,M}$', '$logK_{I,D}$', '$logK_{I,DI}$', '$logK_{S,DI}$']
    _plot_1D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, _dir, '1D_logK'))
    _plot_2D_histogram(trace_log_plot, param_list, label_size=24, tick_size=22, rotation_x=90, 
                       outfile=os.path.join(args.out_dir, _dir, '2D_logK'))
    _heat_map(trace_log_plot, param_list, outfile=os.path.join(args.out_dir, _dir, 'heatmap_logK'),
              nchain=args.nchain, niters=niters)

    ## --------------------------
    param_name = '$k_{cat'
    param_list = [key for key in trace_log_plot.keys() if key.startswith(param_name)]
    _plot_1D_histogram(trace_log_plot, param_list, outfile=os.path.join(args.out_dir, _dir, '1D_kcat'))
    _plot_2D_histogram(trace_log_plot, param_list, outfile=os.path.join(args.out_dir, _dir, '2D_kcat'))
    _heat_map(trace_log_plot, param_list, outfile=os.path.join(args.out_dir, _dir, 'heatmap_kcat'), dpi=200,
              fig_size=(3.25, 2.75), fontsize_tick=14, nchain=args.nchain, niters=niters)

    ## --------------------------
    for key in trace_log.keys():
        if key.startswith('logK_I_DI'): logK_I_DI = trace_log[key]
        if key.startswith('logK_S_DI'): logK_S_DI = trace_log[key]
        if key.startswith('logK_I_D') and not key.startswith('logK_I_DI'): logK_I_D  = trace_log[key]
    
    _linear_corr(logK_I_D, logK_S_DI, 'log$K_{I,D}$', 'log$K_{S,DI}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_1'))
    _linear_corr(logK_I_D, logK_I_DI, 'log$K_{I,D}$', 'log$K_{I,DI}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_2'))
    _linear_corr(logK_S_DI, logK_I_DI, 'log$K_{S,DI}$', 'log$K_{I,DI}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_3'))

    _linear_corr(logK_I_D, logK_S_DI, 'log$K_{I,D}$', 'log$K_{S,DI}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_1_Text'), legend=True)
    _linear_corr(logK_I_D, logK_I_DI, 'log$K_{I,D}$', 'log$K_{I,DI}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_2_Text'), legend=True)
    _linear_corr(logK_S_DI, logK_I_DI, 'log$K_{S,DI}$', 'log$K_{I,DI}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_3_Text'), legend=True)
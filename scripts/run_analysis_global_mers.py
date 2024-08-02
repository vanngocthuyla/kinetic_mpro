import warnings
import numpy as np
import sys
import os
import argparse

import pickle
import json5 as json

import jax
import jax.numpy as jnp

from _trace_analysis import TraceAdjustment, TraceConverter, TraceExtraction
from _plotting import plot_data_conc_log, AnalysisPlot 

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_file",                     type=str,               default="")
parser.add_argument( "--shared_params_infor",           type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--setting",                       type=str,               default="")

args = parser.parse_args()

## Loading files
trace_init    = pickle.load(open(args.mcmc_file, "rb"))
setting       = pickle.load(open(args.setting, "rb"))
if len(args.shared_params_infor)>0:
    shared_params = json.load(open(args.shared_params_infor))
else:
    shared_params = None

## Updating trace by shared parameter information
Adjustment = TraceAdjustment(trace=trace_init, shared_params=shared_params)
no_expt = Adjustment.no_expt

for idx in range(no_expt): 
    os.mkdir(os.path.join(args.out_dir, str(idx)))
    ## Convert from ln to log10
    trace = Adjustment.adjust_extract_ith_params(idx)
    trace_log = TraceConverter(trace=trace).ln_to_log()
    
    # 1D histogram for all logK and kcat
    trace_log_plot = TraceConverter(trace_log).convert_name()
    AnalysisPlot(trace_log_plot).plot_1D_histogram(outfile=os.path.join(args.out_dir, str(idx), '1D'))

    # 2D histogram for all logK and kcat
    for param_type in ['logK', 'kcat']:
        trace_log_plot = TraceConverter(trace_log).convert_name(name_startswith=param_type)
        AnalysisPlot(trace_log_plot).plot_2D_histogram(outfile=os.path.join(args.out_dir, str(idx), f'2D_{param_type}'))
        AnalysisPlot(trace_log_plot).heat_map(nchain=setting['nchain'], niters=setting['niters'], outfile=os.path.join(args.out_dir, str(idx), f'HM_{param_type}'))

    # Liear correlation
    trace_log_plot = TraceConverter(trace_log).convert_name()
    AnalysisPlot(trace_log_plot).linear_corr('$logK_{I,D}$', '$logK_{S,DI}$', outfile=os.path.join(args.out_dir, str(idx), 'Linear_1'))
    AnalysisPlot(trace_log_plot).linear_corr('$logK_{I,D}$', '$logK_{I,DI}$', outfile=os.path.join(args.out_dir, str(idx), 'Linear_2'))
    AnalysisPlot(trace_log_plot).linear_corr('$logK_{S,DI}$', '$logK_{I,DI}$', outfile=os.path.join(args.out_dir, str(idx), 'Linear_3'))

    del trace, trace_log, trace_log_plot

#Plot of alpha and dE
change_names = {}
for _key in trace.keys():
    if _key.startswith("alpha"):
        _new_key = r'{'+_key[6:]+'}'
        _new_key =_new_key.replace("_", "-")
        change_names[_key] = f'$\\alpha^{_new_key}$'
    if _key.startswith("dE"):
        _new_key = r'{'+_key[3:]+'}'
        _new_key =_new_key.replace("_", "-")
        change_names[_key] = f'$[E]^{_new_key}$'

trace = TraceAdjustment(trace=trace_init).adjust_by_shared_params()
trace_log = TraceConverter(trace=trace).ln_to_log()
trace_log_plot = TraceConverter(trace_log).convert_name(change_names)
# pickle.dump(trace_log_plot, open(os.path.join(args.out_dir, "traces_log_plot_alpha_dE.pickle"), "wb"))

param_type = 'alpha'
_plot_1D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, f'1D_{param_type}'))
_plot_2D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, f'2D_{param_type}'), label_size=20, tick_size=18, rotation_x=90)
_heat_map(trace_log_plot, param_list, os.path.join(args.out_dir, f'heatmap_{param_type}'), dpi=200,
          fig_size=(4.25, 3.75), fontsize_tick=18, nchain=setting['nchain'], niters=setting['niters'])

param_type = 'dE'
_plot_1D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, f'1D_{param_type}'))
_plot_2D_histogram(trace_log_plot, param_list, os.path.join(args.out_dir, f'2D_{param_type}'), label_size=24, tick_size=22, rotation_x=90)
_heat_map(trace_log_plot, param_list, os.path.join(args.out_dir, f'heatmap_{param_type}'), dpi=200,
          fig_size=(3.25, 2.75), fontsize_tick=14, nchain=setting['nchain'], niters=setting['niters'])
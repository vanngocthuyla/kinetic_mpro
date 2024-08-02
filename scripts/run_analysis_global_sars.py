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
setting = pickle.load(open(args.setting, "rb"))
if len(args.shared_params_infor)>0:
    shared_params = json.load(open(args.shared_params_infor))
else:
    shared_params = None

## Updating trace by shared parameter information
trace = TraceAdjustment(trace=trace_init, shared_params=shared_params).adjust_by_shared_params()

## Convert from ln to log10
trace_log = TraceConverter(trace=trace).ln_to_log()

## Change name for plotting
change_names_mut = {'logKd:0':      '$logK_d^{Mut}$',
                    'logK_S_M:0':   '$logK_{S,M}^{Mut}$',
                    'logK_I_M:0':   '$logK_{I,M}^{Mut}$',
                    'kcat_DS:0' :   '$kcat_{DS}^{Mut}$',
                    'kcat_DSI:0' :  '$kcat_{DSI}^{Mut}$',
                    'kcat_DSS:0' :  '$kcat_{DSS}^{Mut}$'}
change_names_WT = {'logKd:1':      '$logK_d^{WT}$',
                   'kcat_DS:1' :   '$kcat_{DS}^{WT}$',
                   'kcat_DSI:1' :  '$kcat_{DSI}^{WT}$',
                   'kcat_DSS:1' :  '$kcat_{DSS}^{WT}$',
                    }
change_names = {**change_names_mut, **change_names_WT}

# 1D histogram for all logK and kcat
trace_log_plot = TraceConverter(trace_log).convert_name(change_names)
# pickle.dump(trace_log_plot, open(os.path.join(args.out_dir, "traces_log_plot.pickle"), "wb"))
AnalysisPlot(trace_log_plot, dpi=100).plot_1D_histogram(outfile=os.path.join(args.out_dir, '1D'))

# 2D histogram for all logK and kcat
for _change_names, mpro_type in zip([change_names_mut, change_names_WT], ['mut', 'WT']):
    for param_type in ['logK', 'kcat']:
        trace_log_plot = TraceConverter(trace_log).convert_name(_change_names, param_type)
        AnalysisPlot(trace_log_plot).plot_2D_histogram(outfile=os.path.join(args.out_dir, f'2D_{param_type}_{mpro_type}'))
        AnalysisPlot(trace_log_plot).heat_map(nchain=setting['nchain'], niters=setting['niters'], outfile=os.path.join(args.out_dir, f'HM_{param_type}_{mpro_type}'))

# Liear correlation 
trace_log_plot = TraceConverter(trace_log).convert_name(change_names)
AnalysisPlot(trace_log_plot).linear_corr('$logK_{I,D}$', '$logK_{S,DI}$', outfile=os.path.join(args.out_dir, 'Linear_1'))
AnalysisPlot(trace_log_plot).linear_corr('$logK_{I,D}$', '$logK_{I,DI}$', outfile=os.path.join(args.out_dir, 'Linear_2'))
AnalysisPlot(trace_log_plot).linear_corr('$logK_{S,DI}$', '$logK_{I,DI}$', outfile=os.path.join(args.out_dir, 'Linear_3'))

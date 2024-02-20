import warnings
import numpy as np
import sys
import os
from glob import glob
import argparse

import pickle
import arviz as az
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from _load_data_mers import load_data_one_inhibitor
from _plotting import plot_data_conc_log

from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, _trace_ln_to_log

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

parser = argparse.ArgumentParser()

parser.add_argument( "--kinetic_file",                  type=str,               default="")
parser.add_argument( "--mcmc_file",                     type=str,               default="")
parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="")

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--niters",                        type=int,               default=0)

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

# 1D Histogram
def _plot_1D_histogram(trace, param_name, outfile, text_size=24, dpi=150):
    trace_plot = {}
    for key in trace.keys():
        if key.startswith(param_name):
            trace_plot[key] = trace[key]

    az.plot_posterior(trace_plot, textsize=text_size);
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')

def _plot_2D_histogram(trace, param_name, outfile, text_size=40, dpi=150):
    plt.figure()
    traces = {}
    for key in trace.keys():
        if key.startswith(param_name):
            traces[key] = np.reshape(trace[key], (args.nchain, niters))
    data = az.convert_to_inference_data(traces)
    az.plot_pair(data, kind='kde', textsize=text_size, divergences=True)
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight');

def _heat_map(trace, param_name, outfile, dpi=150):
    plt.figure()
    traces = {}
    for key in trace.keys():
        if key.startswith(param_name):
            traces[key] = np.reshape(trace[key], (args.nchain, niters))
    data = az.convert_to_inference_data(traces)
    data_plot = az.InferenceData.to_dataframe(data)
    data_plot = data_plot.drop(['chain', 'draw'], axis=1)
    sns.heatmap(data_plot.corr(), annot=True, fmt="0.2f", linewidth=.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight');


#Linear correlation
def _linear_corr(xdata, ydata, xlabel, ylabel, outfile=None, fontsize_tick=18, fontsize_text=12):
    plt.figure()
    ax = plt.axes()
    res = stats.linregress(xdata, ydata)
    ax.plot(xdata, ydata, 'o', label='data')
    ax.plot(xdata, res.intercept + res.slope*xdata, 'k', label='Fitted line')
    ax.set_xlabel(xlabel, fontsize=fontsize_tick)
    ax.set_ylabel(ylabel, fontsize=fontsize_tick)
    ax.text(0.2, 0.9, ylabel+'='+str(round(res.slope, 2))+'*'+xlabel+'+'+str(round(res.intercept, 2)),
            fontsize=fontsize_text, transform=ax.transAxes);
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight');

#Plot of alpha and dE
params_names = {'alpha:E100_S1350':  '$\\alpha^{E:100nM, S:1350nM}$',
                'alpha:E100_S50':    '$\\alpha^{E:100nM, S:50nM}$',
                'alpha:E100_S750':   '$\\alpha^{E:100nM, S:750nM}$',
                'alpha:E50_S150':    '$\\alpha^{E:50nM, S:150nM}$',
                'dE:100':       '$[E]^{100nM}$',
                'dE:50':        '$[E]^{50nM}$',
                'dE:25':        '$[E]^{25nM}$',
                }

# Change parameter names for plotting
trace_log_plot = {}
for key in trace_log.keys():
    if key in params_names.keys():
        trace_log_plot[params_names[key]] = trace_log[key]
pickle.dump(trace_log_plot, open(os.path.join(args.out_dir, "traces_log_plot_alpha_dE.pickle"), "wb"))

_plot_1D_histogram(trace_log_plot, '$\\alpha', os.path.join(args.out_dir, '1D_alpha'), text_size=28)
_plot_2D_histogram(trace_log_plot, '$\\alpha', os.path.join(args.out_dir, '2D_alpha'), text_size=24)
_heat_map(trace_log_plot, '$\\alpha', os.path.join(args.out_dir, 'heatmap_alpha'))

_plot_1D_histogram(trace_log_plot, '$[E]', os.path.join(args.out_dir, '1D_dE'), text_size=18)
_plot_2D_histogram(trace_log_plot, '$[E]', os.path.join(args.out_dir, '2D_dE'), text_size=24)
_heat_map(trace_log_plot, '$[E]', os.path.join(args.out_dir, 'heatmap_dE'))

# MAP index extraction
with open(os.path.join(args.mcmc_dir, 'map.txt')) as f:
    first_line = f.readline()
map_index = int(first_line.replace("\n", "").split(":")[1])

# Load prior file
df = pd.read_csv(os.path.join(args.mcmc_dir, 'Prior_infor.csv'))
prior_infor_update = []
for index, row in df.iterrows():
    prior_infor_update.append(row.to_dict())

map_sampling = pickle.load(open(os.path.join(args.mcmc_dir, 'map.pickle'), "rb"))
alpha_list = {}
for key in map_sampling.keys():
    if key.startswith('alpha'):
        alpha_list[key] = map_sampling[key]

if args.set_lognormal_dE and args.dE>0:
    E_list = {}
    for key in ['dE:100', 'dE:50', 'dE:25']:
        E_list[key] = map_sampling[key]
else: E_list = None

for idx, inhibitor_name in enumerate(inhibitor_list):
    _dir = inhibitor_name[8:12]

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
                    f'log_sigma:{_dir}:0':  'log$\sigma^{E:100nM, S:50nM}$',
                    f'log_sigma:{_dir}:1':  'log$\sigma^{E:100nM, S:50nM}$',
                    f'log_sigma:{_dir}:2':  'log$\sigma^{E:100nM, S:750nM}$',
                    f'log_sigma:{_dir}:3':  'log$\sigma^{E:50nM, S:150nM}$'
                    }

    # Change parameter names for plotting
    trace_plot = {}
    trace_log_plot = {}
    for key in trace_log.keys():
        if key in params_names.keys():
            trace_plot[key] = trace[key]
            trace_log_plot[params_names[key]] = trace_log[key]
    pickle.dump(trace_log_plot, open(os.path.join(args.out_dir, _dir, "traces_log_plot.pickle"), "wb"))
        
    ## Fitting plot
    params_logK, params_kcat = extract_params_from_map_and_prior(trace_plot, map_index, prior_infor_update)

    n = idx+1
    plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                       extract_kcat_n_idx(params_kcat, n, shared_params),
                       alpha_list=alpha_list, E_list=E_list,
                       OUTFILE=os.path.join(args.out_dir, _dir, 'ESI'))

    _plot_1D_histogram(trace_log_plot, '$logK_', os.path.join(args.out_dir, _dir, '1D_logK'), text_size=24)
    _plot_2D_histogram(trace_log_plot, '$logK_', os.path.join(args.out_dir, _dir, '2D_logK'))
    _heat_map(trace_log_plot, '$logK_', os.path.join(args.out_dir, _dir, 'heatmap_logK'))

    _plot_1D_histogram(trace_log_plot, '$k_{cat', os.path.join(args.out_dir, _dir, '1D_kcat'), text_size=32)
    _plot_2D_histogram(trace_log_plot, '$k_{cat', os.path.join(args.out_dir, _dir, '2D_kcat'))
    _heat_map(trace_log_plot, '$k_{cat', os.path.join(args.out_dir, _dir, 'heatmap_kcat'))

    for key in trace_log.keys():
        if key.startswith('logK_I_DI'): logK_I_DI = trace_log[key]
        if key.startswith('logK_S_DI'): logK_S_DI = trace_log[key]
        if key.startswith('logK_I_D') and not key.startswith('logK_I_DI'): logK_I_D  = trace_log[key]

    _linear_corr(logK_I_DI, logK_S_DI, 'log$K_{I,DI}$', 'log$K_{S,DI}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_1'))
    _linear_corr(logK_S_DI, logK_I_D, 'log$K_{S,DI}$', 'log$K_{I,D}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_2'))
    _linear_corr(logK_I_DI, logK_I_D, 'log$K_{I,DI}$', 'log$K_{I,D}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_3'))
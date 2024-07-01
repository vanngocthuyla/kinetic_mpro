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
parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--dE_alpha_file",                 type=str,               default="")

parser.add_argument( "--multi_var",                     action="store_true",    default=True)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nchain",                        type=int,               default=4)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

args = parser.parse_args()

_mcmc_dir = glob(os.path.join(args.mcmc_dir, "*"), recursive = True)
_mcmc_dir = [os.path.basename(f) for f in _mcmc_dir]
_mcmc_dir.sort()

df_mers = pd.read_csv(args.kinetic_file)
inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

if len(args.dE_alpha_file)>0 and os.path.isfile(args.dE_alpha_file):
    dE_alpha = pickle.load(open(args.dE_alpha_file, "rb"))
    
    if args.set_K_S_DS_equal_K_S_D: 
        dE_alpha['logK_S_DS'] = dE_alpha['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        dE_alpha['logK_S_DI'] = dE_alpha['logK_S_DS']

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'alpha:E100_S1350', 'alpha:E100_S50', 'alpha:E100_S750', 'alpha:E50_S150',
                'kcat_DS', 'kcat_DSS', 'dE:100', 'dE:50', 'dE:25']:
        assert key in dE_alpha.keys(), f"Please provide {key} in map .pickle file."

shared_params = None

for _dir in _mcmc_dir:
    if not os.path.isdir(os.path.join(args.mcmc_dir, _dir)):
        continue

    print("Woking with ASAP-00"+_dir)
    if not os.path.exists(os.path.join(args.out_dir, _dir)):
        os.mkdir(os.path.join(args.out_dir, _dir))
        print("Create", os.path.join(args.out_dir, _dir))

    inhibitor_name = 'ASAP-00'+_dir
    expts, expts_plot = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==inhibitor_name)],
                                                multi_var=args.multi_var)

    # Convert from ln to log10
    trace = pickle.load(open(os.path.join(args.mcmc_dir, _dir, 'traces.pickle'), "rb"))
    trace_log = _trace_ln_to_log(trace, group_by_chain=False, nchain=args.nchain)
    pickle.dump(trace_log, open(os.path.join(args.out_dir, _dir, "traces_log.pickle"), "wb"))

    params_names = {'kcat_DS':      '$k_{cat,DS}$',
                    'kcat_DSI':     '$k_{cat,DSI}$',
                    'kcat_DSI:1':   '$k_{cat,DSI}$',
                    'kcat_DSS':     '$k_{cat,DSS}$',
                    'logK_S_M':     '$logK_{S,M}$',
                    'logK_S_D':     '$logK_{S,D}$',
                    'logK_S_DS':    '$logK_{S,DS}$',
                    'logK_I_M:1':   '$logK_{I,M}$',
                    'logK_I_D:1':   '$logK_{I,D}$',
                    'logK_I_DI:1':  '$logK_{I,DI}$',
                    'logK_S_DI:1':  '$logK_{S,DI}$',
                    'logK_I_M':     '$logK_{I,M}$',
                    'logK_I_D':     '$logK_{I,D}$',
                    'logK_I_DI':    '$logK_{I,DI}$',
                    'logK_S_DI':    '$logK_{S,DI}$',
                    'logKd':        '$logK_d$',
                    'alpha:E100_S1350':  '$\\alpha^{E:100nM, S:1350nM}$',
                    'alpha:E100_S50':    '$\\alpha^{E:100nM, S:50nM}$',
                    'alpha:E100_S750':   '$\\alpha^{E:100nM, S:750nM}$',
                    'alpha:E50_S150':    '$\\alpha^{E:50nM, S:150nM}$',
                    'alpha:E50_S550':    '$\\alpha^{E:50nM, S:550nM}$',
                    'dE:100':       '$\Delta E^{100nM}$',
                    'dE:50':        '$\Delta E^{50nM}$',
                    'dE:25':        '$\Delta E^{25nM}$',
                    'log_sigma:ES'  :       'log$\sigma^{ES}$',
                    'log_sigma:ES:0':       'log$\sigma^{ES}$',
                    'log_sigma:ESI:0':      'log$\sigma^{E:100nM, S:50nM}$',
                    'log_sigma:ESI:1':      'log$\sigma^{E:100nM, S:50nM}$',
                    'log_sigma:ESI:2':      'log$\sigma^{E:100nM, S:750nM}$',
                    'log_sigma:ESI:3':      'log$\sigma^{E:50nM, S:150nM}$',
                    f'log_sigma:{_dir}:0':  'log$\sigma^{E:100nM, S:50nM}$',
                    f'log_sigma:{_dir}:1':  'log$\sigma^{E:100nM, S:50nM}$',
                    f'log_sigma:{_dir}:2':  'log$\sigma^{E:100nM, S:750nM}$',
                    f'log_sigma:{_dir}:3':  'log$\sigma^{E:50nM, S:150nM}$'
                    }
    # Change parameter names for plotting
    trace_log_plot = {}
    for key in trace_log.keys():
        if key in params_names.keys():
            trace_log_plot[params_names[key]] = trace_log[key]
    pickle.dump(trace_log_plot, open(os.path.join(args.out_dir, _dir, "traces_log_plot.pickle"), "wb"))

    try:
        # MAP index extraction
        with open(os.path.join(args.mcmc_dir, _dir, 'map.txt')) as f:
            first_line = f.readline()
        map_index = int(first_line.replace("\n", "").split(":")[1])

        # Load prior file
        prior = pickle.load(open(os.path.join(args.mcmc_dir, _dir, 'Prior.pickle'), "rb"))
        assert (args.set_K_S_DS_equal_K_S_D and 'logK_S_DS' in prior.keys()) or (args.set_K_S_DI_equal_K_S_DS and 'logK_S_DI' in prior.keys()), "The constraint was incorrect."

        prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
        prior_infor_update = check_prior_group(prior_infor, len(expts))
        pd.DataFrame(prior_infor_update).to_csv(os.path.join(args.out_dir, "Prior_infor.csv"), index=False)

        ## Fitting plot
        params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

        E_list = {}
        for key in ['dE:100', 'dE:50', 'dE:25']:
            E_list[key] = dE_alpha[key]

        alpha_list = {}
        for key in dE_alpha.keys():
            if key.startswith('alpha'):
                alpha_list[key] = dE_alpha[key]

        n = 0
        plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                           extract_kcat_n_idx(params_kcat, n, shared_params),
                           alpha_list=alpha_list, E_list=E_list,
                           OUTFILE=os.path.join(args.out_dir, _dir, 'ESI'))
    except:
        pass

    # 1D Histogram
    def _plot_1D_histogram(trace, param_name, outfile, text_size=24, dpi=150):
        trace_plot = {}
        for key in trace.keys():
            if key.startswith(param_name):
                trace_plot[key] = trace[key]

        az.plot_posterior(trace_plot, textsize=text_size);
        plt.savefig(outfile, dpi=dpi)

    _plot_1D_histogram(trace_log_plot, '$logK_', os.path.join(args.out_dir, _dir, '1D_logK'), text_size=24)
    _plot_1D_histogram(trace_log_plot, '$k_{cat', os.path.join(args.out_dir, _dir, '1D_kcat'), text_size=12)

    def _plot_2D_histogram(trace, param_name, outfile, text_size=40):
        plt.figure()
        traces = {}
        for key in trace.keys():
            if key.startswith(param_name):
                traces[key] = np.reshape(trace[key], (args.nchain, args.niters))
        data = az.convert_to_inference_data(traces)
        az.plot_pair(data, kind='kde', textsize=text_size, divergences=True)
        plt.savefig(outfile);

    def _heat_map(trace, param_name, outfile, dpi=150):
        plt.figure()
        traces = {}
        for key in trace.keys():
            if key.startswith(param_name):
                traces[key] = np.reshape(trace[key], (args.nchain, args.niters))
        data = az.convert_to_inference_data(traces)
        data_plot = az.InferenceData.to_dataframe(data)
        data_plot = data_plot.drop(['chain', 'draw'], axis=1)
        sns.heatmap(data_plot.corr(), annot=True, fmt="0.2f", linewidth=.5)
        plt.tight_layout()
        plt.savefig(outfile, dpi=dpi);

    _plot_2D_histogram(trace_log_plot, '$logK_', os.path.join(args.out_dir, _dir, '2D_logK'))
    _heat_map(trace_log_plot, '$logK_', os.path.join(args.out_dir, _dir, 'heatmap_logK'))

    #Linear correlation

    def _linear_corr(xdata, ydata, xlabel, ylabel, outfile, fontsize_tick=18, fontsize_text=12):
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
        plt.savefig(outfile);

    for key in trace_log.keys():
        if key.startswith('logK_I_DI'): logK_I_DI = trace_log[key]
        if key.startswith('logK_S_DI'): logK_S_DI = trace_log[key]
        if key.startswith('logK_I_D') and not key.startswith('logK_I_DI'): logK_I_D  = trace_log[key]

    _linear_corr(logK_I_DI, logK_S_DI, 'log$K_{I,DI}$', 'log$K_{S,DI}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_1'))
    _linear_corr(logK_S_DI, logK_I_D, 'log$K_{S,DI}$', 'log$K_{I,D}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_2'))
    _linear_corr(logK_I_DI, logK_I_D, 'log$K_{I,DI}$', 'log$K_{I,D}$', os.path.join(args.out_dir, _dir, 'Linear_Correlation_3'))
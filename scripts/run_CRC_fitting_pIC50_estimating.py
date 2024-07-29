"""
This file is used to fit to one CRC. First, it checks if there is outlier(s) in the CRC. 
Then, model is fitted. If the model converges, pIC50 is estimated. Otherwise, more samples can be 
generated and all the samples of multiple runnings are combined before checking the convergence again.
"""

import warnings
import numpy as np
import sys
import os
import shutil
from glob import glob
import argparse

import pickle
import arviz as az
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro

import matplotlib.pyplot as plt

from pymbar import timeseries

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore")

from _load_data_mers import load_data_one_inhibitor

from _define_model import Model
from _CRC_fitting import _run_mcmc_CRC, _expt_check_noise_trend

from _MAP_mpro import _map_running
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, _trace_convergence, _convergence_rhat
from _plotting import plot_data_conc_log, plotting_trace

from _pIC50 import _adjust_trace, _pIC_hill

from _save_setting import save_model_setting

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="")
parser.add_argument( "--prior_infor",                   type=str,               default="")
parser.add_argument( "--shared_params_infor",           type=str,               default="")
parser.add_argument( "--initial_values",                type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=1000)
parser.add_argument( "--nburn",                         type=int,               default=200)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

parser.add_argument( "--outlier_removal",               action="store_true",    default=False)
parser.add_argument( "--exclude_first_trace",           action="store_true",    default=False)
parser.add_argument( "--key_to_check",                  type=str,               default="")
parser.add_argument( "--converged_samples",             type=int,               default=500)

parser.add_argument( "--enzyme_conc_nM",                type=float,             default="100")
parser.add_argument( "--substrate_conc_nM",             type=float,             default="1350")

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

### Data
df_mers = pd.read_csv(args.input_file)

inhibitor_name = args.name_inhibitor.split()
for i, name in enumerate(inhibitor_name):
    expts_init, expts_plot = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==name+'-001')*(df_mers['Drop']!=1.0)],
                                                     multi_var=args.multi_var)

## Outlier detection and trend checking
[expts_outliers, outliers, _, _] = _expt_check_noise_trend(expts_init)
if args.outlier_removal:
    expts = expts_outliers.copy()
else:
    expts = expts_init.copy()

if len(args.initial_values)>0 and os.path.isfile(args.initial_values):
    map_sampling = pickle.load(open(args.initial_values, "rb"))
    logK_dE_alpha = map_sampling

    if args.set_K_S_DS_equal_K_S_D:
        map_sampling['logK_S_DS'] = map_sampling['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        map_sampling['logK_S_DI'] = map_sampling['logK_S_DS']

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'kcat_DS', 'kcat_DSS']:
        assert key in map_sampling.keys(), f"Please provide {key} in map_file."
else:
    logK_dE_alpha = None

## Create a model to run
model = Model(len(expts))
model.check_model(args)
traces_name = model.args.traces_name

os.chdir(args.out_dir)

### Concentration for the estimation of dimer-only pIC50
init_logMtot = np.log(args.enzyme_conc_nM*1E-9)
init_logStot = np.log(args.substrate_conc_nM*1E-9)
init_logDtot = init_logMtot-np.log(2)

n_points = 50
min_conc = 1E-12
max_conc = 1E-3

logDtot = np.ones(n_points)*init_logDtot
logStot = np.ones(n_points)*init_logStot
logItot = np.linspace(np.log(min_conc), np.log(max_conc), n_points)

if len(args.key_to_check)>0:
    key_to_check = args.key_to_check.split()
else:
    key_to_check = ''

if not os.path.isdir(os.path.join(args.out_dir, 'Convergence')):
    os.mkdir(os.path.join(args.out_dir, 'Convergence'))

no_limit = 10
no_running = 1

print(f"\nAnalyzing {inhibitor_name[0]}")

while no_running<=no_limit:

    ### Fitting
    if not os.path.isdir(os.path.join(args.out_dir, f'sampling_{no_running}')):
        os.mkdir(os.path.join(args.out_dir, f'sampling_{no_running}'))

    name_expt = inhibitor_name[0]
    expt_dir = os.path.join(args.out_dir, f'sampling_{no_running}', name_expt)
    last_dir = os.path.join(args.out_dir, f'sampling_{no_running-1}', name_expt)
    
    if not os.path.isfile(os.path.join(expt_dir, traces_name+'.pickle')):
        if not os.path.isdir(expt_dir):
            os.mkdir(expt_dir)

        mes = f'\nFitting sampling_{no_running}/{name_expt}:'
        print(mes)
        with open(os.path.join(expt_dir, "log.txt"), "a") as f:
            print(mes, file=f)

        if no_running>1 and os.path.isfile(os.path.join(last_dir, "Last_state.pickle")):
            if no_running==2 and args.exclude_first_trace:
                init_values = pickle.load(open(os.path.join(last_dir, "map.pickle"), 'rb'))
                mes = f"Initial values: {init_values}"
                print(mes)
                with open(os.path.join(expt_dir, "log.txt"), "a") as f:
                    print(mes, file=f)

                trace = _run_mcmc_CRC(expts, model.prior_infor, model.shared_params, init_values, '', expt_dir, model.args)

            else:
                last_state = pickle.load(open(os.path.join(last_dir, "Last_state.pickle"), "rb"))
                mes = "Keep running from last state."
                print(mes)
                with open(os.path.join(expt_dir, "log.txt"), "a") as f:
                    print(mes, file=f)

                trace = _run_mcmc_CRC(expts, model.prior_infor, model.shared_params, None, last_dir, expt_dir, model.args)
        else:
            trace = _run_mcmc_CRC(expts, model.prior_infor, model.shared_params, None, last_dir, expt_dir, model.args)

        ## Finding MAP
        [trace_map, map_index] = _map_running(trace.copy(), expts, model.prior_infor, model.shared_params, model.args)

        ## Fitting plot
        params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, model.prior_infor)

        alpha_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('alpha')}
        E_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('dE')}

        n = 0
        plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, model.shared_params),
                           extract_kcat_n_idx(params_kcat, n, model.shared_params),
                           alpha_list=alpha_list, E_list=E_list, outliers=outliers,
                           OUTFILE=os.path.join(expt_dir,'EI'))
            
        ## Saving the model fitting condition
        save_model_setting(args, OUTDIR=expt_dir, OUTFILE='setting.pickle')
        
        del trace, trace_map

    ### Extracting all traces.pickles of one experiment from multiple sampling runs
    if not os.path.isdir(os.path.join(args.out_dir, 'Convergence', name_expt)):
        os.mkdir(os.path.join(args.out_dir, 'Convergence', name_expt))
    
    _trace_files = [f'{args.out_dir}/sampling_{i}/{name_expt}/{traces_name}.pickle' for i in range(1, no_running+1)]
    if args.exclude_first_trace and len(_trace_files)>1: 
        trace_files = _trace_files[1:]
    else:
        trace_files = _trace_files

    [trace, flag, nchain_updated] = _trace_convergence(mcmc_files=trace_files, out_dir=os.path.join(args.out_dir, 'Convergence', name_expt), 
                                                       nchain=args.nchain, expected_nsample=args.converged_samples,
                                                       key_to_check=key_to_check, one_chain_removal=True)

    if flag: #if number of converged samples returned from pymbar is enough
        data = az.InferenceData.to_dataframe(az.convert_to_inference_data(trace))
        
        ### Estimating dimer-only pIC50
        _nthin = int(len(data)/100)
        if logK_dE_alpha is not None:
            df = _adjust_trace(data.iloc[::_nthin, :].copy(), logK_dE_alpha)
        else:
            df = data.iloc[::_nthin, :].copy()

        thetas = _pIC_hill(df, logDtot, logStot, logItot)
        pIC50_list = thetas[2]
        hill_list = thetas[3]
        
        if _convergence_rhat(trace=pIC50_list, nchain=nchain_updated, digit=1):
            
            pIC50 = pIC50_list[hill_list>0]
            mes = "pIC50: %0.3f" % np.median(pIC50) + " +- %0.3f" % np.std(pIC50) + "\n"
            
            print(mes)
            with open(os.path.join(args.out_dir, 'Convergence', name_expt, "log.txt"), "a") as f:
                print(mes, file=f)
            
            pickle.dump(trace, open(os.path.join(args.out_dir, 'Convergence', name_expt, "traces.pickle"), "wb"))
            plotting_trace(trace, os.path.join(args.out_dir, 'Convergence', name_expt), nchain_updated)
            
            break

    del trace
    no_running += 1

if no_running > no_limit:
    mes = "The number of fitting was exceeded."
    print(mes)
    with open(os.path.join(args.out_dir, 'Convergence', name_expt, "log.txt"), "a") as f:
        print(mes, file=f)
else:
    os.chdir(expt_dir)
    MAP_figs = glob('EI*', recursive=False)
    for fig in MAP_figs:
        shutil.copy(os.path.join(expt_dir, fig), os.path.join(args.out_dir, 'Convergence', name_expt, fig))

[expts_outliers, _, mes_noise, mes_trend] = _expt_check_noise_trend(expts_init, OUT_DIR=os.path.join(args.out_dir, 'Convergence', name_expt))

## Report if curve is increasing or decreasing
if mes_noise is not None and len(mes_noise)>0:
    for _mes in mes_noise:
        print(_mes)
        with open(os.path.join(args.out_dir, 'Convergence', name_expt, "log.txt"), "a") as f:
            print(_mes, file=f)

if mes_trend is not None and len(mes_trend)>0:
    for _mes in mes_trend:
        print(_mes)
        with open(os.path.join(args.out_dir, 'Convergence', name_expt, "log.txt"), "a") as f:
            print(_mes, file=f)
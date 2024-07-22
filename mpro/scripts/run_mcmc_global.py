"""
This code is designed to fit the model that involves enzyme - substrate - inhibitors. 
Posterior from previous steps can be used to assign the prior for this global fitting of multiple inhibitors. 
"""

import warnings
import os
import numpy as np
import pandas as pd
import argparse

import pickle
import matplotlib.pyplot as plt

import jax
import numpyro

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _load_data_mers import load_data_no_inhibitor, load_data_one_inhibitor

from _define_model import Model
from _model_fitting import _run_mcmc

from _MAP_mpro import _map_running
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, _trace_one_to_nchain
from _plotting import plot_data_conc_log, plotting_trace_global

from _save_setting import save_model_setting

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="")
parser.add_argument( "--prior_infor",                   type=str,               default="")
parser.add_argument( "--shared_params_infor",           type=str,               default="")
parser.add_argument( "--initial_values",                type=str,               default="")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--list_inhibitor",                type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--fixing_log_sigmas",             action="store_true",    default=False)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

df_mers = pd.read_csv(args.input_file)
if len(args.list_inhibitor)>0:
    inhibitor_name = np.unique(args.list_inhibitor.split())
else:
    inhibitor_name = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

expts_no_I, expts_plot_no_I = load_data_no_inhibitor(df_mers[df_mers['Inhibitor (nM)']==0.0], 
                                                     multi_var=args.multi_var)
no_expt = [len(expts_plot_no_I)]
expts = expts_no_I
expts_plot = expts_plot_no_I

for i, name in enumerate(inhibitor_name):
    expts_, expts_plot_ = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==name)*(df_mers['Drop']!=1)],
                                                  multi_var=args.multi_var)
    expts = expts + expts_
    expts_plot = expts_plot + expts_plot_
    no_expt.append(len(expts_plot_))

## Prior and shared parameters
logKd_min = -27.63
logKd_max = 0.
kcat_min = 0.
kcat_max = 20

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': 'normal', 'loc': -9.9, 'scale': 0.5}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': -14, 'upper': 0} 
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': -20.72, 'upper': -13}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': -20.72, 'upper': 0.}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [logKd_min,-17.05,-12.95,-13.93,-16.91,-13.02,-17.46,-13.09,-12.01,-13.86,-14.55,-14.05,-12.57,-16.29], 'upper': 0}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [logKd_min,-23.79,-18.92,-19.79,-22.71,-20.25,-24.02,-24.,-18.96,-20.24,-20.94,-20.31,-18.32,-22.48], 'upper': [0,-3.44,-3.63,-5.92,-9.38,-4.47,-3.71,-10.16,-3.05,-6.63,-6.55,-5.89,-4.05,-2.97]}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [logKd_min,-27.63,-27.63,-27.63,-27.63,-27.63,-27.63,-27.63,-27.63,-27.63,-27.63,-27.62,-27.63,-27.63], 'upper': [0,0,-14.64,-16.11,-1.5,-14.84,0,-16.81,-15.35,-16.12,-15.95,-15.69,-15.38,0]}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [logKd_min,-27.61,-26.74,-24.66,-23.69,-26.53,-27.63,-23.48,-26.68,-24.67,-25.1,-25.51,-26.37,-27.6], 'upper': [0,-9.45,-11.25,-11.45,-11.63,-11.31,-10.22,-12.56,-11.75,-11.72,-11.34,-11.38,-11.27,-10.42]}

# prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0., 'lower': kcat_min, 'upper': kcat_max}

## Create a model to run
model = Model(len(expts), prior)
model.check_model(args)

## Fitting model
trace = _run_mcmc(expts, model.prior_infor, model.shared_params, model.init_values, model.args)

## Trace and autocorrelation plots
plotting_trace_global(trace=trace, out_dir=os.path.join(args.out_dir, 'Trace_plot'), nchain=args.nchain)

## Finding MAP
[trace_map, map_index] = _map_running(trace.copy(), expts, model.prior_infor, model.shared_params, model.args)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace_map, map_index, model.prior_infor)

if args.set_lognormal_dE and args.dE>0:
    E_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('dE')}
else: E_list = None

alpha_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('alpha')}
if len(alpha_list) == 0:
    alpha_list = None

for n in range(len(expts)):
    if args.set_K_S_DS_equal_K_S_D:
        try: params_logK[f'logK_S_DS:{n}'] = params_logK[f'logK_S_D:{n}']
        except: params_logK['logK_S_DS'] = params_logK['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        try: params_logK[f'logK_S_DI:{n}'] = params_logK[f'logK_S_DS:{n}']
        except: params_logK['logK_S_DI'] = params_logK['logK_S_DS']

if not os.path.isdir('Fitting'):
    os.mkdir('Fitting')

n = 0
plot_data_conc_log(expts_plot_no_I, extract_logK_n_idx(params_logK, n, model.shared_params),
                   extract_kcat_n_idx(params_kcat, n, model.shared_params),
                   line_colors=['black', 'red', 'tab:brown'], ls='-.',
                   E_list=E_list, plot_legend=True, combined_plots=True,
                   OUTFILE=os.path.join(args.out_dir, 'Fitting', 'ES'))

start = 0
end = 3
for i in range(len(inhibitor_name)):
    n = i + 1
    start = end
    end   = end+no_expt[n]
    plot_data_conc_log(expts_plot[start:end], extract_logK_n_idx(params_logK, n, model.shared_params),
                       extract_kcat_n_idx(params_kcat, n, model.shared_params),
                       alpha_list=alpha_list, E_list=E_list,
                       plot_legend=True, combined_plots=True,
                       OUTFILE=os.path.join(args.out_dir, 'Fitting', 'ESI_'+str(i)))

## Saving the model fitting condition
save_model_setting(model.args, OUTDIR=args.out_dir, OUTFILE='setting.pickle')
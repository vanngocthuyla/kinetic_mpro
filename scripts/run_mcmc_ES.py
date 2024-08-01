"""
This code is designed to fit the model that involves only enzyme and substrate.
"""

import warnings
import os
import pandas as pd
import argparse

import pickle
import matplotlib.pyplot as plt

import jax
import numpyro

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _load_data_mers import load_data_no_inhibitor

from _define_model import Model
from _model_fitting import _run_mcmc

from _MAP_mpro import _map_running
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, _trace_one_to_nchain
from _plotting import plot_data_conc_log

from _save_setting import save_model_setting

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--prior_infor",                   type=str,               default="")
parser.add_argument( "--shared_params_infor",           type=str,               default="")
parser.add_argument( "--initial_values",                type=str,               default="")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--niters",				        type=int, 				default=10000)
parser.add_argument( "--nburn",                         type=int, 				default=2000)
parser.add_argument( "--nthin",                         type=int, 				default=1)
parser.add_argument( "--nchain",                        type=int, 				default=4)
parser.add_argument( "--random_key",                    type=int, 				default=0)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

df_mers = pd.read_csv(args.input_file)
expts_no_I, expts_plot_no_I = load_data_no_inhibitor(df_mers, args.multi_var)

expts = expts_no_I

## Create a model to run
model = Model(len(expts))
model.check_model(args)

## Fitting model
trace = _run_mcmc(expts=expts, prior_infor=model.prior_infor, shared_params=model.shared_params, 
                  init_values=model.init_values, args=model.args)

## Finding MAP
[trace_map, map_index] = _map_running(trace=trace.copy(), expts=expts, prior_infor=model.prior_infor, 
                                      shared_params=model.shared_params, args=model.args)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace_map, map_index, model.prior_infor)

if args.set_lognormal_dE and args.dE>0:
    E_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('dE')}
else: E_list = None

n = 0
plot_data_conc_log(experiments=expts_plot_no_I, 
                   params_logK=extract_logK_n_idx(params_logK, n, model.shared_params),
                   params_kcat=extract_kcat_n_idx(params_kcat, n, model.shared_params),
                   line_colors=['black', 'red', 'tab:brown'], ls='-.',
                   E_list=E_list, plot_legend=True, combined_plots=True,
                   fontsize_tick=12, fontsize_label=18,
                   OUTFILE=os.path.join(args.out_dir,'ES'))

## Saving the model fitting condition
save_model_setting(model.args, OUTDIR=args.out_dir, OUTFILE='setting.pickle')
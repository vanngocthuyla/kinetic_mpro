"""
This code is designed to run MCMC fitting to one concentration-response curve.
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

from _load_data_mers import load_data_one_inhibitor

from _define_model import Model
from _model_fitting import _run_mcmc
from _CRC_fitting import _expt_check_noise_trend

from _MAP_mpro import _map_running
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior
from _plotting import plot_data_conc_log

from _save_setting import save_model_setting

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="")
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

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

parser.add_argument( "--outlier_removal",               action="store_true",    default=False)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

df_mers = pd.read_csv(args.input_file)

inhibitor_name = np.array([args.name_inhibitor])
for i, name in enumerate(inhibitor_name):
    expts_init, expts_plot = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==name)*(df_mers['Drop']!=1.0)],
                                                     multi_var=args.multi_var)

if len(expts_plot)>0:

    ## Outlier detection and trend checking
    [expts_outliers, outliers, _, _] = _expt_check_noise_trend(expts_init)
    if args.outlier_removal:
        print("Checking outlier(s) in the curve...")
        expts = expts_outliers.copy()
    else:
        expts = expts_init.copy()
        outliers = None

    ## Create a model to run
    model = Model(len(expts))
    model.check_model(args)

    ## Fitting model
    trace = _run_mcmc(expts, model.prior_infor, model.shared_params, model.init_values, model.args)

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

    n = 0
    plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, model.shared_params),
                       extract_kcat_n_idx(params_kcat, n, model.shared_params),
                       alpha_list=alpha_list, E_list=E_list, outliers=outliers,
                       OUTFILE=os.path.join(args.out_dir,'EI'))
else:
    print("There is no data found.")
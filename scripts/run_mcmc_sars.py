"""
This code is designed to fit the model that involves enzyme - substrate - inhibitors. 
Data was extracted from the publication of Nash et al. (2022) and Vuong et al (2022). 
"""

import warnings
import os
import argparse

import pickle
import matplotlib.pyplot as plt

import jax
import numpyro

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _load_data_sars import load_data_mut_wt

from _define_model import Model
from _model_fitting import _run_mcmc

from _MAP_mpro import _map_running
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, _trace_one_to_nchain
from _plotting import plot_data_conc_log

from _save_setting import save_model_setting

parser = argparse.ArgumentParser()

parser.add_argument( "--prior_infor",                   type=str,               default="")
parser.add_argument( "--shared_params_infor",           type=str,               default="")
parser.add_argument( "--initial_values",                type=str,               default="")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--fit_mutant_kinetics",           action="store_true",    default=False)
parser.add_argument( "--fit_mutant_AUC",                action="store_true",    default=False)
parser.add_argument( "--fit_mutant_ICE",                action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Nashed",           action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Vuong",            action="store_true",    default=False)

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var_mut",                 action="store_true",    default=False)
parser.add_argument( "--multi_var_wt",                  action="store_true",    default=False)

parser.add_argument( "--set_K_I_M_equal_K_S_M",         action="store_true",    default=False)
parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSS_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DSS",   action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=1)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

## Loading data from Nashed paper
expts, expts_mut, expts_wt, expts_wt_Vuong = load_data_mut_wt(args.fit_mutant_kinetics, args.fit_mutant_AUC, args.fit_mutant_ICE,
                                                              args.fit_wildtype_Nashed, args.fit_wildtype_Vuong,
                                                              args.fit_E_S, args.fit_E_I, args.multi_var_mut, args.multi_var_wt)
## Create a model to run
model = Model(len(expts))
model.check_model(args)

## Fitting model
trace = _run_mcmc(expts, model.prior_infor, model.shared_params, model.init_values, model.args)

## Finding MAP
[trace_map, map_index] = _map_running(trace.copy(), expts, model.prior_infor, model.shared_params, model.args)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace_map, map_index, model.prior_infor)

if not os.path.isdir('Fitting'):
    os.mkdir('Fitting')

n = 0
plot_data_conc_log(expts_mut, extract_logK_n_idx(params_logK, n),
                   extract_kcat_n_idx(params_kcat, n),
                   OUTFILE=os.path.join(args.out_dir,'Fitting','mut'))

n = 1
plot_data_conc_log(expts_wt, extract_logK_n_idx(params_logK, n),
                   extract_kcat_n_idx(params_kcat, n),
                   OUTFILE=os.path.join(args.out_dir,'Fitting','wt'))

## Saving the model fitting condition
save_model_setting(model.args, OUTDIR=args.out_dir, OUTFILE='setting.pickle')
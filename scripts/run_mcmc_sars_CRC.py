"""
This code is designed to fit the model that involves enzyme - substrate - inhibitors. 
Data was extracted from the publication of Nash et al. (2022) and Vuong et al (2022)
depending on the input argument --fit_mutant or --fit_wildtype. Additional CRC with 
the ID as --name_inhibitor from the --input_file can be included when running mcmc.
"""

import warnings
import numpy as np
import os
import argparse

import pandas as pd
import pickle
import matplotlib.pyplot as plt

import jax
import numpyro

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _load_data_sars import load_data_mut_wt
from _load_data import load_data_one_inhibitor

from _define_model import Model
from _model_fitting import _run_mcmc

from _MAP_mpro import _map_running
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import TraceExtraction
from _plotting import plot_data_conc_log

from _save_setting import save_model_setting

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="Inhibitor")

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
parser.add_argument( "--multi_var_CRC",                 action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)

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
expts_combine, expts_mut, expts_wt, expts_wt_Vuong = load_data_mut_wt(args.fit_mutant_kinetics, args.fit_mutant_AUC, args.fit_mutant_ICE,
                                                                      args.fit_wildtype_Nashed, args.fit_wildtype_Vuong,
                                                                      args.fit_E_S, args.fit_E_I, args.multi_var_mut, args.multi_var_wt)

# Load CRC
df_sars = pd.read_csv(args.input_file)

inhibitor_name = np.array([args.name_inhibitor])
for i, name in enumerate(inhibitor_name):
    expts_CRC, expts_plot_CRC = load_data_one_inhibitor(df_sars[(df_sars['Inhibitor_ID']==name)*(df_sars['Drop']!=1.0)])

expts = expts_combine + expts_CRC

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
params_logK, params_kcat = TraceExtraction(trace=trace_map).extract_params_from_map_and_prior(map_index, model.prior_infor)

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

alpha_list = {key: trace_map[key][map_index] for key in trace_map.keys() if key.startswith('alpha')}
n = 2
plot_data_conc_log(expts_plot_CRC, extract_logK_n_idx(params_logK, n, shared_params=None),
                   extract_kcat_n_idx(params_kcat, n, shared_params=None),
                   alpha_list=alpha_list, OUTFILE=os.path.join(args.out_dir,'Fitting','CRC'))

## Saving the model fitting condition
save_model_setting(model.args, OUTDIR=args.out_dir, OUTFILE='setting.pickle')
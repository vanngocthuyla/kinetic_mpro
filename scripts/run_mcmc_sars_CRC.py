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

inhibitor_name = 'CVD-0020650'

Itot = np.array([99.5, 99.5, 49.75, 49.75, 24.8799991607666, 24.8799991607666, 12.4399995803833, 12.4399995803833, 7.4629998207092285, 7.4629998207092285, 2.48799991607666, 2.48799991607666, 1.559000015258789, 1.559000015258789, 0.7789999842643738, 0.7789999842643738, 0.38999998569488525, 0.38999998569488525, 0.19499999284744263, 0.19499999284744263, 0.10000000149011612, 0.10000000149011612, 0.05000000074505806, 0.05000000074505806])*1E-6
logItot = np.log(Itot)
logMtot = np.log(40*1E-9)*np.ones(len(logItot))
logStot = np.log(3*1E-6)*np.ones(len(logItot))
inhibition = 100-np.array([100, 99.1, 99.1, 99.0, 99.5, 98.9, 98.5, 98.5, 97.3, 96.9, 76.6, 77.7, 61.2, 60.4, 37.4, 36.1, 28.0, 25.3, 12.4, 11.7, 13.1, 9.04, 2.70, 1.44])
v = inhibition/100*1E-9 #Convert to the same scale of velocity

expts_CRC = []
expts_plot_CRC = []
data_rate_CRC = {}
expts_plot_CRC.append({'type':'CRC', 'enzyme': 'wild_type', 'plate': '20650',
                       'index': '20650', 'figure': 'CVD-0020650', 'sub_figure': '0',
                       'logMtot': logMtot, # M
                       'logStot': logStot, # M
                       'logItot': logItot, # M
                       'v': v, 'x':'logItot'})
data_rate_CRC = [v, logMtot, logStot, logItot]
expts_CRC.append({'enzyme': 'wild_type', 'figure': inhibitor_name, 'index': '20650',
                  'plate' : '20650', 'kinetics': None, 'AUC': None, 'ICE': None,
                  'CRC': data_rate_CRC})

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

alpha_list = {key: trace_map[key][map_index] for key in trace_map.keys() if key.startswith('alpha')}
n = 2
plot_data_conc_log(expts_plot_CRC, extract_logK_n_idx(params_logK, n, shared_params=None),
                   extract_kcat_n_idx(params_kcat, n, shared_params=None),
                   alpha_list=alpha_list, OUTFILE=os.path.join(args.out_dir,'Fitting','CRC'))

## Saving the model fitting condition
save_model_setting(model.args, OUTDIR=args.out_dir, OUTFILE='setting.pickle')
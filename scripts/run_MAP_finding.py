import warnings
import numpy as np
import sys
import os
import argparse

import pickle
import arviz as az
import pandas as pd

import jax
import jax.numpy as jnp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _load_data_mers import load_data_no_inhibitor, load_data_one_inhibitor
from _define_model import Model
from _MAP_mpro import _map_running

from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, extract_params_from_trace_and_prior
from _plotting import plot_data_conc_log

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="")
parser.add_argument( "--prior_infor",                   type=str,               default="")
parser.add_argument( "--shared_params_infor",           type=str,               default="")
parser.add_argument( "--initial_values",                type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--mcmc_file",                     type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=0)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--nsamples_MAP",                  type=str,               default=None)

args = parser.parse_args()

# Loading experimental information
df_mers = pd.read_csv(args.input_file)
if len(args.name_inhibitor)>0:
    inhibitor_name = np.array([args.name_inhibitor.split()+'-001'])
else:
    inhibitor_name = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

expts_no_I, expts_plot_no_I = load_data_no_inhibitor(df_mers[df_mers['Inhibitor (nM)']==0.0], 
                                                     multi_var=args.multi_var)
no_expt = [len(expts_plot_no_I)]
expts = expts_no_I
expts_plot = expts_plot_no_I
for i, name in enumerate(inhibitor_name):
    expts_, expts_plot_ = load_data_one_inhibitor(df_mers[df_mers['Inhibitor_ID']==name],
                                                  multi_var=args.multi_var)
    expts = expts + expts_
    expts_plot = expts_plot + expts_plot_
    no_expt.append(len(expts_plot_))

## Loading the model
model = Model(len(expts))
model.check_model(args)

## Loading trace
trace = pickle.load(open(args.mcmc_file, "rb"))

## Finding MAP
[trace_map, map_index] = _map_running(trace.copy(), expts, model.prior_infor, model.shared_params, model.args)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace_map, map_index, model.prior_infor)

alpha_list = model.args.alpha_list
E_list = model.args.E_list

n = 0
plot_data_conc_log(expts_plot_no_I, extract_logK_n_idx(params_logK, n, model.shared_params),
                   extract_kcat_n_idx(params_kcat, n, model.shared_params),
                   line_colors=['black', 'red', 'tab:brown'], ls='-.',
                   E_list=E_list, plot_legend=True, combined_plots=True,
                   OUTFILE=os.path.join(args.out_dir,'ES'))

for i in range(len(inhibitor_name)):
    n = i + 1
    start = i*no_expt[n]+no_expt[0]
    end   = i*no_expt[n]+no_expt[0]+no_expt[n]    
    plot_data_conc_log(expts_plot[start:end], extract_logK_n_idx(params_logK, n, shared_params),
                       extract_kcat_n_idx(params_kcat, n, shared_params),
                       alpha_list=alpha_list, E_list=E_list, plot_legend=False,
                       OUTFILE=os.path.join(args.out_dir, 'ESI_'+str(i)))
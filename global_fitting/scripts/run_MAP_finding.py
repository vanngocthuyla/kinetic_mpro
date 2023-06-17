import warnings
import numpy as np
import sys
import os
import argparse

import pickle
import arviz as az

import jax
import jax.numpy as jnp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _MAP_finding import map_finding
from _trace_analysis import extract_params_from_map_and_prior
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _load_data import load_data_mut_wt

parser = argparse.ArgumentParser()

parser.add_argument( "--fit_mutant_kinetics",   action="store_true",    default=False)
parser.add_argument( "--fit_mutant_AUC",        action="store_true",    default=False)
parser.add_argument( "--fit_mutant_ICE",        action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Nashed",   action="store_true",    default=True)
parser.add_argument( "--fit_wildtype_Vuong",    action="store_true",    default=False)
parser.add_argument( "--fit_E_S",               action="store_true",    default=True)
parser.add_argument( "--fit_E_I",               action="store_true",    default=False)
parser.add_argument( "--multi_var_mut",         action="store_true",    default=False)
parser.add_argument( "--multi_var_wt",          action="store_true",    default=False)

parser.add_argument( "--mcmc_file",             type=str, 				default="")
parser.add_argument( "--prior_infor",           type=str,               default="")
parser.add_argument( "--nsamples",              type=str, 				default=None)
parser.add_argument( "--out_dir",               type=str,               default="")

args = parser.parse_args()


expts, expts_mut, expts_wt, expts_wt_2 = load_data_mut_wt(args.fit_mutant_kinetics, args.fit_mutant_AUC, args.fit_mutant_ICE,
                                                          args.fit_wildtype_Nashed, args.fit_wildtype_Vuong,
                                                          args.fit_E_S, args.fit_E_I, args.multi_var_mut, args.multi_var_wt)

trace = pickle.load(open(args.mcmc_file, "rb"))
df = pd.read_csv(args.prior_infor)
prior_infor_update = []
for index, row in df.iterrows():
    prior_infor_update.append(row.to_dict())

# Finding MAP
[map_index, map_params, log_probs] = map_finding(trace, expts, prior_infor_update)

with open("map.txt", "w") as f:
  	print("MAP index:" + str(map_index), file=f)
  	print("Kinetics parameters: \n" + str(map_params), file=f)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

n = 0
for expt_plot in [expts_mut, expts_wt, expts_wt_2]:
    if len(expt_plot)>0:
        adjustable_plot_data(expt_plot, extract_logK_n_idx(params_logK, n), extract_kcat_n_idx(params_kcat, n),
                             OUTDIR=args.out_dir)
        n = n + 1
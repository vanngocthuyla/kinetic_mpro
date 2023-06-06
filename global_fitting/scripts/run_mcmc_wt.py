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
from jax import random
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _bayesian_model_WT import global_fitting_WT
from _load_data import load_data_mut_wt
from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx_WT, extract_kcat_n_idx_WT
from _trace_analysis import extract_params_from_trace_and_prior
from _plotting import plot_kinetics_data_WT

parser = argparse.ArgumentParser()

parser.add_argument( "--out_dir",               type=str, 				default="")
parser.add_argument( "--fit_wildtype_Nashed",   action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Vuong",    action="store_true",    default=False)
parser.add_argument( "--fit_E_S",               action="store_true",    default=False)
parser.add_argument( "--fit_E_I",               action="store_true",    default=False)
parser.add_argument( "--multi_var_wt",          action="store_true",    default=False)

parser.add_argument( "--niters",				type=int, 				default=10000)
parser.add_argument( "--nburn",                 type=int, 				default=2000)
parser.add_argument( "--nthin",                 type=int, 				default=1)
parser.add_argument( "--nchain",                type=int, 				default=4)
parser.add_argument( "--random_key",            type=int, 				default=0)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

expts, expts_mut, expts_wt, expts_wt_2 = load_data_mut_wt(False, False, False, 
                                                          args.fit_wildtype_Nashed, args.fit_wildtype_Vuong, 
                                                          args.fit_E_S, args.fit_E_I, False, args.multi_var_wt)

logKd_min = -20.
logKd_max = 0.
kcat_min = 0. 
kcat_max = 100.

prior = {}
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}

prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv("Prior_infor.csv", index=False)

print("Prior information: \n", pd.DataFrame(prior_infor_update))

if args.fit_E_S and args.fit_E_I and args.fit_wildtype_Nashed and args.fit_wildtype_Vuong:
    traces_name = "traces_both"
elif args.fit_E_S and args.fit_E_I and args.fit_wildtype_Vuong: 
    traces_name = "traces_Vuong"
elif args.fit_E_S and args.fit_wildtype_Nashed: 
    traces_name = "traces_Nashed"
else:
    traces_name = "traces"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

kernel = NUTS(global_fitting_WT)
mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
mcmc.run(rng_key_, expts, prior_infor_update)
mcmc.print_summary()

trace = mcmc.get_samples(group_by_chain=False)
pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

trace = mcmc.get_samples(group_by_chain=True)
az.summary(trace).to_csv(traces_name+"_summary.csv")

## Trace plot
data = az.convert_to_inference_data(trace)
az.plot_trace(data, compact=False)
plt.tight_layout();
plt.savefig(os.path.join(args.out_dir, 'Plot_trace'))
plt.ioff()

## Autocorrelation plot
az.plot_autocorr(trace);
plt.savefig(os.path.join(args.out_dir, 'Plot_autocorr'))
plt.ioff()

## Fitting plot
params_logK, params_kcat = extract_params_from_trace_and_prior(trace, prior_infor_update)

n = 0
for expt_plot in [expts_wt, expts_wt_2]:
    if len(expt_plot)>0:
        plot_kinetics_data_WT(expt_plot, extract_logK_n_idx_WT(params_logK, n), extract_kcat_n_idx_WT(params_kcat, n),
                              OUTDIR=args.out_dir)
        n = n + 1
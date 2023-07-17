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
from numpyro.infer import MCMC, NUTS

import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _bayesian_model_adjustable import adjustable_global_fitting
from _load_data import load_data_mut_wt
from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _MAP_finding import map_finding
from _trace_analysis import extract_params_from_map_and_prior
from _plotting import adjustable_plot_data

parser = argparse.ArgumentParser()

parser.add_argument( "--out_dir",                       type=str, 				default="")

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
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSS_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DSS",   action="store_true",    default=False)

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

expts, expts_mut, expts_wt, expts_wt_2 = load_data_mut_wt(args.fit_mutant_kinetics, args.fit_mutant_AUC, args.fit_mutant_ICE, 
                                                          args.fit_wildtype_Nashed, args.fit_wildtype_Vuong, 
                                                          args.fit_E_S, args.fit_E_I, args.multi_var_mut, args.multi_var_wt)

logKd_min = -20.
logKd_max = 0.
kcat_min = 0. 

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'local','dist': ['uniform', 'normal', None], 'loc': [None, -14, None], 'scale': [None, 2, None], 'value': None, 'lower': [-7, None, None], 'upper': [0, None, None]}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'global', 'dist': 'normal', 'loc': -15, 'scale': 2}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'global', 'dist': 'normal', 'loc': -15, 'scale': 2}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': [1., 200, 300]}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': [1., 200., 300.]}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': [1., 200., 300.]}

if args.set_K_I_M_equal_K_S_M: 
    del prior['logK_I_M']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']
if args.set_kcat_DSS_equal_kcat_DS:
    del prior['kcat_DSS']
if args.set_kcat_DSI_equal_kcat_DS or args.set_kcat_DSI_equal_kcat_DSS:
    del prior['kcat_DSI']

shared_params = {}
shared_params['logKd'] = {'assigned_idx': 2, 'shared_idx': 1}
# shared_params['logK_S_M'] = {'assigned_idx': 1, 'shared_idx': 0}
# shared_params['logK_S_D'] = {'assigned_idx': 1, 'shared_idx': 0}
# shared_params['logK_S_DS'] = {'assigned_idx': 1, 'shared_idx': 0}
# shared_params['logK_S_DI'] = {'assigned_idx': 1, 'shared_idx': 0}

prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv("Prior_infor.csv", index=False)

print("Prior information: \n", pd.DataFrame(prior_infor_update))

if args.fit_E_S and args.fit_E_I: traces_name = "traces"
elif args.fit_E_S: traces_name = "traces_E_S"
elif args.fit_E_I: traces_name = "traces_E_I"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

kernel = NUTS(adjustable_global_fitting)
mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
mcmc.run(rng_key_, experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params, 
         set_K_I_M_equal_K_S_M=args.set_K_I_M_equal_K_S_M, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS, 
         set_kcat_DSS_equal_kcat_DS=args.set_kcat_DSS_equal_kcat_DS, set_kcat_DSI_equal_kcat_DS=args.set_kcat_DSI_equal_kcat_DS,
         set_kcat_DSI_equal_kcat_DSS=args.set_kcat_DSI_equal_kcat_DSS)
mcmc.print_summary()

trace = mcmc.get_samples(group_by_chain=False)
pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

## Autocorrelation plot
az.plot_autocorr(trace);
plt.savefig(os.path.join(args.out_dir, 'Plot_autocorr'))
plt.ioff()

trace = mcmc.get_samples(group_by_chain=True)
az.summary(trace).to_csv(traces_name+"_summary.csv")

## Trace plot
if len(trace.keys())>=15:
    for param_name in ['logK', 'kcat', 'log_sigma']:
        trace_2 = {}
        for key in trace.keys():
            if key.startswith(param_name):
                trace_2[key] = trace[key]
        ## Trace plot
        data = az.convert_to_inference_data(trace_2)
        az.plot_trace(data, compact=False)
        plt.tight_layout();
        plt.savefig(os.path.join(args.out_dir, 'Plot_trace_'+param_name))
        plt.ioff()
else:
    data = az.convert_to_inference_data(trace)
    az.plot_trace(data, compact=False)
    plt.tight_layout();
    plt.savefig(os.path.join(args.out_dir, 'Plot_trace'))
    plt.ioff()

# Finding MAP
trace = mcmc.get_samples(group_by_chain=False)
if shared_params is not None and len(shared_params)>0:
    for name in shared_params.keys():
        param = shared_params[name]
        assigned_idx = param['assigned_idx']
        shared_idx = param['shared_idx']
        trace[f'{name}:{assigned_idx}'] = trace[f'{name}:{shared_idx}']

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'local','dist': ['uniform', 'normal', 'normal'], 'loc': [None, -14, -14], 'scale': [None, 2, 2], 'lower': [-7, None, None], 'upper': [0, None, None]}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'global', 'dist': 'normal', 'loc': -15, 'scale': 2}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'global', 'dist': 'normal', 'loc': -15, 'scale': 2}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': [1., 200, 300]}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': [1., 200., 300.]}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': [1., 200., 300.]}

if args.set_K_I_M_equal_K_S_M: 
    del prior['logK_I_M']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']
if args.set_kcat_DSS_equal_kcat_DS:
    del prior['kcat_DSS']
if args.set_kcat_DSI_equal_kcat_DS or args.set_kcat_DSI_equal_kcat_DSS:
    del prior['kcat_DSI']

prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv("Prior_infor_2.csv", index=False)

df = pd.read_csv("Prior_infor_2.csv")
prior_infor_update = []
for index, row in df.iterrows():
    prior_infor_update.append(row.to_dict())

[map_index, map_params, log_probs] = map_finding(trace, expts, prior_infor_update, 
                                                 args.set_K_I_M_equal_K_S_M, args.set_K_S_DI_equal_K_S_DS, 
                                                 args.set_kcat_DSS_equal_kcat_DS, args.set_kcat_DSI_equal_kcat_DS,
                                                 args.set_kcat_DSI_equal_kcat_DSS)

with open("map.txt", "w") as f:
    print("MAP index: " + str(map_index), file=f)
    print("\nKinetics parameters:", file=f)
    for key in trace.keys():
        print(key, ': %.3f' %trace[key][map_index], file=f)

pickle.dump(log_probs, open('log_probs.pickle', "wb"))

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

if args.set_K_I_M_equal_K_S_M:
    params_logK['logK_I_M'] = params_logK['logK_S_M']
if args.set_K_S_DI_equal_K_S_DS:
    params_logK['logK_S_DI'] = params_logK['logK_S_DS']
if args.set_kcat_DSS_equal_kcat_DS: 
    params_kcat['kcat_DSS'] = params_kcat['kcat_DS']
if args.set_kcat_DSI_equal_kcat_DS: 
    params_kcat['kcat_DSI'] = params_kcat['kcat_DS']
elif args.set_kcat_DSI_equal_kcat_DSS:
    params_kcat['kcat_DSI'] = params_kcat['kcat_DSS']

n = 0
for expt_plot in [expts_mut, expts_wt, expts_wt_2]:
    if len(expt_plot)>0:
        adjustable_plot_data(expt_plot, extract_logK_n_idx(params_logK, n), extract_kcat_n_idx(params_kcat, n),
                             OUTDIR=args.out_dir)
        n = n + 1
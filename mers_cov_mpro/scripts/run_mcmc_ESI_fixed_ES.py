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
from numpyro.infer import MCMC, NUTS, init_to_value

import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _model_mers_ESI_fixed_ES import global_fitting
from _load_data_mers import load_data_no_inhibitor, load_data_one_inhibitor
from _plotting import plot_data_conc_log, plotting_trace
from _MAP_finding_mers_concs import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="Inhibitor")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--map_file",                      type=str,               default="")

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

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

df_mers = pd.read_csv(args.input_file)

inhibitor_name = np.array([args.name_inhibitor+'-001'])
for i, name in enumerate(inhibitor_name):
    expts, expts_plot = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==name)*(df_mers['Drop']!=1)],
                                                multi_var=args.multi_var)

if len(args.map_file)>0 and os.path.isfile(args.map_file):
    map_sampling = pickle.load(open(args.map_file, "rb"))
    
    if args.set_K_S_DS_equal_K_S_D: 
        map_sampling['logK_S_DS'] = map_sampling['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        map_sampling['logK_S_DI'] = map_sampling['logK_S_DS']

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'alpha:ESI:0', 'alpha:ESI:1', 'alpha:ESI:2', 'alpha:ESI:3',
                'kcat_DS', 'kcat_DSS', 'dE:100', 'dE:50', 'dE:25']:
        assert key in map_sampling.keys(), f"Please provide {key} in map_file."

logKd_min = -27.63
logKd_max = 0.
kcat_min = 0.
kcat_max = 20

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': None, 'value': map_sampling['logKd']}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': None, 'value': map_sampling['logK_S_M']} 
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': None, 'value': map_sampling['logK_S_D']} 
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': None, 'value': map_sampling['logK_S_DS']}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'global', 'dist': 'uniform', 'lower': -20.73, 'upper': logKd_max} #1E-9
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': -6.9}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': -6.9}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': -6.9}

prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': None, 'value': map_sampling['kcat_DS']}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': None, 'value': map_sampling['kcat_DSS']}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}

shared_params = None

if args.set_K_S_DS_equal_K_S_D: 
    del prior['logK_S_DS']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']

pickle.dump(prior, open(os.path.join('Prior.pickle'), "wb"))
prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv("Prior_infor.csv", index=False)

print("Prior information: \n", pd.DataFrame(prior_infor_update))

no_expt = 4 #len(expts[0]['kinetics'])
alphas = [map_sampling[f'alpha:ESI:{i}'] for i in range(no_expt)]

E_list = {}
for key in ['dE:100', 'dE:50', 'dE:25']:
    E_list[key] = map_sampling[key]

if args.fit_E_S and args.fit_E_I: traces_name = "traces"
elif args.fit_E_S: traces_name = "traces_E_S"
elif args.fit_E_I: traces_name = "traces_E_I"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

if os.path.isfile(traces_name+'.pickle'):
    trace = pickle.load(open(traces_name+'.pickle', "rb"))
else:
    if len(args.map_file)>0 and os.path.isfile(args.map_file):
        init_values = pickle.load(open(args.map_file, "rb"))
        print("Initial values:", init_values)
        kernel = NUTS(model=global_fitting, init_strategy=init_to_value(values=init_values))
    else:
        kernel = NUTS(global_fitting)

    if os.path.isfile(os.path.join(args.last_run_dir, "Last_state.pickle")):
        last_state = pickle.load(open(os.path.join(args.last_run_dir, "Last_state.pickle"), "rb"))
        print("\nKeep running from last state.")
        mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
        mcmc.post_warmup_state = last_state
        mcmc.run(mcmc.post_warmup_state.rng_key,  
                 experiments=expts, alphas=alphas, E_list=E_list, 
                 prior_infor=prior_infor_update, shared_params=shared_params,
                 multi_var=args.multi_var,
                 set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
    else:
        mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
        mcmc.run(rng_key_, experiments=expts, alphas=alphas, E_list=E_list, 
                 prior_infor=prior_infor_update, shared_params=shared_params,
                 multi_var=args.multi_var,
                 set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
    
    print("Saving last state.")
    mcmc.post_warmup_state = mcmc.last_state
    pickle.dump(jax.device_get(mcmc.post_warmup_state), open("Last_state.pickle", "wb"))

    mcmc.print_summary()

    trace = mcmc.get_samples(group_by_chain=True)
    az.summary(trace).to_csv(traces_name+"_summary.csv")

    trace = mcmc.get_samples(group_by_chain=False)
    pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

    plotting_trace(trace, args.out_dir, nchain=args.nchain, nsample=args.niters)

if shared_params is not None and len(shared_params)>0:
    for name in shared_params.keys():
        param = shared_params[name]
        assigned_idx = param['assigned_idx']
        shared_idx = param['shared_idx']
        trace[f'{name}:{assigned_idx}'] = trace[f'{name}:{shared_idx}']

alpha_list = {}
for i in range(no_expt):
    alpha_list[f'alpha:ESI:{i}'] = alphas[i]

trace_map = trace.copy()
[map_index, map_params, log_probs] = map_finding(trace_map, expts, prior_infor=prior_infor_update, alpha_list=alpha_list, E_list=E_list, 
                                                 set_lognormal_dE=args.set_lognormal_dE, dE=args.dE,
                                                 set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D,
                                                 set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)

with open("map.txt", "w") as f:
    print("MAP index:" + str(map_index), file=f)
    print("\nKinetics parameters:", file=f)
    for key in trace.keys():
        print(key, ': %.3f' %trace[key][map_index], file=f)

pickle.dump(log_probs, open('log_probs.pickle', "wb"))

map_values = {}
for key in trace.keys():
    map_values[key] = trace[key][map_index]
pickle.dump(map_values, open('map.pickle', "wb"))

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

n = 0
plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                   extract_kcat_n_idx(params_kcat, n, shared_params),
                   alpha=alphas, error_E=E_list,
                   OUTFILE=os.path.join(args.out_dir,'ESI'))
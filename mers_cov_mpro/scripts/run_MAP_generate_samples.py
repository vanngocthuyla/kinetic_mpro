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

from _bayesian_model_mers import adjustable_global_fitting
from _load_data_mers import load_data_no_inhibitor
from _plotting import adjustable_plot_data_mers
from _MAP_finding_mers import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior
from _trace_analysis import extract_params_from_trace_and_prior

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)

parser.add_argument( "--set_K_I_M_equal_K_S_M",         action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSS_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DSS",   action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

parser.add_argument( "--map_file",                      type=str,               default="")

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

df_mers = pd.read_csv(args.input_file)
expts, expts_plot = load_data_no_inhibitor(df_mers, args.multi_var)

# Loading prior information
assert os.path.isfile(args.prior_infor), "Please provide the prior information."
prior = pickle.load(open(args.prior_infor, "rb"))
prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv(os.path.join(args.out_dir, "Prior_infor.csv"), index=False)

print("Prior information: \n", pd.DataFrame(prior_infor_update))

if args.fit_E_S and args.fit_E_I: traces_name = "traces"
elif args.fit_E_S: traces_name = "traces_E_S"
elif args.fit_E_I: traces_name = "traces_E_I"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

init_values = pickle.load(open(args.map_file, "rb"))
print("Initial values:", init_values)

kernel = NUTS(model=adjustable_global_fitting, init_strategy=init_to_value(values=init_values))
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
if os.path.isfile(traces_name+'.pickle'):
    trace = pickle.load(open(traces_name+'.pickle', "rb"))
else:
    trace = mcmc.get_samples(group_by_chain=False)

if shared_params is not None and len(shared_params)>0:
    for name in shared_params.keys():
        param = shared_params[name]
        assigned_idx = param['assigned_idx']
        shared_idx = param['shared_idx']
        trace[f'{name}:{assigned_idx}'] = trace[f'{name}:{shared_idx}']

[map_index, map_params, log_probs] = map_finding(trace, expts, prior_infor_update, 
                                                 args.set_K_I_M_equal_K_S_M, args.set_K_S_DI_equal_K_S_DS, 
                                                 args.set_kcat_DSS_equal_kcat_DS, args.set_kcat_DSI_equal_kcat_DS,
                                                 args.set_kcat_DSI_equal_kcat_DSS)

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

# ## Fitting plot
# params_logK, params_kcat = extract_params_from_trace_and_prior(trace, prior_infor_update)

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

for n in range(len(expts)):
    if args.set_K_I_M_equal_K_S_M:
        try: params_logK[f'logK_I_M:{n}'] = params_logK[f'logK_S_M:{n}']
        except: params_logK['logK_I_M'] = params_logK['logK_S_M']
    if args.set_K_S_DI_equal_K_S_DS:
        try: params_logK[f'logK_S_DI:{n}'] = params_logK[f'logK_S_DS:{n}']
        except: params_logK['logK_S_DI'] = params_logK['logK_S_DS']
    if args.set_kcat_DSS_equal_kcat_DS: 
        try: params_kcat[f'kcat_DSS:{n}'] = params_kcat[f'kcat_DS:{n}']
        except: params_kcat['kcat_DSS'] = params_kcat['kcat_DS']
    if args.set_kcat_DSI_equal_kcat_DS: 
        try: params_kcat[f'kcat_DSI:{n}'] = params_kcat[f'kcat_DS:{n}']
        except: params_kcat['kcat_DSI'] = params_kcat['kcat_DS']
    elif args.set_kcat_DSI_equal_kcat_DSS:
        try: params_kcat[f'kcat_DSI:{n}'] = params_kcat[f'kcat_DSS:{n}']
        except: params_kcat['kcat_DSI'] = params_kcat['kcat_DSS']

n = 0
adjustable_plot_data_mers(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                          extract_kcat_n_idx(params_kcat, n, shared_params),
                          OUTDIR=args.out_dir)
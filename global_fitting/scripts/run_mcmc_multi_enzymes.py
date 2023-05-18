import warnings
import numpy as np
import sys
import os
import argparse

import pickle
import arviz as az

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

from _bayesian_model_multi_enzymes import global_fitting_multi_enzyme, check_prior_group
from _load_data_multi_enzyme import load_data

parser = argparse.ArgumentParser()

# parser.add_argument( "--data_dir",              type=str,                 default="")
# parser.add_argument( "--data_file",             type=str,                 default="")
parser.add_argument( "--out_dir",               type=str,               default="")

parser.add_argument( "--fit_mutant_kinetics",   action="store_true",    default=False)
parser.add_argument( "--fit_mutant_AUC",        action="store_true",    default=False)
parser.add_argument( "--fit_mutant_ICE",        action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Nashed",   action="store_true",    default=True)
parser.add_argument( "--fit_wildtype_Vuong",    action="store_true",    default=True)
parser.add_argument( "--fit_E_S",               action="store_true",    default=True)
parser.add_argument( "--fit_E_I",               action="store_true",    default=False)

parser.add_argument( "--niters",                type=int,               default=10000)
parser.add_argument( "--nburn",                 type=int,               default=2000)
parser.add_argument( "--nthin",                 type=int,               default=1)
parser.add_argument( "--nchain",                type=int,               default=4)
parser.add_argument( "--random_key",            type=int,               default=0)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

experiments = load_data(args.fit_mutant_kinetics, args.fit_mutant_AUC, args.fit_mutant_ICE, 
                        args.fit_wildtype_Nashed, args.fit_wildtype_Vuong, 
                        args.fit_E_I, args.fit_E_I)

logKd_min = -20.
logKd_max = 0.
kcat_min = 0. 
kcat_max = 10.

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'local','dist': 'normal', 'loc': -13.5, 'scale': 3}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'global', 'dist': 'normal', 'loc': -13, 'scale': 3}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'global', 'dist': 'normal', 'loc': -15, 'scale': 3}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}

prior_infor = []
prior_infor.append(dict([(key, prior['logKd'][key]) for key in prior['logKd'].keys()]))

for name in ['logK_S_M', 'logK_S_D', 'logK_S_DS']:
    if args.fit_E_S:
        prior_infor.append(dict([(key, prior[name][key]) for key in prior[name].keys()]))
    else: 
        prior_infor.append({'type':'logK', 'name': name, 'fit': prior[name]['fit'], 'dist': None, 'value': 0})

for name in ['logK_I_M', 'logK_I_D', 'logK_I_DI']:
    if args.fit_E_I:
        prior_infor.append(dict([(key, prior[name][key]) for key in prior[name].keys()]))
    else: 
        prior_infor.append({'type':'logK', 'name': name, 'fit': prior[name]['fit'], 'dist': None, 'value': 0})

if args.fit_E_S and args.fit_E_I:
    prior_infor.append(dict([(key, prior['logK_S_DI'][key]) for key in prior['logK_S_DI'].keys()]))
else:
    prior_infor.append({'type':'logK', 'name': 'logK_S_DI', 'fit': prior[name]['fit'], 'dist': None, 'value': 0})
    
for name in ['kcat_MS', 'kcat_DS', 'kcat_DSS']:
    if args.fit_E_S:
        prior_infor.append(dict([(key, prior[name][key]) for key in prior[name].keys()]))
    else: 
        prior_infor.append({'type':'kcat', 'name': name, 'fit': prior[name]['fit'], 'dist': None, 'value': 0})

if args.fit_E_S and args.fit_E_I:
    prior_infor.append(dict([(key, prior['kcat_DSI'][key]) for key in prior['kcat_DSI'].keys()]))
else:
    prior_infor.append({'type':'kcat', 'name': 'kcat_DSI', 'fit': prior[name]['fit'], 'dist': None, 'value': 0})

prior_infor_update = check_prior_group(prior_infor, len(experiments))
prior_infor_update

print("Prior information: \n", prior_infor_update)

if args.fit_E_S and args.fit_E_I:
    traces_name = "traces"
elif args.fit_E_S: 
    traces_name = "traces_E_S"
elif args.fit_E_I: 
    traces_name = "traces_E_I"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

kernel = NUTS(global_fitting_multi_enzyme)
mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, 
            progress_bar=True)
mcmc.run(rng_key_, experiments, prior_infor_update)
mcmc.print_summary()

trace = mcmc.get_samples(group_by_chain=False)
pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

sample = az.convert_to_inference_data(mcmc.get_samples(group_by_chain=True))
az.plot_trace(sample, compact=False)
plt.tight_layout();
plt.savefig(os.path.join(args.out_dir, traces_name+'.pdf'))
plt.ioff()
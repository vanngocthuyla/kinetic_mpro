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
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
from numpyro.infer import MCMC, NUTS

import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _bayesian_model import global_fitting_informative, extract_logK, extract_kcat
from _plotting import plot_kinetics_data

parser = argparse.ArgumentParser()

# parser.add_argument( "--data_dir",              type=str, 				default="")
# parser.add_argument( "--data_file",             type=str, 				default="")
parser.add_argument( "--out_dir",               type=str, 				default="")
parser.add_argument( "--kinetics_fitting",      action="store_true",    default=True)
parser.add_argument( "--auc_fitting",           action="store_true",    default=True)
parser.add_argument( "--ice_fitting",           action="store_true",    default=True)

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

experiments = []
# Fig S2a red WT ##Condition similar to 3a Mutation
experiments.append({'type':'kinetics',
                    'figure':'WT-2a',
                    'logMtot': np.log(np.array([0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0938, 0.0625, 0.0469])*1E-6), # M
                    'logStot': np.array([np.log(200E-6)]*9), # 5 uL of 5mM substrate diluted with 95 uL reaction mixture,
                    'logItot': np.array([np.log(1E-30)]*9), #None
                    'v': np.array([6.71, 3.17, 1.99, 0.89, 0.58, 0.41, 0.195, 0.147, 0.072])*1E-6, # M min^{-1}
                    'x':'logMtot'})

#Fig S2b WT ##Condition similar to 3b Mutation
experiments.append({'type':'kinetics',
                    'figure':'WT-2b',
                    'logMtot': np.array([np.log(200E-9)]*7), # M
                    'logStot': np.log(np.array([16, 32, 64, 77, 102.4, 128, 154])*1E-6), #M
                    'logItot': np.array([np.log(1E-30)]*7), # None
                    'v': np.array([0.285, 0.54, 0.942, 0.972, 1.098, 1.248, 1.338])*1E-6, # M min^{-1}
                    'x':'logStot'})

#Fig S2c WT ##Condition similar to 3b Mutation
experiments.append({'type':'kinetics',
                    'figure':'WT-2b',
                    'logMtot': np.log(np.array([2, 1, 0.5, 0.25, 0.125, 0.0625, 0.0313])*1E-6), # M
                    'logStot': np.array([np.log(200E-6)]*7), # M
                    'logItot': np.array([np.log(1E-30)]*7), # None
                    'v': np.array([30, 11.493, 4.8215, 1.4366, 0.46799, 0.10725, 0.021973])*1E-6, # M min^{-1}
                    #'v/Mtot': np.array([15, 11.4931, 9.643, 5.7465, 3.7439, 1.716, 0.702]), # min^{-1}
                    'x':'logMtot'})

# # Fig 1b WT SARS-Covid-2 #Vuong et al
# experiments.append({'type':'kinetics',
#                     'figure':'WT-1b',
#                     'logMtot': np.array([np.log(80E-9)]*24), # M
#                     'logStot': np.array([np.log(100E-6)]*24), # None
#                     'logItot': np.log(10**(np.array([-8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5, -8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5, -8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5]))), #M
#                     # 'logItot': np.array([-8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5, -8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5, -8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5]), #M
#                     'v': np.array([9.32 , 9.15 , 9.2 , 2.42 , 0.45 , 0.25 , 0.12 , 0.05, 8.03 , 8.29 , 6.26 , 0.51 , 0.02 , 0.045 , 0 , 0.1, 7.93 , 8.21 , 7.26 , 1.26 , 0.312 , 0.15 , 0.09 , 0.08])*1E-6, # M min^{-1}
#                     'x':'logItot'})

# Collate all kinetics experiments
kinetics_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'kinetics'])
kinetics_logStot = np.hstack([experiment['logStot'] for experiment in experiments if experiment['type'] == 'kinetics'])
kinetics_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'kinetics'])
v = np.hstack([experiment['v'] for experiment in experiments if experiment['type'] == 'kinetics'])

prior_information = []
logKd_min = -20
logKd_max = 0
kcat_min = 0 
kcat_max = 100

prior_information.append({'type':'logK', 'name': 'logKd', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
prior_information.append({'type':'logK', 'name': 'logK_S_M', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
prior_information.append({'type':'logK', 'name': 'logK_S_D', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
prior_information.append({'type':'logK', 'name': 'logK_S_DS', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max})
prior_information.append({'type':'logK', 'name': 'logK_I_M', 'dist': None, 'value': 0})
prior_information.append({'type':'logK', 'name': 'logK_I_D', 'dist': None, 'value': 0})
prior_information.append({'type':'logK', 'name': 'logK_I_DI', 'dist': None, 'value': 0})
prior_information.append({'type':'logK', 'name': 'logK_S_DI', 'dist': None, 'value': 0})

prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
prior_information.append({'type':'kcat', 'name': 'kcat_DS', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
prior_information.append({'type':'kcat', 'name': 'kcat_DSS', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max})
prior_information.append({'type':'kcat', 'name': 'kcat_DSI', 'dist': None, 'value': 0})

print("Prior information: \n", pd.DataFrame(prior_information))
pd.DataFrame(prior_information).to_csv("Prior_infor.csv", index=False)

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

kernel = NUTS(global_fitting_informative)
mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, 
            progress_bar=True)
mcmc.run(rng_key_,
         [v, kinetics_logMtot, kinetics_logStot, kinetics_logItot], None, None, 
         prior_information)
mcmc.print_summary()

trace = mcmc.get_samples(group_by_chain=False)
pickle.dump(trace, open(os.path.join(args.out_dir, 'traces.pickle'), "wb"))

trace = mcmc.get_samples(group_by_chain=True)
az.summary(trace).to_csv(traces_name+"_summary.csv")

## Trace plot
data = az.convert_to_inference_data(trace)
az.plot_trace(data, compact=False)
plt.tight_layout();
plt.savefig(os.path.join(args.out_dir,'Plot_trace'))
plt.ioff()

## Autocorrelation plot
az.plot_autocorr(trace);
plt.savefig(os.path.join(args.out_dir, 'Plot_autocorr'))
plt.ioff()

## Fitting plot
params_name_kcat = []
params_name_logK = []
for prior in prior_information:
    if prior['type'] == 'kcat': params_name_kcat.append(prior['name'])
    else: params_name_logK.append(prior['name'])

samples = extract_samples_from_trace(data, params_name_logK)
params_logK = {}
for name in params_name_logK:
    try: params_logK[name] = np.mean(samples[name])
    except: params_logK[name] = 0

samples = extract_samples_from_trace(data, params_name_kcat)
params_kcat = {}
for name in params_name_kcat:
    try: params_kcat[name] = np.mean(samples[name])
    except: params_kcat[name] = 0

plot_kinetics_data(experiments, extract_logK(params_logK), extract_kcat(params_kcat),
                   OURDIR=args.out_dir)
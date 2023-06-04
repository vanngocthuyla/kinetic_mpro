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

from _bayesian_model_informative import global_fitting_informative

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
# Fig 3a red
experiments.append({'type':'kinetics',
                    'figure':'3a',
                    'logMtot': np.log(np.array([6, 10, 25, 50, 75, 100, 120])*1E-6), # M
                    'logStot': np.array([np.log(200E-6)]*7), # 5 uL of 5mM substrate diluted with 95 uL reaction mixture,
                    'logItot': np.array([np.log(1E-20)]*7), # None
                    'v': np.array([0.0195, 0.024, 0.081, 0.36, 0.722, 1.13, 1.64])*1E-6, # M min^{-1}
                    'x':'logMtot'})

# Fig 5b 
experiments.append({'type':'kinetics',
                    'figure':'5b',
                    'logMtot': np.array([np.log(10E-6)]*12),  # M
                    'logStot': np.array([np.log(0.25E-3)]*12), #5 uL of 5mM substrate diluted with 95 uL reaction mixture,
                    'logItot': np.log(np.array([0.25, 0.5, 1, 1.75, 2.5, 5, 10, 16.6, 33.3, 66.6, 100, 133.3])*1E-6), 
                    'v': np.array([0.1204, 0.1862, 0.243, 0.3234, 0.3816, 0.4788, 0.5302, 0.3887, 0.2751, 0.1694, 0.129, 0.0947])*1E-6, # M min^{-1}
                    'x': 'logItot'})

# Fig 5c
experiments.append({'type':'kinetics',
                    'figure':'5c',
                    'logMtot': np.array([np.log(10E-6)]*6),
                    'logStot': np.log(np.array([96, 64, 48, 32, 16, 8])*1E-6),
                    'logItot': np.array([np.log(10E-6)]*6), 
                    'v': np.array([0.269, 0.189, 0.135, 0.097, 0.0395, 0.0229])*1E-6, # M min^{-1}
                    'x':'logStot'})
# Fig 6b
experiments.append({'type':'AUC',
                    'figure':'6b',
                    'logMtot': np.array([np.log(7E-6)]*8), # 6-7 micromolar
                    'logStot': np.array([np.log(1E-20)]*8), # None
                    'logItot': np.log(np.array([1, 3, 6, 10, 12.5, 15, 20, 50])*1E-6), 
                    'M': np.array([7, 6, 5, 3.25, 2.75, 2.55, 1.85, 0.8])*1E-6, # M min^{-1}
                    'x':'logItot'})
# Fig 6d
experiments.append({'type':'catalytic_efficiency',
                    'figure':'6d',
                    'logMtot':np.array([np.log(10E-6)]*5),
                    'logStot': np.array([np.log(1E-20)]*5), # Not used
                    'logItot':np.log((np.array([0, 23.3, 56.6, 90, 123]) + 10)*1E-6),
                    'Km_over_kcat':np.array([2170, 7640, 17800, 29700, 36600])/10.,
                    'x':'logItot'})


# Collate all kinetics experiments
kinetics_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'kinetics'])
kinetics_logStot = np.hstack([experiment['logStot'] for experiment in experiments if experiment['type'] == 'kinetics'])
kinetics_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'kinetics'])
v = np.hstack([experiment['v'] for experiment in experiments if experiment['type'] == 'kinetics'])

# Collate AUC experiments
AUC_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'AUC'])
AUC_logStot = np.hstack([experiment['logStot'] for experiment in experiments if experiment['type'] == 'AUC'])
AUC_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'AUC'])
auc = np.hstack([experiment['M'] for experiment in experiments if experiment['type'] == 'AUC'])

# Collate inverse catalytic efficiency experiments
ice_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'catalytic_efficiency'])
ice_logStot = None
ice_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'catalytic_efficiency'])
ice = np.hstack([experiment['Km_over_kcat'] for experiment in experiments if experiment['type'] == 'catalytic_efficiency'])


rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

kernel = NUTS(global_fitting_informative)
mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, 
            progress_bar=True, jit_model_args=True)
mcmc.run(rng_key_,
         [v, kinetics_logMtot, kinetics_logStot, kinetics_logItot], 
         [auc, AUC_logMtot, AUC_logStot, AUC_logItot], 
         [ice, ice_logMtot, ice_logStot, ice_logItot],
         kcat_max=10)
mcmc.print_summary()

trace = mcmc.get_samples(group_by_chain=False)
pickle.dump(trace, open(os.path.join(args.out_dir, 'traces_informative.pickle'), "wb"))

sample = az.convert_to_inference_data(mcmc.get_samples(group_by_chain=True))
az.plot_trace(sample)
plt.tight_layout();
plt.savefig(os.path.join(args.out_dir,'trace_informative.pdf'))
plt.ioff()

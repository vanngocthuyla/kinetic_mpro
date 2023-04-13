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

from _bayesian_fitting_3a import fit_mut_3a_alone

parser = argparse.ArgumentParser()

# parser.add_argument( "--data_dir",              type=str, 				default="")
# parser.add_argument( "--data_file",             type=str, 				default="")
parser.add_argument( "--out_dir",               type=str, 				default="")
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
# Fig 6b
experiments.append({'type':'kinetics',
                    'figure':'3a',
                    'logMtot': np.log(np.array([6, 10, 25, 50, 75, 100, 120])*1E-6), # M
                    'logStot': np.array([np.log(200E-6)]*7), # 5 uL of 5mM substrate diluted with 95 uL reaction mixture,
                    'logItot': np.array([np.log(1E-20)]*7), # None
                    'v': np.array([0.0195, 0.024, 0.081, 0.36, 0.722, 1.13, 1.64])*1E-6, # M min^{-1}
                    'x':'logMtot'})

kinetics_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'kinetics'])
kinetics_logStot = np.hstack([experiment['logStot'] for experiment in experiments if experiment['type'] == 'kinetics'])
kinetics_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'kinetics'])
v = np.hstack([experiment['v'] for experiment in experiments if experiment['type'] == 'kinetics'])


rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

params_logK = {'logKd': None, 'logK_S_M': None, 'logK_S_D': None, 'logK_S_DS': None,
               'logK_I_M': 0, 'logK_I_D': 0, 'logK_I_DI': 0., 'logK_S_DI': 0.}

kernel = NUTS(fit_mut_3a_alone)
mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, 
            progress_bar=True)
mcmc.run(rng_key_, [v, kinetics_logMtot, kinetics_logStot, kinetics_logItot], params_logK)
mcmc.print_summary()

trace = mcmc.get_samples(group_by_chain=False)
pickle.dump(trace, open(os.path.join(args.out_dir, 'traces_fig3a.pickle'), "wb"))

sample = az.convert_to_inference_data(mcmc.get_samples(group_by_chain=True))
az.plot_trace(sample)
plt.tight_layout();
plt.savefig(os.path.join(args.out_dir,'traces_fixed_fig3a.pdf'))
plt.ioff()
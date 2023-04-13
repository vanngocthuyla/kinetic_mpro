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

from _bayesian_model import global_fitting_jit, global_fitting

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
experiments.append({'type':'AUC',
                    'figure':'6b',
                    'logMtot': np.array([np.log(7E-6)]*8), # 6-7 micromolar
                    'logStot': np.array([np.log(1E-20)]*8), # None
                    'logItot': np.log(np.array([1, 3, 6, 10, 12.5, 15, 20, 50])*1E-6), 
                    'M': np.array([7, 6, 5, 3.25, 2.75, 2.55, 1.85, 0.8])*1E-6, # M min^{-1}
                    'x':'logItot'})

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

params_logK = {'logKd': None, 'logK_S_M': 0., 'logK_S_D': 0.0, 'logK_S_DS': -0.0, 
               'logK_I_M': None, 'logK_I_D': None, 'logK_I_DI': None, 'logK_S_DI': 0.}
params_kcat = {'kcat_MS': 0.0, 'kcat_DS': 0.0, 'kcat_DSI': 0., 'kcat_DSS': 0.0}

kernel = NUTS(global_fitting)
mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, 
            progress_bar=True)
mcmc.run(rng_key_, experiments, params_logK, params_kcat)
mcmc.print_summary()

trace = mcmc.get_samples(group_by_chain=False)
pickle.dump(trace, open(os.path.join(args.out_dir, 'traces_fig6b.pickle'), "wb"))

sample = az.convert_to_inference_data(mcmc.get_samples(group_by_chain=True))
az.plot_trace(sample)
plt.tight_layout();
plt.savefig(os.path.join(args.out_dir,'traces_fixed_fig6b.pdf'))
plt.ioff()
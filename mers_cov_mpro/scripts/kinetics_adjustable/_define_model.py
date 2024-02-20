import sys
import os
import pickle
import numpy as np
import arviz as az
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from _model_mers import adjustable_global_fitting

class kinetic_model:

    def __init__(self, rng_key_, data, prior, setting, constraints, init_values=None, shared_params=None,
                 out_dir=None, traces_name='traces', trace=None):

        self.rng_key_ = rng_key_
        self.data = data
        self.prior = prior
        self.nburn  = setting['nburn']
        self.niters = setting['niters']
        self.nchain = setting['nchain']
        self.nthin  = setting['nthin']

        self.set_K_I_M_equal_K_S_M        = constraints['set_K_I_M_equal_K_S_M']
        self.set_K_S_DI_equal_K_S_DS      = constraints['set_K_S_DI_equal_K_S_DS']
        self.set_kcat_DSS_equal_kcat_DS   = constraints['set_kcat_DSS_equal_kcat_DS']
        self.set_kcat_DSI_equal_kcat_DS   = constraints['set_kcat_DSI_equal_kcat_DS']
        self.set_kcat_DSI_equal_kcat_DSS  = constraints['set_kcat_DSI_equal_kcat_DSS']
        self.constraint_logK_S_DS         = constraints['constraint_logK_S_DS']
        self.constraint_logK_I_M          = constraints['constraint_logK_I_M']

        self.init_values = init_values
        self.shared_params = shared_params
        self.out_dir = out_dir
        self.traces_name = traces_name

        self.mcmc  = None
        self.trace = trace

    def fitting(self):

        if self.init_values is None:
            kernel = NUTS(adjustable_global_fitting)
        else:
            print("Initial values:", init_values)
            kernel = NUTS(model=adjustable_global_fitting, init_strategy=init_to_value(values=self.init_values))

        mcmc = MCMC(kernel, num_warmup=self.nburn, num_samples=self.niters, num_chains=self.nchain, progress_bar=True)
        mcmc.run(rng_key_, experiments=self.data, prior_infor=self.prior, shared_params=self.shared_params,
                 set_K_I_M_equal_K_S_M=self.set_K_I_M_equal_K_S_M, set_K_S_DI_equal_K_S_DS=self.set_K_S_DI_equal_K_S_DS,
                 set_kcat_DSS_equal_kcat_DS=self.set_kcat_DSS_equal_kcat_DS, set_kcat_DSI_equal_kcat_DS=self.set_kcat_DSI_equal_kcat_DS,
                 set_kcat_DSI_equal_kcat_DSS=self.set_kcat_DSI_equal_kcat_DSS,
                 constraint_logK_S_DS=self.constraint_logK_S_DS, constraint_logK_I_M=self.constraint_logK_I_M)
        self.mcmc = mcmc
        self.trace = self.mcmc.get_samples(group_by_chain=False)
        self.trace_group = self.mcmc.get_samples(group_by_chain=True)

    def summary(self):
        if self.mcmc is not None:
            self.mcmc.print_summary()
            trace = self.mcmc.get_samples(group_by_chain=True)
        elif self.trace_group is not None:
            trace = self.trace_group
        if self.out_dir is not None:
            az.summary(trace).to_csv(os.path.join(self.out_dir, self.traces_name+"_summary.csv"))

    def plot_autocorrelation(self):
        if self.trace is not None:
            az.plot_autocorr(self.trace);
            if self.out_dir is not None:
                plt.savefig(os.path.join(self.out_dir, 'Plot_autocorr'))
                plt.ioff()

    def plot_trace(self):
        if self.trace is not None:
            if len(self.trace.keys())>=10:
                for param_name in ['logK', 'kcat', 'log_sigma', 'alpha', 'error', 'logE', 'logS', 'I:']:
                    trace_group = {}
                    for key in self.trace.keys():
                        if key.startswith(param_name):
                            trace_group[key] = np.reshape(self.trace[key], (self.nchain, self.niters))
                    if len(trace_group)>0:
                        data = az.convert_to_inference_data(trace_group)
                        az.plot_trace(data, compact=False)
                        plt.tight_layout();
                        if self.out_dir is not None:
                            plt.savefig(os.path.join(self.out_dir, 'Plot_trace_'+param_name))
                            plt.ioff()
            else:
                trace_group = {}
                for key in self.trace.keys():
                    trace_group[key] = np.reshape(self.trace[key], (self.nchain, self.niters))
                data = az.convert_to_inference_data(trace_group)
                az.plot_trace(data, compact=False)
                plt.tight_layout();
                if self.out_dir is not None:
                    plt.savefig(os.path.join(args.out_dir, 'Plot_trace'))
                    plt.ioff()

    def save_trace(self):
        if self.trace is not None and self.out_dir is not None:
            pickle.dump(self.trace, open(os.path.join(self.out_dir, self.traces_name+'.pickle'), "wb"))
        else:
            print("Please provide the output directory.")
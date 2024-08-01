import os
import pickle

import arviz as az

import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from _plotting import plotting_trace
from _model import global_fitting, EI_fitting


def _run_mcmc(expts, prior_infor, shared_params, init_values, args):
    """
    Parameters:
    ----------
    expts           : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    shared_params   : dict, information for shared parameters
    init_values     : dict, initial value for model fitting
    args            : class comprises other model arguments. For more information, check _define_model.py
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of all datasets
    
    Return mcmc.trace
    """
    rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
    os.chdir(args.out_dir)
    traces_name = args.traces_name

    if not os.path.isfile(traces_name+'.pickle'):
        if not init_values is None:
            kernel = NUTS(model=global_fitting, init_strategy=init_to_value(values=init_values))
        else:
            kernel = NUTS(global_fitting)

        if os.path.isfile(os.path.join(args.last_run_dir, "Last_state.pickle")):
            last_state = pickle.load(open(os.path.join(args.last_run_dir, "Last_state.pickle"), "rb"))
            print("\nKeep running from last state.")
            mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
            mcmc.post_warmup_state = last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, experiments=expts, 
                     prior_infor=prior_infor, shared_params=shared_params, args=args)
        else:
            mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
            mcmc.run(rng_key_, experiments=expts, prior_infor=prior_infor, shared_params=shared_params, args=args)
        
        mcmc.print_summary()

        print("Saving last state.")
        mcmc.post_warmup_state = mcmc.last_state
        pickle.dump(jax.device_get(mcmc.post_warmup_state), open("Last_state.pickle", "wb"))

        trace = mcmc.get_samples(group_by_chain=False)
        pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

        if not os.path.isdir('Trace_plot'):
            os.mkdir('Trace_plot')

        ## Trace and autocorrelation plots
        plotting_trace(trace=trace, out_dir=os.path.join(args.out_dir, 'Trace_plot'), nchain=args.nchain)

        trace = mcmc.get_samples(group_by_chain=True)
        az.summary(trace).to_csv(traces_name+"_summary.csv")

        trace = mcmc.get_samples(group_by_chain=False)
    else:
        trace = pickle.load(open(traces_name+'.pickle', "rb"))

    return trace


def _run_mcmc_EI(expts, prior_infor, shared_params, init_values, args):
    """
    Parameters:
    ----------
    expts           : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    shared_params   : dict, information for shared parameters
    init_values     : dict, initial value for model fitting
    args            : class comprises other model arguments. For more information, check _define_model.py
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of all datasets
    
    Return mcmc.trace
    """
    rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
    os.chdir(args.out_dir)
    traces_name = args.traces_name

    if not os.path.isfile(traces_name+'.pickle'):
        if not init_values is None:
            kernel = NUTS(model=EI_fitting, init_strategy=init_to_value(values=init_values))
        else:
            kernel = NUTS(EI_fitting)

        if os.path.isfile(os.path.join(args.last_run_dir, "Last_state.pickle")):
            last_state = pickle.load(open(os.path.join(args.last_run_dir, "Last_state.pickle"), "rb"))
            print("\nKeep running from last state.")
            mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
            mcmc.post_warmup_state = last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, experiments=expts, 
                     prior_infor=prior_infor, shared_params=shared_params, args=args)
        else:
            mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
            mcmc.run(rng_key_, experiments=expts, prior_infor=prior_infor, shared_params=shared_params, args=args)
        
        mcmc.print_summary()

        print("Saving last state.")
        mcmc.post_warmup_state = mcmc.last_state
        pickle.dump(jax.device_get(mcmc.post_warmup_state), open("Last_state.pickle", "wb"))

        trace = mcmc.get_samples(group_by_chain=False)
        pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

        if not os.path.isdir('Trace_plot'):
            os.mkdir('Trace_plot')

        ## Trace and autocorrelation plots
        plotting_trace(trace=trace, out_dir=os.path.join(args.out_dir, 'Trace_plot'), nchain=args.nchain)

        trace = mcmc.get_samples(group_by_chain=True)
        az.summary(trace).to_csv(traces_name+"_summary.csv")

        trace = mcmc.get_samples(group_by_chain=False)
    else:
        trace = pickle.load(open(traces_name+'.pickle', "rb"))

    return trace
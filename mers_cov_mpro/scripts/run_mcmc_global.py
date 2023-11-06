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

from _model_mers_global import global_fitting
from _load_data_mers import load_data_no_inhibitor, load_data_one_inhibitor
from _plotting import plot_data_conc_log
from _MAP_finding_mers_concs import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="Inhibitor")
parser.add_argument( "--map_file",                      type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_error_E",                   action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=1)

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
expts_no_I, expts_plot_no_I = load_data_no_inhibitor(df_mers[df_mers['Inhibitor (nM)']==0.0], 
                                                     multi_var=args.multi_var)
inhibitor_name = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])
expts = expts_no_I
expts_plot = expts_plot_no_I
for i, name in enumerate(inhibitor_name):
    expts_, expts_plot_ = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==name)*(df_mers['Drop']!=1)],
                                                  multi_var=args.multi_var)
    expts = expts + expts_
    expts_plot = expts_plot + expts_plot_

logKd_min = -27.63
logKd_max = 0.
kcat_min = 0.
kcat_max = 10

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': 'normal', 'loc': -11.51, 'scale': 0.76}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': -13.75, 'upper': 0} #posterior of ES
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': -20.67, 'upper': -12.24} #posterior of ES
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': -18.29, 'upper': 0} #posterior of ES
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': -13.82, 'upper': 0.}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [None, -26.17, -18.31, -18.99, -24.43, -18.77, -26.9, -23.21, -18.55, -19.06, -19.79, -18.9, -17.82, -21.23], 'upper': [None, -7.77, -6.9, -6.9, -7.93, -6.9, -8.99, -8.97, -6.9, -6.9, -6.9, -6.9, -6.9, -7.49]}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': logKd_min, 'upper': [None, -14.87, -14.2, -16.22, -16.03, -14.85, -15.73, -16.46, -14.62, -16.43, -15.85, -15.85, -14.92, -16.26]}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [None, -24., -25.65, -25.29, -24.41, -25.77, -23.9, -22.26, -24.81, -25.13, -25.53, -25.32, -24.79, -24.83], 'upper': [None, -10.24, -11.64, -11.93, -9.7, -11.26, -9.5, -9.81, -11.6, -12.25, -11.52, -12.02, -11.62, -11]}

# prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0., 'lower': kcat_min, 'upper': 5}

shared_params = None

prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv("Prior_infor.csv", index=False)

print("Prior information: \n", pd.DataFrame(prior_infor_update))

if args.fit_E_S and args.fit_E_I: traces_name = "traces"
elif args.fit_E_S: traces_name = "traces_E_S"
elif args.fit_E_I: traces_name = "traces_E_I"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

if os.path.isfile(traces_name+'.pickle'):
    samples = pickle.load(open(traces_name+'.pickle', "rb"))
    trace = {}
    for key in samples.keys():
        trace[key] = np.reshape(samples[key], (args.nchain, args.niters))
else:
    if len(args.map_file)>0 and os.path.isfile(args.map_file):
        init_values = pickle.load(open(args.map_file, "rb"))
        print("Initial values:", init_values)
        kernel = NUTS(model=global_fitting, init_strategy=init_to_value(values=init_values))
        log_sigmas = init_values
    else:
        kernel = NUTS(global_fitting)
        log_sigmas = None

    if os.path.isfile("Last_state.pickle"):
        last_state = pickle.load(open("Last_state.pickle", "rb"))
        print("\nKeep running from last state.")
        mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
        mcmc.post_warmup_state = last_state
        mcmc.run(mcmc.post_warmup_state.rng_key, 
                 experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params,
                 multi_alpha=args.multi_alpha, set_error_E=args.set_error_E, dE=args.dE,
                 multi_var=args.multi_var, log_sigmas=log_sigmas)
    else:
        mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
        mcmc.run(rng_key_, experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params,
                 multi_alpha=args.multi_alpha, set_error_E=args.set_error_E, dE=args.dE, 
                 multi_var=args.multi_var, log_sigmas=log_sigmas)
    
    mcmc.print_summary()

    print("Saving last state.")
    mcmc.post_warmup_state = mcmc.last_state
    pickle.dump(jax.device_get(mcmc.post_warmup_state), open("Last_state.pickle", "wb"))

    trace = mcmc.get_samples(group_by_chain=False)
    pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

    ## Autocorrelation plot
    az.plot_autocorr(trace);
    plt.savefig(os.path.join(args.out_dir, 'Plot_autocorr'))
    plt.ioff()

    trace = mcmc.get_samples(group_by_chain=True)
    az.summary(trace).to_csv(traces_name+"_summary.csv")

## Trace plot
if len(trace.keys())>=10:
    for param_name in ['logK', 'kcat', 'log_sigma', 'alpha', 'dE', 'I0']:
        trace_2 = {}
        for key in trace.keys():
            if key.startswith(param_name):
                trace_2[key] = trace[key]
        if len(trace_2)>0:
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

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': 'normal', 'loc': -11.51, 'scale': 0.76}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': -13.75, 'upper': 0} #posterior of ES
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': -20.67, 'upper': -12.24} #posterior of ES
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': -18.29, 'upper': 0} #posterior of ES
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'local', 'dist': 'uniform', 'lower': -13.82, 'upper': 0.}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'local', 'dist': 'uniform', 'lower': [logKd_min, -26.17, -18.31, -18.99, -24.43, -18.77, -26.9, -23.21, -18.55, -19.06, -19.79, -18.9, -17.82, -21.23], 'upper': [0, -7.77, -6.9, -6.9, -7.93, -6.9, -8.99, -8.97, -6.9, -6.9, -6.9, -6.9, -6.9, -7.49]}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'local', 'dist': 'uniform', 'lower': logKd_min, 'upper': [0., -14.87, -14.2, -16.22, -16.03, -14.85, -15.73, -16.46, -14.62, -16.43, -15.85, -15.85, -14.92, -16.26]}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'local', 'dist': 'uniform', 'lower': [logKd_min, -24., -25.65, -25.29, -24.41, -25.77, -23.9, -22.26, -24.81, -25.13, -25.53, -25.32, -24.79, -24.83], 'upper': [0, -10.24, -11.64, -11.93, -9.7, -11.26, -9.5, -9.81, -11.6, -12.25, -11.52, -12.02, -11.62, -11]}

prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}

prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))

if shared_params is not None and len(shared_params)>0:
    for name in shared_params.keys():
        param = shared_params[name]
        assigned_idx = param['assigned_idx']
        shared_idx = param['shared_idx']
        trace[f'{name}:{assigned_idx}'] = trace[f'{name}:{shared_idx}']

trace_map = trace.copy()
init_values = pickle.load(open(args.map_file, "rb"))
for name in ['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
    trace_map[f'{name}:0'] = trace_map[f'{name}:1']
trace_map['kcat_DSI:0'] = jnp.zeros(args.nchain*args.niters)
for key in init_values.keys():
    if key.startswith('log_sigma'):
        trace_map[key] = jnp.repeat(init_values[key], args.niters*args.nchain)

[map_index, map_params, log_probs] = map_finding(trace_map, expts, prior_infor_update)

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
params_logK, params_kcat = extract_params_from_map_and_prior(trace_map, map_index, prior_infor_update)

if args.set_error_E and args.dE>0:
    _error_E = {key: trace[key][map_index] for key in trace.keys() if key.startswith('dE')}
else: _error_E = None

n = 0
plot_data_conc_log(expts_plot_no_I, extract_logK_n_idx(params_logK, n, shared_params),
                   extract_kcat_n_idx(params_kcat, n, shared_params),
                   line_colors=['black', 'red', 'tab:brown'], ls='-.',
                   error_E=_error_E, plot_legend=True,
                   OUTFILE=os.path.join(args.out_dir,'ES'))
no_expt = 4
_alpha = [trace[f'alpha:ESI:{i}'][map_index] for i in range(no_expt)]
for i in range(len(inhibitor_name)):
    n = i + 1
    plot_data_conc_log(expts_plot[(i*no_expt+3):(i*no_expt+3+no_expt)], extract_logK_n_idx(params_logK, n, shared_params),
                       extract_kcat_n_idx(params_kcat, n, shared_params),
                       alpha=_alpha, error_E=_error_E,
                       OUTFILE=os.path.join(args.out_dir,'ESI_'+str(i)))
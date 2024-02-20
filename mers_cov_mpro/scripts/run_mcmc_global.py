<<<<<<< HEAD
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

from _model_mers_ESI import global_fitting
from _load_data_mers import load_data_no_inhibitor, load_data_one_inhibitor
from _plotting import plot_data_conc_log, plotting_trace_global
from _MAP_finding_mers_concs import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior

from _save_setting import save_model_setting

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--list_inhibitor",                type=str,               default="")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--map_file",                      type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--fixing_log_sigmas",             action="store_true",    default=False)

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
if len(args.list_inhibitor)>0:
    inhibitor_name = np.unique(args.list_inhibitor.split())
else:
    inhibitor_name = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

expts_no_I, expts_plot_no_I = load_data_no_inhibitor(df_mers[df_mers['Inhibitor (nM)']==0.0], 
                                                     multi_var=args.multi_var)
no_expt = [len(expts_plot_no_I)]
expts = expts_no_I
expts_plot = expts_plot_no_I

for i, name in enumerate(inhibitor_name):
    expts_, expts_plot_ = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==name)*(df_mers['Drop']!=1)],
                                                  multi_var=args.multi_var)
    expts = expts + expts_
    expts_plot = expts_plot + expts_plot_
    no_expt.append(len(expts_plot_))

logKd_min = -27.63
logKd_max = 0.
kcat_min = 0.
kcat_max = 20

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': 'normal', 'loc': -9.9, 'scale': 0.5}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': -14, 'upper': 0} 
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': -20.72, 'upper': -13}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': -8, 'upper': 0.}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [-19.11,-12.92,-14.06,-17.,-13.12,-17.6,-17.76,-12.,-14.05,-14.42,-13.97,-12.24,-16.28], 'upper': 0}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [-27.58,-18.84,-19.77,-22.17,-19.64,-25.57,-24.62,-18.16,-19.43,-20.54,-20.08,-18.17,-22.77], 'upper': [-3.41,-0.01,-6.08,-3.94,-4.71,-3.77,-4.4,-3.3,-0.79,-6.2,-5.84,-3.56,-3.38]}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': logKd_min, 'upper': [0,-14.6,-16.04,0,-14.68,0,0,-15.04,-16.33,-15.63,-16.26,-15.4,0.]}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [-27.6,-26.39,-24.85,-27.62,-26.95,-27.59,-27.61,-26.47,-24.63,-25.63,-25.59,-25.98,-27.59], 'upper': [-7.4,0,-11.66,-9.45,-11.24,-9.6,-9.09,-11.81,0,-11.32,-11.64,-11.49,-10.38]}

# prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0., 'lower': kcat_min, 'upper': kcat_max}

shared_params = None

if args.set_K_S_DS_equal_K_S_D: 
    del prior['logK_S_DS']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']

prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv("Prior_infor.csv", index=False)

print("Prior information: \n", pd.DataFrame(prior_infor_update))

if args.fit_E_S and args.fit_E_I: traces_name = "traces"
elif args.fit_E_S: traces_name = "traces_E_S"
elif args.fit_E_I: traces_name = "traces_E_I"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

if args.fixing_log_sigmas:
    log_sigmas = pickle.load(open(args.map_file, "rb"))
    print("Using initial values from MAP to fix ln(sigma).")
else:
    log_sigmas = None

if not os.path.isfile(traces_name+'.pickle'):
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
                 experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params,
                 multi_alpha=args.multi_alpha, alpha_min=0.5, alpha_max=1.5,
                 set_lognormal_dE=args.set_lognormal_dE, dE=args.dE, log_sigmas=log_sigmas,
                 set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
    else:
        mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
        mcmc.run(rng_key_, experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params,
                 multi_alpha=args.multi_alpha, alpha_min=0.5, alpha_max=1.5,
                 set_lognormal_dE=args.set_lognormal_dE, dE=args.dE, log_sigmas=log_sigmas,
                 set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
    
    mcmc.print_summary()

    print("Saving last state.")
    mcmc.post_warmup_state = mcmc.last_state
    pickle.dump(jax.device_get(mcmc.post_warmup_state), open("Last_state.pickle", "wb"))

    trace = mcmc.get_samples(group_by_chain=False)
    pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

    os.mkdir('Trace_plot')
    ## Trace and autocorrelation plots
    plotting_trace_global(trace=trace, out_dir=os.path.join(args.out_dir, 'Trace_plot'), nchain=args.nchain)

    trace = mcmc.get_samples(group_by_chain=True)
    az.summary(trace).to_csv(traces_name+"_summary.csv")

# Finding MAP
if os.path.isfile(traces_name+'.pickle'):
    trace = pickle.load(open(traces_name+'.pickle', "rb"))
else:
    trace = mcmc.get_samples(group_by_chain=False)

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': 'normal', 'loc': -9.9, 'scale': 0.5}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': -14, 'upper': 0} 
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': -18, 'upper': -12}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': -17.5, 'upper': 0.}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'local', 'dist': 'uniform', 'lower': -20.73, 'upper': 0.}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'local', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'local', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'local', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}

if args.set_K_S_DS_equal_K_S_D: 
    del prior['logK_S_DS']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']

pickle.dump(prior, open(os.path.join('MAP_prior.pickle'), "wb"))
prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))

if shared_params is not None and len(shared_params)>0:
    for name in shared_params.keys():
        param = shared_params[name]
        assigned_idx = param['assigned_idx']
        shared_idx = param['shared_idx']
        trace[f'{name}:{assigned_idx}'] = trace[f'{name}:{shared_idx}']

trace_map = trace.copy()
for name in ['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
    if f'{name}:1' in trace_map.keys():
        trace_map[f'{name}:0'] = trace_map[f'{name}:1']
trace_map['kcat_DSI:0'] = jnp.zeros(args.nchain*args.niters)

if args.fixing_log_sigmas and log_sigmas is not None: 
    for key in init_values.keys():
        if key.startswith('log_sigma'):
            trace_map[key] = jnp.repeat(init_values[key], args.niters*args.nchain)

pickle.dump(trace_map, open(os.path.join('MAP_'+traces_name+'.pickle'), "wb"))

[map_index, map_params, log_probs] = map_finding(trace_map, expts, prior_infor_update, 
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
params_logK, params_kcat = extract_params_from_map_and_prior(trace_map, map_index, prior_infor_update)

if args.set_lognormal_dE and args.dE>0:
    E_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('dE')}
else: E_list = None

alpha_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('alpha')}
if len(alpha_list) == 0:
    alpha_list = None

for n in range(len(expts)):
    if args.set_K_S_DS_equal_K_S_D:
        try: params_logK[f'logK_S_DS:{n}'] = params_logK[f'logK_S_D:{n}']
        except: params_logK['logK_S_DS'] = params_logK['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        try: params_logK[f'logK_S_DI:{n}'] = params_logK[f'logK_S_DS:{n}']
        except: params_logK['logK_S_DI'] = params_logK['logK_S_DS']

if not os.path.isdir('Fitting'):
    os.mkdir('Fitting')

n = 0
plot_data_conc_log(expts_plot_no_I, extract_logK_n_idx(params_logK, n, shared_params),
                   extract_kcat_n_idx(params_kcat, n, shared_params),
                   line_colors=['black', 'red', 'tab:brown'], ls='-.',
                   E_list=E_list, plot_legend=True,
                   OUTFILE=os.path.join(args.out_dir, 'Fitting', 'ES'))

start = 0
end = 3
for i in range(len(inhibitor_name)):
    n = i + 1
    start = end
    end   = end+no_expt[n]
    plot_data_conc_log(expts_plot[start:end], extract_logK_n_idx(params_logK, n, shared_params),
                       extract_kcat_n_idx(params_kcat, n, shared_params),
                       alpha_list=alpha_list, E_list=E_list,
                       OUTFILE=os.path.join(args.out_dir, 'Fitting', 'ESI_'+str(i)))

## Saving the model fitting condition
save_model_setting(args, OUTDIR=args.out_dir, OUTFILE='setting.pickle')
=======
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
from _plotting import plot_data_conc_log, plotting_trace_global
from _MAP_finding_mers_concs import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--map_file",                      type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--fixing_log_sigmas",             action="store_true",    default=False)

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
kcat_max = 20

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': 'normal', 'loc': -9.9, 'scale': 0.5}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': -14, 'upper': 0} 
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': -20.72, 'upper': -13}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': -12, 'upper': 0.}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [logKd_min,-17.23,-13.45,-14.41,-12.38,-12.89,-18.25,-7.83,-11.75,-14.22,-14.4,-14.,-12.34,-16.24], 'upper': 0}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [logKd_min,-25.31,-18.8,-19.75,-24.19,-19.4,-27.22,-20.78,-18.06,-20.04,-21.01,-20.52,-18.99,-23.02], 'upper': [0,-4.95,-3.82,-6.82,-9.96,-4.43,-9.57,-10.9,-3.98,-6.06,-6.55,-6.73,-3.9,-5.47]}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': logKd_min, 'upper': [0,0,-14.92,-15.95,-16.36,-14.29,-7.07,-19.22,-15.19,-16.14,-15.05,-15.48,-15.27,0]}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0, 'lower': [logKd_min,-27.58,-26.28,-25.1,-24.1,-27.05,-23.83,-23.74,-26.54,-24.1,-25.1,-24.59,-25.95,-27.09], 'upper': [0,-8.7,-11.34,-11.85,-9.41,-10.64,-8.62,-15.29,-11.72,-11.16,-10.49,-11.13,-11.82,-9.87]}

# prior['kcat_MS'] = {'type':'kcat', 'name': 'kcat_MS', 'fit':'global', 'dist': None, 'value': 0.}
prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': [None, 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 'value': 0., 'lower': kcat_min, 'upper': kcat_max}

shared_params = None

if args.set_K_S_DS_equal_K_S_D: 
    del prior['logK_S_DS']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']

prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv("Prior_infor.csv", index=False)

print("Prior information: \n", pd.DataFrame(prior_infor_update))

if args.fit_E_S and args.fit_E_I: traces_name = "traces"
elif args.fit_E_S: traces_name = "traces_E_S"
elif args.fit_E_I: traces_name = "traces_E_I"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

if args.fixing_log_sigmas:
    log_sigmas = pickle.load(open(args.map_file, "rb"))
    print("Using initial values from MAP to fix ln(sigma).")
else:
    log_sigmas = None

if not os.path.isfile(traces_name+'.pickle'):
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
                 experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params,
                 multi_alpha=args.multi_alpha, set_lognormal_dE=args.set_lognormal_dE, dE=args.dE, 
                 multi_var=args.multi_var, log_sigmas=log_sigmas,
                 set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
    else:
        mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
        mcmc.run(rng_key_, experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params,
                 multi_alpha=args.multi_alpha, set_lognormal_dE=args.set_lognormal_dE, dE=args.dE, 
                 multi_var=args.multi_var, log_sigmas=log_sigmas,
                 set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
    
    mcmc.print_summary()

    print("Saving last state.")
    mcmc.post_warmup_state = mcmc.last_state
    pickle.dump(jax.device_get(mcmc.post_warmup_state), open("Last_state.pickle", "wb"))

    trace = mcmc.get_samples(group_by_chain=False)
    pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

    os.mkdir('Trace_plot')
    ## Trace and autocorrelation plots
    plotting_trace_global(trace, os.path.join(args.out_dir, 'Trace_plot'), args.nchain, args.niters)

    trace = mcmc.get_samples(group_by_chain=True)
    az.summary(trace).to_csv(traces_name+"_summary.csv")

# Finding MAP
if os.path.isfile(traces_name+'.pickle'):
    trace = pickle.load(open(traces_name+'.pickle', "rb"))
else:
    trace = mcmc.get_samples(group_by_chain=False)

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': 'normal', 'loc': -9.9, 'scale': 0.5}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': 'uniform', 'lower': -14, 'upper': 0} 
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': 'uniform', 'lower': -18, 'upper': -12}
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': 'uniform', 'lower': -17.5, 'upper': 0.}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'local', 'dist': 'uniform', 'lower': -20.73, 'upper': 0.}
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'local', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'local', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'local', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': 5}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'local', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}

if args.set_K_S_DS_equal_K_S_D: 
    del prior['logK_S_DS']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']

pickle.dump(prior, open(os.path.join('MAP_prior.pickle'), "wb"))
prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))

if shared_params is not None and len(shared_params)>0:
    for name in shared_params.keys():
        param = shared_params[name]
        assigned_idx = param['assigned_idx']
        shared_idx = param['shared_idx']
        trace[f'{name}:{assigned_idx}'] = trace[f'{name}:{shared_idx}']

trace_map = trace.copy()
for name in ['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
    if f'{name}:1' in trace_map.keys():
        trace_map[f'{name}:0'] = trace_map[f'{name}:1']
trace_map['kcat_DSI:0'] = jnp.zeros(args.nchain*args.niters)

if args.fixing_log_sigmas and log_sigmas is not None: 
    for key in init_values.keys():
        if key.startswith('log_sigma'):
            trace_map[key] = jnp.repeat(init_values[key], args.niters*args.nchain)

pickle.dump(trace_map, open(os.path.join('MAP_'+traces_name+'.pickle'), "wb"))

[map_index, map_params, log_probs] = map_finding(trace_map, expts, prior_infor_update, 
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
params_logK, params_kcat = extract_params_from_map_and_prior(trace_map, map_index, prior_infor_update)

if args.set_lognormal_dE and args.dE>0:
    E_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('dE')}
else: E_list = None

for n in range(len(expts)):
    if args.set_K_S_DS_equal_K_S_D:
        try: params_logK[f'logK_S_DS:{n}'] = params_logK[f'logK_S_D:{n}']
        except: params_logK['logK_S_DS'] = params_logK['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        try: params_logK[f'logK_S_DI:{n}'] = params_logK[f'logK_S_DS:{n}']
        except: params_logK['logK_S_DI'] = params_logK['logK_S_DS']

os.mkdir('Fitting')
n = 0
plot_data_conc_log(expts_plot_no_I, extract_logK_n_idx(params_logK, n, shared_params),
                   extract_kcat_n_idx(params_kcat, n, shared_params),
                   line_colors=['black', 'red', 'tab:brown'], ls='-.',
                   error_E=E_list, plot_legend=True,
                   OUTFILE=os.path.join(args.out_dir, 'Fitting', 'ES'))
no_expt = 4
_alpha = [trace[f'alpha:ESI:{i}'][map_index] for i in range(no_expt)]
for i in range(len(inhibitor_name)):
    n = i + 1
    plot_data_conc_log(expts_plot[(i*no_expt+3):(i*no_expt+3+no_expt)], extract_logK_n_idx(params_logK, n, shared_params),
                       extract_kcat_n_idx(params_kcat, n, shared_params),
                       alpha=_alpha, error_E=E_list,
                       OUTFILE=os.path.join(args.out_dir, 'Fitting', 'ESI_'+str(i)))
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0

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

from pymbar import timeseries

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore")

from _CRC_fitting import CRC_EI_fitting, _expt_check_noise_trend
from _load_data_mers import load_data_no_inhibitor, load_data_one_inhibitor
from _plotting import plot_data_conc_log, plotting_trace
from _MAP_finding_mers_concs import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, _trace_convergence, _convergence_rhat

from _pIC50 import _adjust_trace, _pIC50_hill
from _save_setting import save_model_setting

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="Inhibitor")
parser.add_argument( "--map_file",                      type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=1000)
parser.add_argument( "--nburn",                         type=int,               default=200)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

parser.add_argument( "--outlier_removal",               action="store_true",    default=False)
parser.add_argument( "--exclude_first_trace",           action="store_true",    default=True)
parser.add_argument( "--key_to_check",                  type=str,               default="")
parser.add_argument( "--converged_samples",             type=int,               default=500)

parser.add_argument( "--enzyme_conc_nM",                type=int,               default="100")
parser.add_argument( "--inhibitor_conc_nM",             type=int,               default="1350")

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

### Data
df_mers = pd.read_csv(args.input_file)

inhibitor_name = args.name_inhibitor.split()
for i, name in enumerate(inhibitor_name):
    expts_init, expts_plot = load_data_one_inhibitor(df_mers[(df_mers['Inhibitor_ID']==name+'-001')*(df_mers['Drop']!=1.0)],
                                                     multi_var=args.multi_var)

## Outlier detection and trend checking
[expts_outliers, outliers, _, _] = _expt_check_noise_trend(expts_init)
if args.outlier_removal:
    expts = expts_outliers.copy()
else:
    expts = expts_init.copy()

if len(args.map_file)>0 and os.path.isfile(args.map_file):
    map_sampling = pickle.load(open(args.map_file, "rb"))

    if args.set_K_S_DS_equal_K_S_D:
        map_sampling['logK_S_DS'] = map_sampling['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        map_sampling['logK_S_DI'] = map_sampling['logK_S_DS']

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'kcat_DS', 'kcat_DSS']:
        assert key in map_sampling.keys(), f"Please provide {key} in map_file."

### Prior
logKd_min = -27.63
logKd_max = 0.
kcat_min = 0.
kcat_max = 10

prior = {}
prior['logKd'] = {'type':'logKd', 'name': 'logKd', 'fit':'global', 'dist': None, 'value': map_sampling['logKd']}
prior['logK_S_M'] = {'type':'logK', 'name': 'logK_S_M', 'fit':'global', 'dist': None, 'value': map_sampling['logK_S_M']} 
prior['logK_S_D'] = {'type':'logK', 'name': 'logK_S_D', 'fit':'global', 'dist': None, 'value': map_sampling['logK_S_D']} 
prior['logK_S_DS'] = {'type':'logK', 'name': 'logK_S_DS', 'fit':'global', 'dist': None, 'value': map_sampling['logK_S_DS']}
prior['logK_I_M'] = {'type':'logK', 'name': 'logK_I_M', 'fit':'global', 'dist': 'uniform', 'lower': -20.73, 'upper': logKd_max} #1E-9
prior['logK_I_D'] = {'type':'logK', 'name': 'logK_I_D', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_I_DI'] = {'type':'logK', 'name': 'logK_I_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}
prior['logK_S_DI'] = {'type':'logK', 'name': 'logK_S_DI', 'fit':'global', 'dist': 'uniform', 'lower': logKd_min, 'upper': logKd_max}

prior['kcat_DS'] = {'type':'kcat', 'name': 'kcat_DS', 'fit':'global', 'dist': None, 'value': map_sampling['kcat_DS']}
prior['kcat_DSS'] = {'type':'kcat', 'name': 'kcat_DSS', 'fit':'global', 'dist': None, 'value': map_sampling['kcat_DSS']}
prior['kcat_DSI'] = {'type':'kcat', 'name': 'kcat_DSI', 'fit':'global', 'dist': 'uniform', 'lower': kcat_min, 'upper': kcat_max}

shared_params = None

if args.set_K_S_DS_equal_K_S_D: 
    del prior['logK_S_DS']
if args.set_K_S_DI_equal_K_S_DS: 
    del prior['logK_S_DI']

pickle.dump(prior, open(os.path.join(args.out_dir, 'Prior.pickle'), "wb"))
prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
prior_infor_update = check_prior_group(prior_infor, len(expts))
pd.DataFrame(prior_infor_update).to_csv(os.path.join(args.out_dir, "Prior_infor.csv"), index=False)

# print("Prior information: \n", pd.DataFrame(prior_infor_update), "\n")

# E_list = {}
# for key in ['dE:100', 'dE:50', 'dE:25']:
#     E_list[key] = map_sampling[key]

### Other information for fitting
if args.fit_E_S and args.fit_E_I: traces_name = "traces"
elif args.fit_E_S: traces_name = "traces_E_S"
elif args.fit_E_I: traces_name = "traces_E_I"

rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
os.chdir(args.out_dir)

### Concentration for the estimation of dimer-only pIC50
init_logMtot = np.log(args.enzyme_conc_nM*1E-9)
init_logStot = np.log(args.inhibitor_conc_nM*1E-9)
init_logDtot = init_logMtot-np.log(2)

n_points = 50
min_conc = 1E-12
max_conc = 1E-3

logDtot = np.ones(n_points)*init_logDtot
logStot = np.ones(n_points)*init_logStot
logItot = np.linspace(np.log(min_conc), np.log(max_conc), n_points)

if len(args.map_file)>0 and os.path.isfile(args.map_file):
    init_values = pickle.load(open(args.map_file, "rb"))
    # print("Initial values:", init_values, "\n")
    kernel = NUTS(model=CRC_EI_fitting, init_strategy=init_to_value(values=init_values))
    logK_dE_alpha = init_values
else:
    kernel = NUTS(CRC_EI_fitting)
    logK_dE_alpha = None

if len(args.key_to_check)>0:
    key_to_check = args.key_to_check.split()
else:
    key_to_check = ''

if not os.path.isdir(os.path.join(args.out_dir, 'Convergence')):
    os.mkdir(os.path.join(args.out_dir, 'Convergence'))

no_limit = 10
no_running = 1

print(f"\nAnalyzing {inhibitor_name[0]}")

while no_running<=no_limit:

    ### Fitting
    if not os.path.isdir(os.path.join(args.out_dir, f'sampling_{no_running}')):
        os.mkdir(os.path.join(args.out_dir, f'sampling_{no_running}'))

    name_expt = inhibitor_name[0][7:12]
    expt_dir = os.path.join(args.out_dir, f'sampling_{no_running}', name_expt)
    last_dir = os.path.join(args.out_dir, f'sampling_{no_running-1}', name_expt)
    
    if not os.path.isfile(os.path.join(expt_dir, traces_name+'.pickle')):
        if not os.path.isdir(expt_dir):
            os.mkdir(expt_dir)

        mes = f'\nFitting sampling_{no_running}/{name_expt}:'
        print(mes)
        with open(os.path.join(expt_dir, "log.txt"), "a") as f:
            print(mes, file=f)

        if no_running>2 and os.path.isfile(os.path.join(last_dir, "Last_state.pickle")):
            last_state = pickle.load(open(os.path.join(last_dir, "Last_state.pickle"), "rb"))
            mes = "Keep running from last state."
            print(mes)
            with open(os.path.join(expt_dir, "log.txt"), "a") as f:
                print(mes, file=f)

            mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
            mcmc.post_warmup_state = last_state
            mcmc.run(mcmc.post_warmup_state.rng_key,
                     experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params,
                     multi_alpha=args.multi_alpha, set_lognormal_dE=args.set_lognormal_dE, dE=args.dE, 
                     set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
        else:
            mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
            mcmc.run(rng_key_, experiments=expts, prior_infor=prior_infor_update, shared_params=shared_params,
                     multi_alpha=args.multi_alpha, set_lognormal_dE=args.set_lognormal_dE, dE=args.dE, 
                     set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D, set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)
        
        mcmc.post_warmup_state = mcmc.last_state
        pickle.dump(jax.device_get(mcmc.post_warmup_state), open(os.path.join(expt_dir, "Last_state.pickle"), "wb"))

        mcmc.print_summary()

        trace = mcmc.get_samples(group_by_chain=True)
        az.summary(trace).to_csv(os.path.join(expt_dir, traces_name+"_summary.csv"))

        trace = mcmc.get_samples(group_by_chain=False)
        pickle.dump(trace, open(os.path.join(expt_dir, traces_name+'.pickle'), "wb"))

        plotting_trace(trace, expt_dir, nchain=args.nchain, nsample=args.niters)

        if shared_params is not None and len(shared_params)>0:
            for name in shared_params.keys():
                param = shared_params[name]
                assigned_idx = param['assigned_idx']
                shared_idx = param['shared_idx']
                trace[f'{name}:{assigned_idx}'] = trace[f'{name}:{shared_idx}']

        trace_map = trace.copy()
        [map_index, map_params, log_probs] = map_finding(trace_map, expts, prior_infor=prior_infor_update, 
                                                         set_lognormal_dE=args.set_lognormal_dE, dE=args.dE,
                                                         set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D,
                                                         set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)

        with open(os.path.join(expt_dir,"map.txt"), "w") as f:
            print("MAP index:" + str(map_index), file=f)
            print("\nKinetics parameters:", file=f)
            for key in trace.keys():
                print(key, ': %.3f' %trace[key][map_index], file=f)

        pickle.dump(log_probs, open(os.path.join(expt_dir,'log_probs.pickle'), "wb"))

        map_values = {}
        for key in trace.keys():
            map_values[key] = trace[key][map_index]
        map_input = pickle.load(open(args.map_file, "rb"))
        for key in 'logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'kcat_DS', 'kcat_DSS':
            if not key in map_values:
                map_values[key] = map_input[key]
        pickle.dump(map_values, open(os.path.join(expt_dir,'map.pickle'), "wb"))

        ## Fitting plot
        params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

        alpha_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('alpha')}
        E_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('dE')}

        n = 0
        plot_data_conc_log(expts_plot, extract_logK_n_idx(params_logK, n, shared_params),
                           extract_kcat_n_idx(params_kcat, n, shared_params),
                           alpha_list=alpha_list, E_list=E_list, outliers=outliers,
                           OUTFILE=os.path.join(expt_dir,'EI'))
            
        ## Saving the model fitting condition
        save_model_setting(args, OUTDIR=expt_dir, OUTFILE='setting.pickle')
        
        del trace, trace_map

    ### Extracting all traces.pickles of one experiment from multiple sampling runs
    if not os.path.isdir(os.path.join(args.out_dir, 'Convergence', name_expt)):
        os.mkdir(os.path.join(args.out_dir, 'Convergence', name_expt))
    
    _trace_files = [f'{args.out_dir}/sampling_{i}/{name_expt}/{traces_name}.pickle' for i in range(1, no_running+1)]
    if args.exclude_first_trace and len(_trace_files)>1: 
        trace_files = _trace_files[1:]
    else:
        trace_files = _trace_files

    [trace, flag, nchain_updated] = _trace_convergence(mcmc_files=trace_files, out_dir=os.path.join(args.out_dir, 'Convergence', name_expt), 
                                                       nchain=args.nchain, expected_nsample=args.converged_samples,
                                                       key_to_check=key_to_check, one_chain_removal=True)

    if flag: #if number of converged samples returned from pymbar is enough
        data = az.InferenceData.to_dataframe(az.convert_to_inference_data(trace))
        
        ### Estimating dimer-only pIC50
        _nthin = int(len(data)/100)
        if logK_dE_alpha is not None:
            df = _adjust_trace(data.iloc[::_nthin, :].copy(), logK_dE_alpha)
        else:
            df = data.iloc[::_nthin, :].copy()

        pIC50_list, hill_list = _pIC50_hill(df, logDtot, logStot, logItot)
        
        if _convergence_rhat(trace=pIC50_list, nchain=nchain_updated, digit=1):
            
            pIC50 = pIC50_list[hill_list>0]
            mes = "pIC50: %0.3f" % np.median(pIC50) + " +- %0.3f" % np.std(pIC50) + "\n"
            
            print(mes)
            with open(os.path.join(args.out_dir, 'Convergence', name_expt, "log.txt"), "a") as f:
                print(mes, file=f)
            
            pickle.dump(trace, open(os.path.join(args.out_dir, 'Convergence', name_expt, "traces.pickle"), "wb"))
            plotting_trace(trace, os.path.join(args.out_dir, 'Convergence', name_expt), args.nchain)
            
            break

    del trace
    no_running += 1


if no_running > no_limit:
    mes = "The number of fitting was exceeded."
    print(mes)
    with open(os.path.join(args.out_dir, 'Convergence', name_expt, "log.txt"), "a") as f:
        print(mes, file=f)

[expts_outliers, _, mes_noise, mes_trend] = _expt_check_noise_trend(expts_init, OUT_DIR=os.path.join(args.out_dir, 'Convergence', name_expt))

## Report if curve is increasing or decreasing
if mes_noise is not None and len(mes_noise)>0:
    for _mes in mes_noise:
        print(_mes)
        with open(os.path.join(args.out_dir, 'Convergence', name_expt, "log.txt"), "a") as f:
            print(_mes, file=f)

if mes_trend is not None and len(mes_trend)>0:
    for _mes in mes_trend:
        print(_mes)
        with open(os.path.join(args.out_dir, 'Convergence', name_expt, "log.txt"), "a") as f:
            print(_mes, file=f)
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

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _load_data_mers import load_data_no_inhibitor, load_data_one_inhibitor
from _plotting import plot_data_conc_log
from _MAP_finding_mers_concs import map_finding

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _trace_analysis import extract_params_from_map_and_prior, extract_params_from_trace_and_prior

parser = argparse.ArgumentParser()

parser.add_argument( "--kinetic_file",                  type=str,               default="")
parser.add_argument( "--mcmc_file",                     type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="")
parser.add_argument( "--prior_infor",                   type=str,               default="")
parser.add_argument( "--map_file",                      type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--global_fitting",                action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=0)
parser.add_argument( "--nchain",                        type=int,               default=4)

parser.add_argument( "--nsamples",                      type=str,               default=None)

args = parser.parse_args()

# Loading experimental information
df_mers = pd.read_csv(args.kinetic_file)
if len(args.name_inhibitor)>0:
    inhibitor_name = np.array([args.name_inhibitor.split()+'-001'])
else:
    inhibitor_name = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

expts_no_I, expts_plot_no_I = load_data_no_inhibitor(df_mers[df_mers['Inhibitor (nM)']==0.0], 
                                                     multi_var=args.multi_var)
no_expt = [len(expts_plot_no_I)]
expts = expts_no_I
expts_plot = expts_plot_no_I
for i, name in enumerate(inhibitor_name):
    expts_, expts_plot_ = load_data_one_inhibitor(df_mers[df_mers['Inhibitor_ID']==name],
                                                  multi_var=args.multi_var)
    expts = expts + expts_
    expts_plot = expts_plot + expts_plot_
    no_expt.append(len(expts_plot_))

shared_params = None

trace = pickle.load(open(args.mcmc_file, "rb"))

if args.global_fitting:
    assert os.path.isfile(args.prior_infor), "Please provide the prior information."
    prior = pickle.load(open(args.prior_infor, "rb"))

    if args.set_K_S_DS_equal_K_S_D:
        assert not 'logK_S_DS' in prior.keys(), "The constraint was incorrect."
    if args.set_K_S_DI_equal_K_S_DS:
        assert not 'logK_S_DI' in prior.keys(), "The constraint was incorrect."

    pickle.dump(prior, open(os.path.join('MAP_prior.pickle'), "wb"))
    prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
    prior_infor_update = check_prior_group(prior_infor, len(expts))
    pd.DataFrame(prior_infor_update).to_csv(os.path.join(args.out_dir, "Prior_infor.csv"), index=False)
    print("Prior information: \n", pd.DataFrame(prior_infor_update))

    trace_map = trace.copy()
    if args.niters == 0:
        key = list(trace.keys())
        niters = int(len(trace[key[0]])/args.nchain)

    for name in ['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']:
        if f'{name}:1' in trace_map.keys():
            trace_map[f'{name}:0'] = trace_map[f'{name}:1']
    trace_map['kcat_DSI:0'] = jnp.zeros(args.nchain*niters)
    pickle.dump(trace_map, open(os.path.join('MAP_traces.pickle'), "wb"))

    [map_index, map_params, log_probs] = map_finding(trace_map, expts, prior_infor_update, 
                                                     set_lognormal_dE=args.set_lognormal_dE, dE=args.dE,
                                                     set_K_S_DS_equal_K_S_D=args.set_K_S_DS_equal_K_S_D,
                                                     set_K_S_DI_equal_K_S_DS=args.set_K_S_DI_equal_K_S_DS)

    if args.set_lognormal_dE and args.dE>0:
        E_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('dE')}
    else: E_list = None

    alpha_list = {key: trace[key][map_index] for key in trace.keys() if key.startswith('alpha')}
    if len(alpha_list) == 0:
        alpha_list = None

else:
    # Loading prior information
    assert os.path.isfile(args.prior_infor), "Please provide the prior information."
    prior = pickle.load(open(args.prior_infor, "rb"))

    if args.set_K_S_DS_equal_K_S_D:
        assert not 'logK_S_DS' in prior.keys(), "The constraint was incorrect."
    if args.set_K_S_DI_equal_K_S_DS:
        assert not 'logK_S_DI' in prior.keys(), "The constraint was incorrect."

    pickle.dump(prior, open(os.path.join('MAP_prior.pickle'), "wb"))
    prior_infor = convert_prior_from_dict_to_list(prior, args.fit_E_S, args.fit_E_I)
    prior_infor_update = check_prior_group(prior_infor, len(expts))
    pd.DataFrame(prior_infor_update).to_csv(os.path.join(args.out_dir, "Prior_infor.csv"), index=False)
    print("Prior information: \n", pd.DataFrame(prior_infor_update))

    if len(args.map_file)>0 and os.path.isfile(args.map_file):
        map_sampling = pickle.load(open(args.map_file, "rb"))
    
    if args.set_K_S_DS_equal_K_S_D: 
        map_sampling['logK_S_DS'] = map_sampling['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        map_sampling['logK_S_DI'] = map_sampling['logK_S_DS']

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'alpha:E100_S1350', 'alpha:E100_S50', 'alpha:E100_S750', 'alpha:E50_S150',
                'kcat_DS', 'kcat_DSS', 'dE:100', 'dE:50', 'dE:25']:
        assert key in map_sampling.keys(), f"Please provide {key} in map_file."

    alpha_list = {}
    for key in map_sampling.keys():
        if key.startswith('alpha'):
            alpha_list[key] = map_sampling[key]

    E_list = {}
    for key in ['dE:100', 'dE:50', 'dE:25']:
        E_list[key] = map_sampling[key]

    [map_index, map_params, log_probs] = map_finding(trace, expts, prior_infor=prior_infor_update,
                                                     alpha_list=alpha_list, E_list=E_list, 
                                                     set_lognormal_dE=args.set_lognormal_dE, dE=args.dE)

with open(os.path.join(args.out_dir, "map.txt"), "w") as f:
    print("MAP index:" + str(map_index), file=f)
    print("\nKinetics parameters:", file=f)
    for key in trace.keys():
        print(key, ': %.3f' %trace[key][map_index], file=f)

pickle.dump(log_probs, open(os.path.join(args.out_dir, 'log_probs.pickle'), "wb"))

map_values = {}
for key in trace.keys():
    map_values[key] = trace[key][map_index]
pickle.dump(map_values, open(os.path.join(args.out_dir, 'map.pickle'), "wb"))

## Fitting plot
params_logK, params_kcat = extract_params_from_map_and_prior(trace, map_index, prior_infor_update)

for n in range(len(expts)):
    if args.set_K_S_DS_equal_K_S_D:
        try: params_logK[f'logK_S_DS:{n}'] = params_logK[f'logK_S_D:{n}']
        except: params_logK['logK_S_DS'] = params_logK['logK_S_D']
    if args.set_K_S_DI_equal_K_S_DS:
        try: params_logK[f'logK_S_DI:{n}'] = params_logK[f'logK_S_DS:{n}']
        except: params_logK['logK_S_DI'] = params_logK['logK_S_DS']

n = 0
plot_data_conc_log(expts_plot_no_I, extract_logK_n_idx(params_logK, n, shared_params),
                   extract_kcat_n_idx(params_kcat, n, shared_params),
                   line_colors=['black', 'red', 'tab:brown'], ls='-.',
                   E_list=E_list, plot_legend=False,
                   OUTFILE=os.path.join(args.out_dir, 'ES'))

for i in range(len(inhibitor_name)):
    n = i + 1
    start = i*no_expt[n]+no_expt[0]
    end   = i*no_expt[n]+no_expt[0]+no_expt[n]    
    plot_data_conc_log(expts_plot[start:end], extract_logK_n_idx(params_logK, n, shared_params),
                       extract_kcat_n_idx(params_kcat, n, shared_params),
                       alpha_list=alpha_list, E_list=E_list, plot_legend=False,
                       OUTFILE=os.path.join(args.out_dir, 'ESI_'+str(i)))
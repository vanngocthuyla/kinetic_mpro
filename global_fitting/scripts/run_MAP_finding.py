import warnings
import numpy as np
import sys
import os
import argparse

import pickle
import arviz as az

import jax
import jax.numpy as jnp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _MAP_finding import _map_finding

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_file",             type=str, 				default="")
parser.add_argument( "--evaluated_file",        type=str,               default="")
parser.add_argument( "--data_dir",              type=str, 				default="")
# parser.add_argument( "--nsamples",              type=str, 				default="")

args = parser.parse_args()

os.chdir(args.data_dir)
trace = pickle.load(open(args.mcmc_file, "rb"))
data = az.convert_to_inference_data(trace)

df = pd.read_csv(args.evaluated_file, header=0)
df_2 = dict([(name, np.array(df[name])) for name in df.columns])
# for sigma_name in ['log_sigma_rate', 'log_sigma_auc', 'log_sigma_ice']: 
#     df_2[sigma_name] = trace[sigma_name][:len(df_2['logKd'])]
df_2['log_sigma_rate'] = np.repeat(np.log(np.std(experiments_combine[0]['v'])/np.sqrt(len(experiments_combine[0]['v']))), len(df_2['logKd']))
df_2['log_sigma_auc'] = np.repeat(np.log(np.std(experiments_combine[1]['M'])/np.sqrt(len(experiments_combine[1]['M']))), len(df_2['logKd']))
df_2['log_sigma_ice'] = np.repeat(np.log(np.std(experiments_combine[2]['Km_over_kcat'])/np.sqrt(len(experiments_combine[2]['Km_over_kcat']))), len(df_2['logKd']))

experiments = []
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


# Collate all kinetics experiments
kinetics_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'kinetics'])
kinetics_logStot = np.hstack([experiment['logStot'] for experiment in experiments if experiment['type'] == 'kinetics'])
kinetics_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'kinetics'])
v = np.hstack([experiment['v'] for experiment in experiments if experiment['type'] == 'kinetics'])

experiments_combined = []
# Fig 3a red
experiments_combined.append({'type':'kinetics',
                             'logMtot': kinetics_logMtot,
                             'logStot': kinetics_logStot,
                             'logItot': kinetics_logItot,
                             'v': v
                             })

experiments_combined.append({'type':'kinetics',
                             'figure':'3a',
                             'logMtot': np.log(np.array([6, 10, 25, 50, 75, 100, 120])*1E-6), # M
                             'logStot': np.array([np.log(200E-6)]*7), # 5 uL of 5mM substrate diluted with 95 uL reaction mixture,
                             'logItot': None, #np.array([np.log(1E-20)]*7), 
                             'v': np.array([0.0195, 0.024, 0.081, 0.36, 0.722, 1.13, 1.64])*1E-6, # M min^{-1}
                             'x':'logMtot'})

experiments_combined.append({'type':'AUC',
                             'figure':'6b',
                             'logMtot': np.array([np.log(7E-6)]*8), # 6-7 micromolar
                             'logStot': None, #np.array([np.log(1E-20)]*8),
                             'logItot': np.log(np.array([1, 3, 6, 10, 12.5, 15, 20, 50])*1E-6), 
                             'M': np.array([7, 6, 5, 3.25, 2.75, 2.55, 1.85, 0.8])*1E-6, # M min^{-1}
                             'x':'logItot'})

# Fig 6d
experiments_combined.append({'type':'catalytic_efficiency',
                             'figure':'6d',
                             'logMtot':np.array([np.log(10E-6)]*5),
                             'logStot': np.array([np.log(1E-20)]*5), # Not used
                             'logItot':np.log((np.array([0, 23.3, 56.6, 90, 123]) + 10)*1E-6),
                             'Km_over_kcat':np.array([2170, 7640, 17800, 29700, 36600])/10.,
                             'x':'logItot'})

# [map_params, map_index] = _map_uniform(trace, experiments_combined, nsamples=n_samples)

prior_information = []

prior_information.append({'type':'logK', 'name': 'logKd', 'dist': 'normal', 'loc': -2.914, 'scale': 1.702})
prior_information.append({'type':'logK', 'name': 'logK_S_M', 'dist': 'normal', 'loc': -4.236, 'scale': 2.133})
prior_information.append({'type':'logK', 'name': 'logK_S_D', 'dist': 'normal', 'loc': -5.957, 'scale': 3.557})
prior_information.append({'type':'logK', 'name': 'logK_S_DS', 'dist': 'normal', 'loc': -13.743, 'scale': 3.354})
prior_information.append({'type':'logK', 'name': 'logK_I_M', 'dist': 'normal', 'loc': -4.46, 'scale': 2.596})
prior_information.append({'type':'logK', 'name': 'logK_I_D', 'dist': 'normal', 'loc': -14.486, 'scale': 2.043})
prior_information.append({'type':'logK', 'name': 'logK_I_DI', 'dist': 'normal', 'loc': -18.330, 'scale': 1.909})
prior_information.append({'type':'logK', 'name': 'logK_S_DI', 'dist': 'normal', 'loc': -14.044, 'scale': 1.930})

prior_information.append({'type':'kcat', 'name': 'kcat_MS', 'dist': 'uniform', 'lower': 0, 'upper': 1})
prior_information.append({'type':'kcat', 'name': 'kcat_DS', 'dist': 'uniform', 'lower': 0, 'upper': 1})
prior_information.append({'type':'kcat', 'name': 'kcat_DSS', 'dist': 'normal', 'loc': 0.515, 'scale': 0.148})
prior_information.append({'type':'kcat', 'name': 'kcat_DSI', 'dist': 'normal', 'loc': 0.632, 'scale': 0.208})

# Finding MAP
[log_probs, map_params, map_index] = _map_hill(mcmc_trace=trace, 
                                               experiments=experiments_combine, 
                                               params_to_evaluate=df_2, 
                                               params_dist=prior_information)

with open("output.txt", "w") as f:
  	print("MAP index:" + str(map_index), file=f)
  	print("Kinetics parameters: \n" + str(map_params), file=f)
import os
import argparse
from glob import glob
import numpy as np
import jax
import jax.numpy as jnp

import pickle

from jax.config import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--map_file",                      type=str,               default="map.pickle")

args = parser.parse_args()

expts_dir = glob(os.path.join(args.mcmc_dir , "*"), recursive = True)
expts_name = [os.path.basename(f) for f in expts_dir if os.path.isdir(f)]
expts_name.sort()

combined_map = {}
for n, name in enumerate(expts_name):
    if os.path.isdir(os.path.join(args.mcmc_dir, name)):
        map = pickle.load(open(os.path.join(args.mcmc_dir, name, args.map_file), "rb"))
        for key in ['logK_I_M:1', 'logK_I_D:1', 'logK_I_DI:1', 'logK_S_DI:1']:
            if key in map.keys():
                combined_map[key[:-1]+str(n+1)] = map[key]
        # for key in ['log_sigma:ESI:0', 'log_sigma:ESI:1', 'log_sigma:ESI:2', 'log_sigma:ESI:3']:
        for key in [f'log_sigma:{name}:0', f'log_sigma:{name}:1', f'log_sigma:{name}:2', f'log_sigma:{name}:3']:
            if key in map.keys():
                combined_map['log_sigma:'+name+':'+key[-1]] = map[key]

combined_map['log_sigma:ES:0'] = map['log_sigma:ES:0']
combined_map['log_sigma:ES'] = map['log_sigma:ES:0']

for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
            'alpha:E100_S1350', 'alpha:E100_S50', 'alpha:E100_S750', 'alpha:E50_S150',
            'kcat_DS', 'kcat_DSS', 'dE:100', 'dE:50', 'dE:25']:
    if key in map.keys():
        combined_map[key] = map[key]

os.chdir(args.out_dir)
pickle.dump(combined_map, open('map_sampling.pickle', "wb"))

with open("map_sampling.txt", "w") as f:
    for key in combined_map.keys():
        print(key, ': %.3f' %combined_map[key], file=f)

np.set_printoptions(precision=2, suppress=True)

#Shared parameters
params_logK = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS']
params_alpha = ['alpha:E100_S1350', 'alpha:E100_S50', 'alpha:E100_S750', 'alpha:E50_S150']
params_kcat = ['kcat_DS', 'kcat_DSS']
params_dE = ['dE:100', 'dE:50', 'dE:25']
params_list = params_logK + params_alpha + params_kcat + params_dE

with open("params_infor.txt", "w") as f:
    for key in params_list:
        if key in params_logK:
            min_trace = -30
            max_trace = 0
        if key in params_kcat:
            min_trace = 0
            max_trace = 50.
        if key in params_dE:
            min_trace = 0
            max_trace = 200.
        if key in params_alpha:
            min_trace = 0.
            max_trace = 2.
        for name in expts_name:
            trace = pickle.load(open(os.path.join(args.mcmc_dir, name, 'traces.pickle'), "rb"))
            if key in trace.keys():
                if min(trace[key])>min_trace:
                    min_trace=min(trace[key])
                if max(trace[key])<max_trace:
                    max_trace = max(trace[key])
        print(key, ': %.3f' %min_trace, '; %.3f' %max_trace, file=f)

    print("\nBroader prior:", file=f)
    for key in params_list:
        if key in params_logK:
            min_trace = 0
            max_trace = -30
        if key in params_kcat:
            min_trace = 50
            max_trace = 0.
        if key in params_dE:
            min_trace = 200
            max_trace = 0.
        if key in params_alpha:
            min_trace = 2.
            max_trace = 0.
        for name in expts_name:
            trace = pickle.load(open(os.path.join(args.mcmc_dir, name, 'traces.pickle'), "rb"))
            if key in trace.keys():
                if min(trace[key])<min_trace:
                    min_trace=min(trace[key])
                if max(trace[key])>max_trace:
                    max_trace = max(trace[key])
        print(key, ': %.3f' %min_trace, '; %.3f' %max_trace, file=f)

    for key in ['logK_I_M:1', 'logK_I_D:1', 'logK_I_DI:1', 'logK_S_DI:1', 'kcat_DSI:1']:
        min_trace = []
        max_trace = []
        for name in expts_name:
            trace = pickle.load(open(os.path.join(args.mcmc_dir, name, 'traces.pickle'), "rb"))
            if key in trace.keys():
                min_trace.append(min(trace[key]))
                max_trace.append(max(trace[key]))
        print('\n', key, file=f)
        print(np.array2string(np.array(min_trace), separator=","), file=f)
        print(np.array2string(np.array(max_trace), separator=","), file=f)
        # print('\n', 'min:', min(min_trace), '; max:', max(max_trace))
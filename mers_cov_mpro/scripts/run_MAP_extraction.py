<<<<<<< HEAD
import os
import argparse
from glob import glob
import numpy as np
import jax
import jax.numpy as jnp

import pickle
import pandas as pd

from jax.config import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()

parser.add_argument( "--data_file",                     type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--map_file",                      type=str,               default="map.pickle")

args = parser.parse_args()

np.set_printoptions(precision=2, suppress=True)

df_mers = pd.read_csv(args.data_file)
inhibitor_name = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

global_map = pickle.load(open(args.map_file, "rb"))

for n in range(len(inhibitor_name)):
    name = inhibitor_name[n][7:12]
    local_map = {}
    if not os.path.isdir(os.path.join(args.out_dir, name)):
        os.mkdir(os.path.join(args.out_dir, name))

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'alpha:E100_S1350', 'alpha:E100_S50', 'alpha:E100_S750', 'alpha:E50_S150',
                'kcat_DS', 'kcat_DSS', 'dE:100', 'dE:50', 'dE:25']:
        if key in global_map.keys():
            local_map[key] = global_map[key]

    for key in ['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI', 'kcat_DSI']:
        if f'{key}:{n+1}' in global_map.keys():
            local_map[key] = global_map[f'{key}:{n+1}']

    for key in [f'log_sigma:{name}:0', f'log_sigma:{name}:1', f'log_sigma:{name}:2', f'log_sigma:{name}:3']:
        if key in global_map.keys():
            local_map[key] = global_map[key]

    pickle.dump(local_map, open(os.path.join(args.out_dir, name, 'map_sampling.pickle'), "wb"))

    with open(os.path.join(args.out_dir, name, "map_sampling.txt"), "w") as f:
        for key in local_map.keys():
=======
import os
import argparse
from glob import glob
import numpy as np
import jax
import jax.numpy as jnp

import pickle
import pandas as pd

from jax.config import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()

parser.add_argument( "--data_file",                     type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--map_file",                      type=str,               default="map.pickle")

args = parser.parse_args()

np.set_printoptions(precision=2, suppress=True)

df_mers = pd.read_csv(args.data_file)
inhibitor_name = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])

global_map = pickle.load(open(args.map_file, "rb"))

for n in range(len(inhibitor_name)):
    name = inhibitor_name[n][7:12]
    local_map = {}
    if not os.path.isdir(os.path.join(args.out_dir, name)):
        os.mkdir(os.path.join(args.out_dir, name))

    for key in ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS',
                'alpha:ESI:0', 'alpha:ESI:1', 'alpha:ESI:2', 'alpha:ESI:3',
                'kcat_DS', 'kcat_DSS', 'dE:100', 'dE:50', 'dE:25']:
        if key in global_map.keys():
            local_map[key] = global_map[key]

    for key in ['logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI', 'kcat_DSI']:
        if f'{key}:{n+1}' in global_map.keys():
            local_map[key] = global_map[f'{key}:{n+1}']

    for key in [f'log_sigma:{name}:0', f'log_sigma:{name}:1', f'log_sigma:{name}:2', f'log_sigma:{name}:3']:
        if key in global_map.keys():
            local_map[key] = global_map[key]

    pickle.dump(local_map, open(os.path.join(args.out_dir, name, 'map_sampling.pickle'), "wb"))

    with open(os.path.join(args.out_dir, name, "map_sampling.txt"), "w") as f:
        for key in local_map.keys():
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
            print(key, ': %.3f' %local_map[key], file=f)
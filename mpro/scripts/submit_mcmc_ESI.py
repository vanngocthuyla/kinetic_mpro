"""
This file is used to submit the batch file and run the code in one of the following files:
- run_mcmc_EI.py
- run_mcmc_ES.py
- run_mcmc_ESI_2_curves.py
- run_mcmc_ESI_4_curves.py
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="")
parser.add_argument( "--prior_infor",                   type=str,               default="")
parser.add_argument( "--shared_params_infor",           type=str,               default="")
parser.add_argument( "--map_dir",                       type=str,               default="")
parser.add_argument( "--map_name",                      type=str,               default="map.pickle")
parser.add_argument( "--map_file",                      type=str,               default="")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--running_script",                type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

args = parser.parse_args()

if args.fit_E_S: 
    fit_E_S = " --fit_E_S "
else:
    fit_E_S = " "

if args.fit_E_I: 
    fit_E_I = " --fit_E_I "
else:
    fit_E_I = " "

if args.multi_var: 
    multi_var = " --multi_var "
else:
    multi_var = " "

if args.multi_alpha: 
    multi_alpha = " --multi_alpha "
else:
    multi_alpha = " "

if args.set_lognormal_dE: 
    set_lognormal_dE = " --set_lognormal_dE "
else:
    set_lognormal_dE = " "

name_inhibitors = args.name_inhibitor.split()
if len(name_inhibitors) == 0:
    df_mers = pd.read_csv(args.input_file)
    inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])
else:
    inhibitor_list = name_inhibitors

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

for n, inhibitor in enumerate(inhibitor_list):

    inhibitor_dir = inhibitor[7:12]
    inhibitor_name = inhibitor[:12]

    if len(args.prior_infor)>0:
        prior_infor = " --prior_infor " + args.prior_infor
    else:
        prior_infor = " "

    if len(args.shared_params_infor)>0:
        shared_params = " --shared_params_infor " + args.shared_params_infor
    else:
        shared_params = " "

    if len(args.map_file)>0: 
        map_file = " --initial_values " + args.map_file
    elif len(args.map_dir)>0:
        map_file = " --initial_values " + os.path.join(args.map_dir, inhibitor_dir, args.map_name)
    else:
        map_file = " "

    if len(args.last_run_dir)>0:
        last_run_dir = " --last_run_dir " + os.path.join(args.last_run_dir, inhibitor_dir)
    else:
        last_run_dir = " "

    if not os.path.exists(os.path.join(args.out_dir, inhibitor_dir)):
        os.makedirs(os.path.join(args.out_dir, inhibitor_dir))

    qsub_file = os.path.join(args.out_dir, inhibitor_dir, inhibitor_dir+".job")
    log_file  = os.path.join(args.out_dir, inhibitor_dir, inhibitor_dir+".log")
    out_dir = os.path.join(args.out_dir, inhibitor_dir)

    qsub_script = '''#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -o %s '''%log_file + '''

module load anaconda3/2022.10
conda activate mpro
cd ''' + out_dir + '''\n''' + \
    '''date\n''' + \
    '''python ''' + args.running_script + \
    ''' --input_file ''' + args.input_file + prior_infor + shared_params + \
    map_file + last_run_dir + ''' --out_dir ''' + out_dir + \
    ''' --name_inhibitor ''' + inhibitor_name + \
    fit_E_S + fit_E_I + multi_var + multi_alpha + \
    set_lognormal_dE + ''' --dE %0.5f '''%args.dE + \
    ''' --niters %d '''%args.niters + \
    ''' --nburn %d '''%args.nburn + \
    ''' --nthin %d '''%args.nthin + \
    ''' --nchain %d '''%args.nchain + \
    ''' --random_key %d '''%args.random_key + \
    '''\ndate \n''' 

    print("Submitting " + qsub_file)
    open(qsub_file, "w").write(qsub_script)
    # os.system("sbatch %s"%qsub_file)
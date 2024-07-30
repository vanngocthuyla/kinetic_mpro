"""
This file is used to submit the batch file and run the code in run_CRC_fitting_pIC50_estimating.py
A list of experiments can be provided, then separated into multiple batch file to run them in parallel.
"""

import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser() 

parser.add_argument( "--running_script",                type=str,               default="")

parser.add_argument( "--prior_infor",                   type=str,               default="")
parser.add_argument( "--shared_params_infor",           type=str,               default="")
parser.add_argument( "--initial_values",                type=str,               default="")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--fit_mutant_kinetics",           action="store_true",    default=False)
parser.add_argument( "--fit_mutant_AUC",                action="store_true",    default=False)
parser.add_argument( "--fit_mutant_ICE",                action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Nashed",           action="store_true",    default=False)
parser.add_argument( "--fit_wildtype_Vuong",            action="store_true",    default=False)

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var_mut",                 action="store_true",    default=False)
parser.add_argument( "--multi_var_wt",                  action="store_true",    default=False)

parser.add_argument( "--set_K_I_M_equal_K_S_M",         action="store_true",    default=False)
parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSS_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DSS",   action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=1)

args = parser.parse_args()

file_name = 'sars_2'

if len(args.prior_infor)>0:
    prior_infor = " --prior_infor " + args.prior_infor
else:
    prior_infor = ""
if len(args.shared_params_infor)>0:
    shared_params = " --shared_params_infor " + args.shared_params_infor
else:
    shared_params = ""
if len(args.initial_values)>0: 
    initial_values = " --initial_values " +args.initial_values
else:
    initial_values = ""
if len(args.last_run_dir)>0:
    last_run_dir = " --last_run_dir " + args.last_run_dir
else:
    last_run_dir = ""

if args.fit_mutant_kinetics: 
    fit_mutant_kinetics = " --fit_mutant_kinetics "
else:
    fit_mutant_kinetics = ""
if args.fit_mutant_AUC: 
    fit_mutant_AUC = " --fit_mutant_AUC "
else:
    fit_mutant_AUC = ""
if args.fit_mutant_ICE:
    fit_mutant_ICE = " --fit_mutant_ICE "
else:
    fit_mutant_ICE = ""
if args.fit_wildtype_Nashed:
    fit_wildtype_Nashed = " --fit_wildtype_Nashed "
else:
    fit_wildtype_Nashed = ""
if args.fit_wildtype_Vuong: 
    fit_wildtype_Vuong = " --fit_wildtype_Vuong "
else:
    fit_wildtype_Vuong = ""

if args.fit_E_S: 
    fit_E_S = " --fit_E_S "
else:
    fit_E_S = ""
if args.fit_E_I: 
    fit_E_I = " --fit_E_I "
else:
    fit_E_I = ""

if args.multi_var_mut: 
    multi_var_mut = " --multi_var_mut "
else:
    multi_var_mut = ""
if args.multi_var_wt: 
    multi_var_wt = " --multi_var_wt "
else:
    multi_var_wt = ""

if args.set_K_I_M_equal_K_S_M:
    set_K_I_M_equal_K_S_M = " --set_K_I_M_equal_K_S_M "
else:
    set_K_I_M_equal_K_S_M = ""
if args.set_K_S_DS_equal_K_S_D:
    set_K_S_DS_equal_K_S_D = " --set_K_S_DS_equal_K_S_D "
else:
    set_K_S_DS_equal_K_S_D = ""
if args.set_K_S_DI_equal_K_S_DS:
    set_K_S_DI_equal_K_S_DS = " --set_K_S_DI_equal_K_S_DS "
else:
    set_K_S_DI_equal_K_S_DS = ""
if args.set_kcat_DSS_equal_kcat_DS:
    set_kcat_DSS_equal_kcat_DS = " --set_kcat_DSS_equal_kcat_DS "
else:
    set_kcat_DSS_equal_kcat_DS = ""
if args.set_kcat_DSI_equal_kcat_DS:
    set_kcat_DSI_equal_kcat_DS = " --set_kcat_DSI_equal_kcat_DS "
else:
    set_kcat_DSI_equal_kcat_DS = ""
if args.set_kcat_DSI_equal_kcat_DSS:
    set_kcat_DSI_equal_kcat_DSS = " --set_kcat_DSI_equal_kcat_DSS "
else:
    set_kcat_DSI_equal_kcat_DSS = ""

qsub_file = os.path.join(args.out_dir, file_name+".job")
log_file  = os.path.join(args.out_dir, file_name+".log")

qsub_script = '''#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -o %s '''%log_file + '''

module load anaconda3/2022.10
conda activate mpro
cd ''' + args.out_dir + '''\n''' + \
    '''date\n''' + \
    '''python ''' + args.running_script + \
    prior_infor + shared_params + initial_values + last_run_dir + \
    ''' --out_dir ''' + args.out_dir + \
    fit_mutant_kinetics + fit_mutant_AUC + fit_mutant_ICE + fit_wildtype_Nashed + fit_wildtype_Vuong + \
    fit_E_S + fit_E_I + multi_var_mut + multi_var_wt + \
    set_K_I_M_equal_K_S_M + set_K_I_M_equal_K_S_M + set_K_S_DS_equal_K_S_D + \
    set_kcat_DSS_equal_kcat_DS + set_kcat_DSI_equal_kcat_DS + set_kcat_DSI_equal_kcat_DSS + \
    ''' --niters %d '''%args.niters + \
    ''' --nburn %d '''%args.nburn + \
    ''' --nthin %d '''%args.nthin + \
    ''' --nchain %d '''%args.nchain + \
    ''' --random_key %d '''%args.random_key + \
    '''\ndate \n''' 

print("Submitting " + qsub_file)
open(qsub_file, "w").write(qsub_script)
# os.system("sbatch %s"%qsub_file)
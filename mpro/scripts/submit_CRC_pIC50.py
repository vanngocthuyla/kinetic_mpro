import sys
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
parser.add_argument( "--map_file",                      type=str,               default="")
parser.add_argument( "--last_run_dir",                  type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--running_script",                type=str,               default="")

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_lognormal_dE",              action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=0.1)

parser.add_argument( "--set_K_S_DS_equal_K_S_D",        action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)

parser.add_argument( "--split_by",                      type=int,               default=8)
parser.add_argument( "--exclude_experiments",           type=str,               default="")

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

parser.add_argument( "--outlier_removal",               action="store_true",    default=False)
parser.add_argument( "--exclude_first_trace",           action="store_true",    default=False)
parser.add_argument( "--key_to_check",                  type=str,               default="")
parser.add_argument( "--converged_samples",             type=int,               default=500)

parser.add_argument( "--enzyme_conc_nM",                type=int,               default="100")
parser.add_argument( "--inhibitor_conc_nM",             type=int,               default="1350")

args = parser.parse_args()

if len(args.prior_infor)>0:
    prior_infor = " --prior_infor " + args.prior_infor
else:
    prior_infor = " "

if len(args.shared_params_infor)>0:
    shared_params = " --shared_params_infor " + args.shared_params_infor
else:
    shared_params = " "

if len(args.map_file)>0: 
    map_file = " --initial_values " +args.map_file
else:
    map_file = " "

if len(args.last_run_dir)>0:
    last_run_dir = " --last_run_dir " + args.last_run_dir
else:
    last_run_dir = " "

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

if args.set_K_S_DS_equal_K_S_D:
    set_K_S_DS_equal_K_S_D = " --set_K_S_DS_equal_K_S_D "
else:
    set_K_S_DS_equal_K_S_D = " "

if args.set_K_S_DI_equal_K_S_DS:
    set_K_S_DI_equal_K_S_DS = " --set_K_S_DI_equal_K_S_DS "
else:
    set_K_S_DI_equal_K_S_DS = " "

if args.outlier_removal:
    outlier_removal = " --outlier_removal "
else:
    outlier_removal = " "

if args.exclude_first_trace:
    exclude_first_trace = " --exclude_first_trace "
else:
    exclude_first_trace = " "

if args.key_to_check:
    key_to_check = " --key_to_check " + args.key_to_check
else:
    key_to_check = ""

name_inhibitors = args.name_inhibitor.split()
if len(name_inhibitors) == 0:
    df_mers = pd.read_csv(args.input_file)
    _inhibitor_list = np.unique(df_mers[df_mers['Inhibitor (nM)']>0.0]['Inhibitor_ID'])
else:
    _inhibitor_list = name_inhibitors

exclude_experiments = args.exclude_experiments.split()
inhibitor_list = [name for name in _inhibitor_list if name[:12] not in exclude_experiments]
inhibitor_list = np.sort(np.array(inhibitor_list))

if args.split_by != 0:
    inhibitor_multi_list = np.array_split(np.array(inhibitor_list), args.split_by)
else: 
    inhibitor_multi_list = inhibitor_list

for list_idx, _inhibitor_list in enumerate(inhibitor_multi_list):

    if not os.path.isdir(os.path.join(args.out_dir, 'running_files')):
        os.mkdir(os.path.join(args.out_dir, 'running_files'))

    qsub_file = os.path.join(args.out_dir, 'running_files', f"{list_idx}.sh")
    log_file = os.path.join(args.out_dir, 'running_files', f"{list_idx}.log")

    qsub_script = '''#!/bin/bash
conda activate mpro ''' + '''\ncd ''' + args.out_dir + '''\n\n(('''
    open(qsub_file, "w").write(qsub_script)

    for n, inhibitor in enumerate(_inhibitor_list):

        inhibitor_dir = inhibitor[7:12]
        inhibitor_name = inhibitor[:12]

        qsub_script = '''date \n''' + \
        '''python ''' + args.running_script + \
        ''' --name_inhibitor ''' + inhibitor_name + \
        ''' --input_file ''' + args.input_file + prior_infor + shared_params + \
        last_run_dir + map_file + ''' --out_dir ''' + args.out_dir + \
        multi_var + multi_alpha + set_lognormal_dE + ''' --dE %0.5f '''%args.dE + \
        set_K_S_DS_equal_K_S_D + set_K_S_DI_equal_K_S_DS + \
        ''' --niters %d '''%args.niters + \
        ''' --nburn %d '''%args.nburn + \
        ''' --nthin %d '''%args.nthin + \
        ''' --nchain %d '''%args.nchain + \
        ''' --random_key %d '''%args.random_key + \
        outlier_removal + exclude_first_trace + key_to_check + \
        ''' --converged_samples %d '''%args.converged_samples +\
        ''' --enzyme_conc_nM %d '''%args.enzyme_conc_nM + \
        ''' --inhibitor_conc_nM %d '''%args.inhibitor_conc_nM + \
        '''\n\n'''

        open(qsub_file, "a").write(qsub_script)

    qsub_script = ''') 2>&1) | tee ''' + log_file
    open(qsub_file, "a").write(qsub_script)
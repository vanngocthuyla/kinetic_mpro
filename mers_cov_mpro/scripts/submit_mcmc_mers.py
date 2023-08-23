import sys
import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str, 				default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=True)

parser.add_argument( "--set_K_I_M_equal_K_S_M",         action="store_true",    default=False)
parser.add_argument( "--set_K_S_DI_equal_K_S_DS",       action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSS_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DS",    action="store_true",    default=False)
parser.add_argument( "--set_kcat_DSI_equal_kcat_DSS",   action="store_true",    default=False)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

args = parser.parse_args()

running_script = "/ocean/projects/mcb160011p/sophie92/python/mers_mpro/scripts/run_mcmc_ES.py"

if args.fit_E_S and args.fit_E_I:
    file_name = "mers"
elif args.fit_E_S: 
    file_name = "mers_E_S"
elif args.fit_E_I: 
    file_name = "mers_E_I"

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

if args.set_K_I_M_equal_K_S_M: 
    set_K_I_M_equal_K_S_M = " --set_K_I_M_equal_K_S_M "
else:
    set_K_I_M_equal_K_S_M = " "

if args.set_K_S_DI_equal_K_S_DS:
    set_K_S_DI_equal_K_S_DS = " --set_K_S_DI_equal_K_S_DS "
else:
    set_K_S_DI_equal_K_S_DS = " "

if args.set_kcat_DSS_equal_kcat_DS:
    set_kcat_DSS_equal_kcat_DS = " --set_kcat_DSS_equal_kcat_DS "
else:
    set_kcat_DSS_equal_kcat_DS = " "

if args.set_kcat_DSI_equal_kcat_DS:
    set_kcat_DSI_equal_kcat_DS = " --set_kcat_DSI_equal_kcat_DS "
else:
    set_kcat_DSI_equal_kcat_DS = " "

if args.set_kcat_DSI_equal_kcat_DSS:
    set_kcat_DSI_equal_kcat_DSS = " --set_kcat_DSI_equal_kcat_DSS "
else:
    set_kcat_DSI_equal_kcat_DSS = " "

qsub_file = os.path.join(args.out_dir, file_name+".job")
log_file  = os.path.join(args.out_dir, file_name+".log")

qsub_script = '''#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -o %s '''%log_file + '''

module load anaconda3/2022.10
conda activate mpro
cd ''' + args.out_dir + '''\n''' + \
    '''date\n''' + \
    '''python ''' + running_script + \
    ''' --input_file ''' + args.input_file + \
    ''' --out_dir ''' + args.out_dir + \
    multi_var + fit_E_S + fit_E_I + \
    set_K_I_M_equal_K_S_M + set_K_S_DI_equal_K_S_DS + \
    set_kcat_DSS_equal_kcat_DS+ set_kcat_DSI_equal_kcat_DS + set_kcat_DSI_equal_kcat_DSS + \
    ''' --niters %d '''%args.niters + \
    ''' --nburn %d '''%args.nburn + \
    ''' --nthin %d '''%args.nthin + \
    ''' --nchain %d '''%args.nchain + \
    ''' --random_key %d '''%args.random_key + \
    '''\ndate \n''' 

print("Submitting " + qsub_file)
open(qsub_file, "w").write(qsub_script)
os.system("sbatch %s"%qsub_file)
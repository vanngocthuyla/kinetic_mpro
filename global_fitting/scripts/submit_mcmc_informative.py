import sys
import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument( "--out_dir",               type=str, 				default="")
parser.add_argument( "--file_name",             type=str,               default="")
parser.add_argument( "--kinetics_fitting",      action="store_true",    default=True)
parser.add_argument( "--auc_fitting",           action="store_true",    default=True)
parser.add_argument( "--ice_fitting",           action="store_true",    default=True)

parser.add_argument( "--niters",				type=int, 				default=10000)
parser.add_argument( "--nburn",                 type=int, 				default=2000)
parser.add_argument( "--nthin",                 type=int, 				default=1)
parser.add_argument( "--nchain",                type=int, 				default=4)
parser.add_argument( "--random_key",            type=int, 				default=0)

args = parser.parse_args()

running_script = "/home/exouser/python/mpro/scripts/run_mcmc_normal.py"

file_name = args.file_name

# if args.kinetics_fitting:
#     kinetics_fitting = ''' --kinetics_fitting '''
#     file_name = file_name + "_rate"
# else:
#     kinetics_fitting = ''' '''

# if args.auc_fitting:
#     auc_fitting = ''' --auc_fitting '''
#     file_name = file_name + "_auc"
# else:
#     auc_fitting = ''' '''

# if args.ice_fitting:
#     ice_fitting = ''' --ice_fitting '''
#     file_name = file_name + "_ice"
# else:
#     ice_fitting = ''' '''

qsub_file = os.path.join(args.out_dir, file_name+".job")
log_file  = os.path.join(args.out_dir, file_name+".log")

qsub_script = '''#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node=128
#SBATCH -o %s '''%log_file + '''

conda activate mpro
cd ''' + args.out_dir + '''\n''' + \
    '''date\n''' + \
    '''python ''' + running_script + \
    ''' --out_dir ''' + args.out_dir + \
    ''' --niters %d '''%args.niters + \
    ''' --nburn %d '''%args.nburn + \
    ''' --nthin %d '''%args.nthin + \
    ''' --nchain %d '''%args.nchain + \
    ''' --random_key %d '''%args.random_key + \
    '''\ndate \n''' 

print("Submitting " + qsub_file)
open(qsub_file, "w").write(qsub_script)
os.system("sbatch %s"%qsub_file)
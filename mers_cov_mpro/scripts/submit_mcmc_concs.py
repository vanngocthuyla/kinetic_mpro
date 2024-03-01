import sys
import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument( "--input_file",                    type=str,               default="")
parser.add_argument( "--out_dir",                       type=str, 				default="")
parser.add_argument( "--map_file",                      type=str,               default="")
parser.add_argument( "--name_inhibitor",                type=str,               default="Inhibitor")
parser.add_argument( "--running_script",                type=str,               default="")

parser.add_argument( "--fit_E_S",                       action="store_true",    default=False)
parser.add_argument( "--fit_E_I",                       action="store_true",    default=False)

parser.add_argument( "--multi_var",                     action="store_true",    default=False)
parser.add_argument( "--multi_alpha",                   action="store_true",    default=False)
parser.add_argument( "--set_error_E",                   action="store_true",    default=False)
parser.add_argument( "--dE",                            type=float,             default=1.0)

parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nburn",                         type=int,               default=2000)
parser.add_argument( "--nthin",                         type=int,               default=1)
parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--random_key",                    type=int,               default=0)

args = parser.parse_args()

if len(args.map_file)>0: 
    map_file = " --map_file " + args.map_file
else:
    map_file = " "

file_name = args.name_inhibitor[8:]

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

if args.set_error_E: 
    set_error_E = " --set_error_E "
else:
    set_error_E = " "


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
    '''python ''' + args.running_script + \
    ''' --input_file ''' + args.input_file + \
    ''' --out_dir ''' + args.out_dir + map_file + \
    ''' --name_inhibitor ''' + args.name_inhibitor + \
    fit_E_S + fit_E_I + multi_var + multi_alpha + \
    set_error_E + ''' --dE %0.5f '''%args.dE + \
    ''' --niters %d '''%args.niters + \
    ''' --nburn %d '''%args.nburn + \
    ''' --nthin %d '''%args.nthin + \
    ''' --nchain %d '''%args.nchain + \
    ''' --random_key %d '''%args.random_key + \
    '''\ndate \n''' 

print("Submitting " + qsub_file)
open(qsub_file, "w").write(qsub_script)
os.system("sbatch %s"%qsub_file)
import os
import argparse
import pickle

from _plotting import plotting_trace_global

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_file",                     type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--nchain",                        type=int,               default=4)

args = parser.parse_args()

trace = pickle.load(open(args.mcmc_file, "rb"))
os.mkdir(os.path.join(args.out_dir, 'Plotting'))
plotting_trace_global(trace=trace, out_dir=os.path.join(args.out_dir, 'Plotting'), nchain=args.nchain)
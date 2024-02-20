import os
import argparse
from glob import glob
import numpy as np

import pickle
from pymbar import timeseries

from _trace_analysis import _combining_multi_trace
from _plotting import plotting_trace

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--combined_trace_name",           type=str,               default="Combined_trace")

parser.add_argument( "--nchain",                        type=int,               default=4)

parser.add_argument( "--multi_expt",                    action="store_true",    default=False)
parser.add_argument( "--plotting",                      action="store_true",    default=False)

args = parser.parse_args()

mcmc_dir = glob(os.path.join(args.mcmc_dir, "*"), recursive = True)
mcmc_dir = [os.path.basename(f) for f in mcmc_dir if os.path.isdir(f)]
# prefix = os.path.commonprefix(_mcmc_dir)
assert len(mcmc_dir)>0, "Please provide at least one input folder."
mcmc_dir.sort()

if args.multi_expt:
    expt_list = []
    for _dir in mcmc_dir:
        _expt_dir = glob(os.path.join(args.mcmc_dir, _dir, "*"), recursive = True)
        _expt_dir = [os.path.basename(f) for f in _expt_dir if os.path.isdir(f)]
        expt_list.append(_expt_dir)
    expt_list = np.unique(np.concatenate(expt_list))
    expt_list.sort()

    assert len(expt_list)>0, "Please provide at least one experiment."

    for expt in expt_list:
        # Extracting all traces.pickles of one experiment from multiple sampling runs
        trace_files = [os.path.join(args.mcmc_dir, f, expt, "traces.pickle") for f in mcmc_dir if os.path.isfile(os.path.join(args.mcmc_dir, f, expt, "traces.pickle"))]

        if not os.path.exists(os.path.join(args.out_dir, expt)):
            os.mkdir(os.path.join(args.out_dir, expt))
            print("Create", os.path.join(args.out_dir, expt))

        print("Loading", trace_files)
        trace = _combining_multi_trace(trace_files, nchain=args.nchain,
                                       out_dir=os.path.join(args.out_dir, expt), combined_trace_name=args.combined_trace_name)
        if args.plotting: plotting_trace(trace, os.path.join(args.out_dir, expt))
else: 
    trace_files = [os.path.join(args.mcmc_dir, f, "traces.pickle") for f in mcmc_dir if os.path.isfile(os.path.join(args.mcmc_dir, f, "traces.pickle"))]

    print("Loading", trace_files)
    trace = _combining_multi_trace(trace_files, nchain=args.nchain,
                                   out_dir=args.out_dir, combined_trace_name=args.combined_trace_name)
    if args.plotting: plotting_trace(trace, args.out_dir)
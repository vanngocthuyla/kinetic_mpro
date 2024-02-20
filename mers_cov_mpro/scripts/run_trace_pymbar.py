import os
import argparse
from glob import glob
import numpy as np

import pickle
from pymbar import timeseries

from _trace_analysis import _trace_convergence
from _plotting import plotting_trace

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_dir",                      type=str,               default="")
parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--converged_trace_name",          type=str,               default="Converged_trace")

parser.add_argument( "--nchain",                        type=int,               default=4)
parser.add_argument( "--niters",                        type=int,               default=10000)
parser.add_argument( "--nskip",                         type=int,               default=100)

parser.add_argument( "--multi_expt",                    action="store_true",    default=False)
parser.add_argument( "--exclude_first_trace",           action="store_true",    default=False)
parser.add_argument( "--plotting",                      action="store_true",    default=False)

parser.add_argument( "--key_to_check",                  type=str,               default="")

args = parser.parse_args()

mcmc_dir = glob(os.path.join(args.mcmc_dir, "sampling_*"), recursive = True)
mcmc_dir = [os.path.basename(f) for f in mcmc_dir if os.path.isdir(f)]
# prefix = os.path.commonprefix(_mcmc_dir)
assert len(mcmc_dir)>0, "Please provide at least one input folder."
mcmc_dir.sort()

unconverged_list = []
if args.multi_expt:

    expt_list = []
    for _dir in mcmc_dir:
        _expt_dir = glob(os.path.join(args.mcmc_dir, _dir, "*"), recursive = True)
        _expt_dir = [os.path.basename(f) for f in _expt_dir if os.path.isdir(f) and os.path.isfile(os.path.join(f, 'traces.pickle'))]
        expt_list.append(_expt_dir)
    expt_list = np.unique(np.concatenate(expt_list))
    expt_list.sort()

    assert len(expt_list)>0, "Please provide at least one experiment."

    for expt in expt_list:
        # Extracting all traces.pickles of one experiment from multiple sampling runs
        _trace_files = [os.path.join(args.mcmc_dir, f, expt, "traces.pickle") for f in mcmc_dir if os.path.isfile(os.path.join(args.mcmc_dir, f, expt, "traces.pickle"))]
        if args.exclude_first_trace and len(_trace_files)>1: 
            trace_files = _trace_files[1:]
        else:
            trace_files = _trace_files

        if not os.path.exists(os.path.join(args.out_dir, expt)):
            os.mkdir(os.path.join(args.out_dir, expt))
            # print("Create", os.path.join(args.out_dir, expt))

        print("Running", expt)
        [trace, flag, nchain_updated] = _trace_convergence(mcmc_files=trace_files, out_dir=os.path.join(args.out_dir, expt), 
                                                           nskip=args.nskip, nchain=args.nchain, expected_nsample=args.niters,
                                                           key_to_check=args.key_to_check.split(), converged_trace_name=args.converged_trace_name)
        if args.plotting: plotting_trace(trace, os.path.join(args.out_dir, expt), nchain_updated)
        if not flag: unconverged_list.append('ASAP-00'+expt)
else:
    trace_files = [os.path.join(args.mcmc_dir, f, "traces.pickle") for f in mcmc_dir if os.path.isfile(os.path.join(args.mcmc_dir, f, "traces.pickle"))]
    [trace, flag, nchain_updated] = _trace_convergence(mcmc_files=trace_files, out_dir=os.path.join(args.out_dir, expt), 
                                                       nskip=args.nskip, nchain=args.nchain, expected_nsample=args.niters,
                                                       key_to_check=args.key_to_check.split(), converged_trace_name=args.converged_trace_name)
    if args.plotting: plotting_trace(trace, args.out_dir, nchain_updated)
    # if not flag: unconverged_list.append('ASAP-00'+expt)

if len(unconverged_list)>0: 
    mes = "Unconverged experiments:" + unconverged_list
    print(mes)
    with open("Unconverged.txt", "w") as f:
        print(mes, file=f)
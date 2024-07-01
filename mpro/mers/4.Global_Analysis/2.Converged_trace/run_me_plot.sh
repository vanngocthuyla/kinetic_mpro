#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_trace_plot_global.py"

export MCMC_FILE="/home/exouser/python/mpro/mers/4.Global_Analysis/2.Converged_trace/Converged_trace.pickle"

export OUT_DIR="/home/exouser/python/mpro/mers/4.Global_Analysis/2.Converged_trace"

export N_CHAIN=4

python $SCRIPT --mcmc_file $MCMC_FILE --out_dir $OUT_DIR --nchain $N_CHAIN
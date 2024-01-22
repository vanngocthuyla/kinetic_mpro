#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_trace_plot_global.py"

export MCMC_FILE="/home/exouser/python/mers_mpro/4.Global_Analysis/1.Combined_trace/Combined_trace.pickle"

export OUT_DIR="/home/exouser/python/mers_mpro/4.Global_Analysis/1.Combined_trace"

export N_CHAIN=4

python $SCRIPT --mcmc_file $MCMC_FILE --out_dir $OUT_DIR --nchain $N_CHAIN

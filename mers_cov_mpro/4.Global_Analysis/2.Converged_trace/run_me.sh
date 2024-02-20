#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_trace_pymbar.py"

export MCMC_DIR="/home/exouser/python/mers_mpro/4.Global/"

export OUT_DIR="/home/exouser/python/mers_mpro/4.Global_Analysis/2.Converged_trace"

export N_CHAIN=4

python $SCRIPT --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --nchain $N_CHAIN

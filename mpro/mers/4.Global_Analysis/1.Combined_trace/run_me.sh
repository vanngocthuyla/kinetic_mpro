#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_trace_combining.py"

export MCMC_DIR="/home/exouser/python/mpro/mers/4.Global/"

export OUT_DIR="/home/exouser/python/mpro/mers/4.Global_Analysis/1.Combined_trace"

export N_CHAIN=4

python $SCRIPT --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --nchain $N_CHAIN

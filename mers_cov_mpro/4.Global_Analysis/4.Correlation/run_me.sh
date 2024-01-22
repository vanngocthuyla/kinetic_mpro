#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_analysis_global.py"

export INPUT_FILE="/home/exouser/python/mers_mpro/input/ESI_Full.csv"

export MCMC_FILE="/home/exouser/python/mers_mpro/4.Global_Analysis/2.Converged_trace/Converged_trace.pickle"

export MCMC_DIR='/home/exouser/python/mers_mpro/4.Global_Analysis/3.MAP_finding'

export OUT_DIR="/home/exouser/python/mers_mpro/4.Global_Analysis/4.Correlation"

python $SCRIPT --kinetic_file $INPUT_FILE --mcmc_file $MCMC_FILE --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --multi_var --multi_alpha --set_lognormal_dE --dE 0.1
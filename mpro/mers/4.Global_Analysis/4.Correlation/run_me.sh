#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_analysis_global.py"

export INPUT_FILE="/home/exouser/python/mpro/mers/input/ESI_Full.csv"

export MCMC_FILE="/home/exouser/python/mpro/mers/4.Global_Analysis/2.Converged_trace/Converged_trace.pickle"

export MAP_DIR='/home/exouser/python/mpro/mers/4.Global_Analysis/3.MAP_finding'

export OUT_DIR="/home/exouser/python/mpro/mers/4.Global_Analysis/4.Correlation"

python $SCRIPT --kinetic_file $INPUT_FILE --mcmc_file $MCMC_FILE --map_dir $MAP_DIR --out_dir $OUT_DIR --multi_var --set_lognormal_dE --dE 0.1
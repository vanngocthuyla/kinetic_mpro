#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_MAP_finding_mers.py"

export INPUT="/home/exouser/python/mers_mpro/input/ESI_Full.csv"

export MCMC_FILE="/home/exouser/python/mers_mpro/4.Global_Analysis/2.Converged_trace/Converged_trace.pickle"

export OUT_DIR="/home/exouser/python/mers_mpro/4.Global_Analysis/3.MAP_finding"

python $SCRIPT --kinetic_file $INPUT --mcmc_file $MCMC_FILE --out_dir $OUT_DIR --fit_E_S --fit_E_I --multi_var --multi_alpha --set_lognormal_dE --dE 0.1 --global_fitting
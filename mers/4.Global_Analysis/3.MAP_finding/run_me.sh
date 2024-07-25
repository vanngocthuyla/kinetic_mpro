#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_MAP_finding.py"

export INPUT="/home/exouser/python/mpro/mers/input/ESI_Full.csv"

export MCMC_FILE="/home/exouser/python/mpro/mers/4.Global_Analysis/2.Converged_trace/Converged_trace.pickle"

export OUT_DIR="/home/exouser/python/mpro/mers/4.Global_Analysis/3.MAP_finding"

export PRIOR='/home/exouser/python/mpro/mers/4.Global/Prior.json'

export INIT='/home/exouser/python/mpro/mers/4.Global/sampling_1/map_sampling.pickle'

python $SCRIPT --input_file $INPUT --mcmc_file $MCMC_FILE --out_dir $OUT_DIR --prior_infor $PRIOR --initial_values $INIT --fit_E_S --fit_E_I --multi_var --set_lognormal_dE --dE 0.1
#!/bin/bash

FILE=$(cd ../../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_MAP_finding.py'

export INPUT='/kinetic_mpro/mers/input/ESI_Full.csv'

export MCMC_FILE='/kinetic_mpro/mers/4.Global_Analysis/2.Converged_trace/Converged_trace.pickle'

export OUT_DIR='/kinetic_mpro/mers/4.Global_Analysis/3.MAP_finding'

export PRIOR='/kinetic_mpro/mers/4.Global/Prior.json'

export INIT='/kinetic_mpro/mers/4.Global/sampling_1/map_sampling.pickle'

python $DIR$SCRIPT --input_file $DIR$INPUT --mcmc_file $DIR$MCMC_FILE --out_dir $DIR$OUT_DIR --prior_infor $DIR$PRIOR --initial_values $DIR$INIT --fit_E_S --fit_E_I --multi_var --set_lognormal_dE --dE 0.1
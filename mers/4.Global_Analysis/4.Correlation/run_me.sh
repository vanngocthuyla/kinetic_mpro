#!/bin/bash

FILE=$(cd ../../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_analysis_global.py'

export INPUT_FILE='/kinetic_mpro/mers/input/ESI_Full.csv'

export MCMC_FILE='/kinetic_mpro/mers/4.Global_Analysis/2.Converged_trace/Converged_trace.pickle'

export MAP_DIR='/kinetic_mpro/mers/4.Global_Analysis/3.MAP_finding'

export OUT_DIR='/kinetic_mpro/mers/4.Global_Analysis/4.Correlation'

python $DIR$SCRIPT --kinetic_file $DIR$INPUT_FILE --mcmc_file $DIR$MCMC_FILE --map_dir $DIR$MAP_DIR --out_dir $DIR$OUT_DIR --multi_var --set_lognormal_dE --dE 0.1
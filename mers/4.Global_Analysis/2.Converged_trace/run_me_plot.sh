#!/bin/bash

FILE=$(cd ../../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_trace_plot_global.py'

export MCMC_FILE='/kinetic_mpro/mers/4.Global_Analysis/2.Converged_trace/Converged_trace.pickle'

export OUT_DIR='/kinetic_mpro/mers/4.Global_Analysis/2.Converged_trace'

export N_CHAIN=4

python $DIR$SCRIPT --mcmc_file $DIR$MCMC_FILE --out_dir $DIR$OUT_DIR --nchain $N_CHAIN
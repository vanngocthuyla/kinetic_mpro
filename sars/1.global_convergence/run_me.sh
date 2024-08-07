#!/bin/bash

export DIR='/home/exouser/python'

export SCRIPT='/kinetic_mpro/scripts/run_convergence_percent_estimation.py'

export SCRIPT_PLOT='/kinetic_mpro/scripts/run_convergence_percent_plot.py'

export MCMC_FILE='/kinetic_mpro/sars/1.global_analysis/traces_log10.pickle'

export OUT_DIR_1='/kinetic_mpro/sars/1.global_convergence/Convergence'

export OUT_DIR_2='/kinetic_mpro/sars/1.global_convergence/Plot'

python $DIR$SCRIPT --mcmc_file $DIR$MCMC_FILE --out_dir $DIR$OUT_DIR_1

python $DIR$SCRIPT_PLOT --mcmc_file $DIR$MCMC_FILE --data_dir $DIR$OUT_DIR_1 --out_dir $DIR$OUT_DIR_2

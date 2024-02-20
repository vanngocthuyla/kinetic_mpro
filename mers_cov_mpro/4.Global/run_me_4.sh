#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/submit_mcmc_global.py"

export RUNNING_SCRIPT="/home/exouser/python/mers_mpro/scripts/run_mcmc_global.py"

export INPUT="/home/exouser/python/mers_mpro/input/ESI_mean_drop.csv"

export OUT_DIR="/home/exouser/python/mers_mpro/4.Global/sampling_4"

export LAST_RUN_DIR="/home/exouser/python/mers_mpro/4.Global/sampling_3"

export MAP_FILE="/home/exouser/python/mers_mpro/4.Global/sampling_1/map_sampling.pickle"

export N_INTER=500

export N_BURN=0

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --last_run_dir $LAST_RUN_DIR --map_file $MAP_FILE --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.1 --multi_var --multi_alpha
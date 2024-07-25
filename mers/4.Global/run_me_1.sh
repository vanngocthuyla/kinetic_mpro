#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/submit_mcmc_global.py"

export RUNNING_SCRIPT="/home/exouser/python/mpro/scripts/run_mcmc_global.py"

export INPUT="/home/exouser/python/mpro/mers/input/ESI_mean_drop.csv"

export OUT_DIR="/home/exouser/python/mpro/mers/4.Global/sampling_1"

export MAP_FILE="/home/exouser/python/mpro/mers/4.Global/sampling_1/map_sampling.pickle"

export N_INTER=20

export N_BURN=10

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --map_file $MAP_FILE --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.1 --multi_var

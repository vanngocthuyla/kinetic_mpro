#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/submit_mcmc_ESI.py"

export INPUT="/home/exouser/python/mers_mpro/input/ESI_mean_drop.csv"

export OUT_DIR="/home/exouser/python/mers_mpro/2.ESI"

export MAP_FILE="/home/exouser/python/mers_mpro/1.ES/map.pickle"

export RUNNING_SCRIPT="/home/exouser/python/mers_mpro/scripts/run_mcmc_ESI_2_curves.py"

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --map_file $MAP_FILE --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.5 --multi_var
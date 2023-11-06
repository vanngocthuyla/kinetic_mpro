#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_mcmc_ESI_fixed_ES.py"

export INPUT="/home/exouser/python/mers_mpro/input/ESI_mean_drop.csv"

export OUT_DIR="/home/exouser/python/mers_mpro/5.ESI/8374"

export MAP_FILE="/home/exouser/python/mers_mpro/5.ESI/8374/map_sampling.pickle"

export NAME='ASAP-0008374'

export N_INTER=10000

export N_BURN=2000

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --name_inhibitor $NAME --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --map_file $MAP_FILE --set_error_E --multi_var --multi_alpha
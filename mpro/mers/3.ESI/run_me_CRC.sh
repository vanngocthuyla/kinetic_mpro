#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/submit_mcmc_ESI.py"

export RUNNING_SCRIPT="/home/exouser/python/mpro/scripts/run_mcmc_ESI_4_curves.py"

export INPUT="/home/exouser/python/mpro/mers/input/ESI_mean_CDD_20231205_normalized_data.csv"

export OUT_DIR="/home/exouser/python/mpro/mers/3.ESI"

export MAP_FILE="/home/exouser/python/mpro/mers/2.ESI/00773/map.pickle"

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

export NAME='ASAP-0008642 ASAP-0011124 ASAP-0013299'

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --name_inhibitor "$NAME" --running_script $RUNNING_SCRIPT --map_file $MAP_FILE --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.5 --multi_var
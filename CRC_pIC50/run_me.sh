#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/submit_CRC_pIC50.py"

export RUNNING_SCRIPT="/home/exouser/python/mpro/scripts/run_CRC_fitting_pIC50_estimating.py"

export INPUT="/home/exouser/python/mpro/CRC_pIC50/input/Input.csv"

export PRIOR='/home/exouser/python/mpro/CRC_pIC50/input/Prior.json'

export MAP_FILE="/home/exouser/python/mpro/CRC_pIC50/input/map_sampling.pickle"

export OUT_DIR="/home/exouser/python/mpro/CRC_pIC50/output"

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

export NAME='ID_14973'

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --prior_infor $PRIOR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.1 --multi_var --name_inhibitor "$NAME" --map_file $MAP_FILE --outlier_removal --exclude_first_trace
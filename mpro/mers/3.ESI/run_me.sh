#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/submit_mcmc_ESI.py"

export RUNNING_SCRIPT="/home/exouser/python/mpro/scripts/run_mcmc_ESI_4_curves.py"

export INPUT="/home/exouser/python/mpro/mers/input/ESI_mean_drop.csv"

export OUT_DIR="/home/exouser/python/mpro/mers/3.ESI"

export PRIOR='/home/exouser/python/mpro/mers/3.ESI/Prior.json'

export MAP_DIR="/home/exouser/python/mpro/mers/2.ESI"

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

export NAME='ASAP-0000153 ASAP-0000214'

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --map_dir $MAP_DIR --prior_infor $PRIOR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.5 --multi_var --name_inhibitor "$NAME"
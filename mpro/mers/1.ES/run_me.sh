#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_mcmc_ES.py"

export INPUT="/home/exouser/python/mpro/mers/input/ES_mean.csv"

export OUT_DIR="/home/exouser/python/mpro/mers/1.ES"

export PRIOR='/home/exouser/python/mpro/mers/1.ES/Prior.json'

export INIT='/home/exouser/python/mpro/mers/1.ES/map.pickle'

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --prior_infor $PRIOR --initial_values $INIT --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --set_lognormal_dE --dE 0.5

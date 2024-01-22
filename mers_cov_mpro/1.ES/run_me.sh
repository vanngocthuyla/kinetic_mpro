#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_mcmc_ES.py"

export INPUT="/home/exouser/python/mers_mpro/input/ES_mean.csv"

export OUT_DIR="/home/exouser/python/mers_mpro/1.ES"

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --set_lognormal_dE --dE 0.5

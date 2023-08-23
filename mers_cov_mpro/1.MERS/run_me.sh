#!/bin/bash

export SCRIPT="/ocean/projects/mcb160011p/sophie92/python/mers_mpro/scripts/submit_mcmc_mers.py"

export INPUT="/ocean/projects/mcb160011p/sophie92/python/mers_mpro/input/ASAP-0000153-001_2.csv"

export OUT_DIR="/ocean/projects/mcb160011p/sophie92/python/mers_mpro/1.MERS"

export N_INTER=10000

export N_BURN=2000

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --multi_var
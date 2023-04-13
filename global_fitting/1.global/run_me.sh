#!/bin/bash

export SCRIPT="/home/vla/python/mpro/scripts/submit_mcmc_global.py"

export OUT_DIR="/home/vla/python/mpro/1.global"

export N_INTER=10000

export N_BURN=2000

export N_CHAIN=4

python $SCRIPT --out_dir $OUT_DIR --file_name 'mpro' --auc_fitting --ice_fitting --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN

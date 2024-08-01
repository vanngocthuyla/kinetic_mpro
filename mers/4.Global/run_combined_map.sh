#!/bin/bash

FILE=$(cd ../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_MAP_combining.py'

export MCMC_DIR='/kinetic_mpro/mers/3.ESI'

export OUT_DIR='/kinetic_mpro/mers/4.Global/sampling_1'

python $DIR$SCRIPT --mcmc_dir $DIR$MCMC_DIR --out_dir $DIR$OUT_DIR 
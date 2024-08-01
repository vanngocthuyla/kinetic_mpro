#!/bin/bash

FILE=$(cd ../../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_trace_combining.py'

export MCMC_DIR='/kinetic_mpro/mers/4.Global/'

export OUT_DIR='/kinetic_mpro/mers/4.Global_Analysis/1.Combined_trace'

export N_CHAIN=4

python $DIR$SCRIPT --mcmc_dir $DIR$MCMC_DIR --out_dir $DIR$OUT_DIR --nchain $N_CHAIN

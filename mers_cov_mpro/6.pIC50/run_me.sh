#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_pIC50_fixed_ES.py"

export INHIBITOR_FILE="/home/exouser/python/mers_mpro/input/ESI_mean_drop.csv"

export MCMC_DIR="/home/exouser/python/mers_mpro/5.Convergence"

export OUT_DIR="/home/exouser/python/mers_mpro/6.pIC50"

export LOGK_FILE='/home/exouser/python/mers_mpro/5.ESI/sampling_1/map_sampling.pickle'

python $SCRIPT --inhibitor_file $INHIBITOR_FILE --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --logK_dE_alpha $LOGK_FILE

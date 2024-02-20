#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_pIC50_find_concs.py"

export INHIBITOR_FILE="/home/exouser/python/mers_mpro/input/ESI_mean_drop.csv"

export MCMC_DIR="/home/exouser/python/mers_mpro/5.Convergence"

export OUT_DIR="/home/exouser/python/mers_mpro/6.pIC50_find_concs"

export LOGK_FILE='/home/exouser/python/mers_mpro/5.ESI/sampling_1/map_sampling.pickle'

export CELLULAR="/home/exouser/python/mers_mpro/input/Cellular_IC50_filtered.csv"

python $SCRIPT --inhibitor_file $INHIBITOR_FILE --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --logK_dE_alpha_file $LOGK_FILE --cellular_pIC50_file $CELLULAR
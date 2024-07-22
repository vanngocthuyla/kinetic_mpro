#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_pIC50_find_concs.py"

export INHIBITOR_FILE="/home/exouser/python/mpro/mers/input/CDD_20240406_normalized_data.csv"

export MCMC_DIR="/home/exouser/python/mpro/mers/7.CRC_outlier/Convergence"

export OUT_DIR="/home/exouser/python/mpro/mers/8.find_concs"

export LOGK_FILE="/home/exouser/python/mpro/mers/7.CRC_outlier/map_sampling.pickle"

export CELLULAR="/home/exouser/python/mpro/mers/input/20240305_Mavda_TMPRSS2_48.csv"

export EXCLUDE=''

python $SCRIPT --inhibitor_file $INHIBITOR_FILE --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --exclude_experiments "$EXCLUDE" --logK_dE_alpha_file $LOGK_FILE --cellular_pIC50_file $CELLULAR
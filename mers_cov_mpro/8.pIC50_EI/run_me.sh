#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_pIC50.py"

export INHIBITOR_FILE="/home/exouser/python/mers_mpro/input/CDD_20240222_normalized_data.csv"

export MCMC_DIR="/home/exouser/python/mers_mpro/7.CRC_outlier/Convergence"

export OUT_DIR="/home/exouser/python/mers_mpro/8.pIC50_EI"

export LOGK_FILE="/home/exouser/python/mers_mpro/7.CRC_outlier/map_sampling.pickle"

export EXCLUDE='ASAP-0015637'

python $SCRIPT --inhibitor_file $INHIBITOR_FILE --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --exclude_experiments "$EXCLUDE" --logK_dE_alpha_file $LOGK_FILE
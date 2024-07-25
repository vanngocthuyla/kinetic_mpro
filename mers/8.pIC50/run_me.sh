#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_pIC50.py"

export INHIBITOR_FILE="/home/exouser/python/mpro/mers/input/CDD_20240406_normalized_data.csv"

export MCMC_DIR="/home/exouser/python/mpro/mers/7.CRC_outlier/Convergence"

export OUT_DIR="/home/exouser/python/mpro/mers/8.pIC50"

export LOGK_FILE="/home/exouser/python/mpro/mers/7.CRC_outlier/map_sampling.pickle"

export EXCLUDE='ASAP-0015637'

export ENZ_CONC_nM=0.142

export SUB_CONC_nM=141.235

python $SCRIPT --inhibitor_file $INHIBITOR_FILE --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --exclude_experiments "$EXCLUDE" --logK_dE_alpha_file $LOGK_FILE --enzyme_conc_nM $ENZ_CONC_nM --substrate_conc_nM $SUB_CONC_nM

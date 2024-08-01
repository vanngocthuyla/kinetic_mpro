#!/bin/bash

FILE=$(cd ../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_pIC50_find_concs.py'

export INHIBITOR_FILE='/kinetic_mpro/mers/input/CDD_20240406_normalized_data.csv'

export MCMC_DIR='/kinetic_mpro/mers/7.CRC_outlier/Convergence'

export OUT_DIR='/kinetic_mpro/mers/8.find_concs'

export LOGK_FILE='/kinetic_mpro/mers/7.CRC_outlier/map_sampling.pickle'

export CELLULAR='/kinetic_mpro/mers/input/20240305_Mavda_TMPRSS2_48.csv'

export EXCLUDE=''

python $DIR$SCRIPT --inhibitor_file $DIR$INHIBITOR_FILE --mcmc_dir $DIR$MCMC_DIR --out_dir $DIR$OUT_DIR --exclude_experiments "$EXCLUDE" --logK_dE_alpha_file $DIR$LOGK_FILE --cellular_pIC50_file $DIR$CELLULAR
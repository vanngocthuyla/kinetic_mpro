#!/bin/bash

FILE=$(cd ../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_pIC50.py'

export INHIBITOR_FILE='/kinetic_mpro/mers/input/CDD_20240406_normalized_data.csv'

export MCMC_DIR='/kinetic_mpro/mers/7.CRC_outlier/Convergence'

export OUT_DIR='/kinetic_mpro/mers/8.pIC50'

export LOGK_FILE='/kinetic_mpro/mers/7.CRC_outlier/map_sampling.pickle'

export EXCLUDE=''

export ENZ_CONC_nM=0.142

export SUB_CONC_nM=141.254

python $DIR$SCRIPT --inhibitor_file $DIR$INHIBITOR_FILE --mcmc_dir $DIR$MCMC_DIR --out_dir $DIR$OUT_DIR --exclude_experiments "$EXCLUDE" --logK_dE_alpha_file $DIR$LOGK_FILE --enzyme_conc_nM $ENZ_CONC_nM --substrate_conc_nM $SUB_CONC_nM

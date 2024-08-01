#!/bin/bash

FILE=$(cd ../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/submit_CRC_mers.py'

export RUNNING_SCRIPT='/kinetic_mpro/scripts/run_CRC_fitting_pIC50_estimating.py'

export INPUT='/kinetic_mpro/mers/input/CDD_20240406_normalized_data.csv'

export PRIOR='/kinetic_mpro/mers/7.CRC_outlier/Prior.json'

export MAP_FILE='/kinetic_mpro/mers/7.CRC_outlier/map_sampling.pickle'

export OUT_DIR='/kinetic_mpro/mers/7.CRC_outlier'

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

export SPLIT_BY=8

export NAME=''

python $DIR$SCRIPT --input_file $DIR$INPUT --out_dir $DIR$OUT_DIR --running_script $DIR$RUNNING_SCRIPT  --prior_infor $DIR$PRIOR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --set_lognormal_dE --dE 0.1 --multi_var --name_inhibitor "$NAME" --map_file $DIR$MAP_FILE --split_by $SPLIT_BY --outlier_removal --exclude_first_trace
#!/bin/bash

export DIR='/home/exouser/python'

export SCRIPT='/kinetic_mpro/scripts/run_CRC_fitting_pIC50_estimating.py'

export INPUT='/kinetic_mpro/CRC_pIC50/input/Input.csv'

export PRIOR='/kinetic_mpro/CRC_pIC50/input/Prior.json'

export MAP_FILE='/kinetic_mpro/CRC_pIC50/input/map_sampling.pickle'

export OUT_DIR='/kinetic_mpro/CRC_pIC50/output'

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

export NAME='ID_14973'

python $DIR$SCRIPT --input_file $DIR$INPUT --out_dir $DIR$OUT_DIR --prior_infor $DIR$PRIOR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.1 --multi_var --name_inhibitor "$NAME" --initial_values $DIR$MAP_FILE --outlier_removal --exclude_first_trace
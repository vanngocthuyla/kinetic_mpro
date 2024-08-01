#!/bin/bash

FILE=$(cd ../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/submit_mcmc_ESI.py'

export RUNNING_SCRIPT='/kinetic_mpro/scripts/run_mcmc_EI.py'

export INPUT='/kinetic_mpro/mers/input/ESI_mean_drop.csv'

export PRIOR='/kinetic_mpro/mers/5.EI/Prior.json'

export OUT_DIR='/kinetic_mpro/mers/5.EI/sampling_4'

export MAP_DIR='/kinetic_mpro/mers/5.EI/sampling_1'

export MAP_NAME='map_sampling.pickle'

export LAST_RUN_DIR='/kinetic_mpro/mers/5.EI/sampling_3'

export INHIBITOR_NAMES='ASAP-0000153'

export N_INTER=2000

export N_BURN=200

export N_CHAIN=4

python $DIR$SCRIPT --input_file $DIR$INPUT --out_dir $DIR$OUT_DIR --running_script $DIR$RUNNING_SCRIPT --prior_infor $DIR$PRIOR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --dE 0 --multi_var --map_dir $DIR$MAP_DIR --map_name $MAP_NAME --last_run_dir $DIR$LAST_RUN_DIR --name_inhibitor "$INHIBITOR_NAMES"

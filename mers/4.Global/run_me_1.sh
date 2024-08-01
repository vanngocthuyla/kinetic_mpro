#!/bin/bash

FILE=$(cd ../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/submit_mcmc_global.py'

export RUNNING_SCRIPT='/kinetic_mpro/scripts/run_mcmc_global.py'

export INPUT='/kinetic_mpro/mers/input/ESI_mean_drop.csv'

export OUT_DIR='/kinetic_mpro/mers/4.Global/sampling_1'

export MAP_FILE='/kinetic_mpro/mers/4.Global/sampling_1/map_sampling.pickle'

export N_INTER=20

export N_BURN=10

export N_CHAIN=4

python $DIR$SCRIPT --input_file $DIR$INPUT --out_dir $DIR$OUT_DIR --running_script $DIR$RUNNING_SCRIPT --map_file $DIR$MAP_FILE --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.1 --multi_var

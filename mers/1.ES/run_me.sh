#!/bin/bash

FILE=$(cd ../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_mcmc_ES.py'

export INPUT='/kinetic_mpro/mers/input/ES_mean.csv'

export OUT_DIR='/kinetic_mpro/mers/1.ES'

export PRIOR='/kinetic_mpro/mers/1.ES/Prior.json'

export INIT='/kinetic_mpro/mers/1.ES/map.pickle'

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

python $DIR$SCRIPT --input_file $DIR$INPUT --out_dir $DIR$OUT_DIR --prior_infor $DIR$PRIOR --initial_values $DIR$INIT --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --set_lognormal_dE --dE 0.5

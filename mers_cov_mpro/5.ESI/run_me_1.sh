#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/submit_mcmc_ESI.py"

export INPUT="/home/exouser/python/mers_mpro/input/ESI_mean_drop.csv"

export OUT_DIR="/home/exouser/python/mers_mpro/5.ESI/sampling_1"

<<<<<<< HEAD
export MAP_DIR="/home/exouser/python/mers_mpro/3.ESI"
=======
export MAP_DIR="/home/exouser/python/mers_mpro/5.ESI/sampling_1"

export MAP_NAME="map_sampling.pickle"
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0

export RUNNING_SCRIPT="/home/exouser/python/mers_mpro/scripts/run_mcmc_ESI_fixed_ES.py"

export N_INTER=1000

<<<<<<< HEAD
export N_BURN=200

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --set_lognormal_dE --dE 0.5 --multi_var --multi_alpha --map_dir $MAP_DIR
=======
export N_BURN=1000

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --dE 0 --multi_var --multi_alpha --map_dir $MAP_DIR --map_name $MAP_NAME
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0

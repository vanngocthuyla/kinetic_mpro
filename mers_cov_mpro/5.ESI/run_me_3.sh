#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/submit_mcmc_ESI.py"

export INPUT="/home/exouser/python/mers_mpro/input/ESI_mean_drop.csv"

export OUT_DIR="/home/exouser/python/mers_mpro/5.ESI/sampling_3"

export MAP_DIR="/home/exouser/python/mers_mpro/5.ESI/sampling_2"

export RUNNING_SCRIPT="/home/exouser/python/mers_mpro/scripts/run_mcmc_ESI_fixed_ES.py"

export INHIBITOR_NAMES="ASAP-0000272 ASAP-0000733 ASAP-0008420"

export N_INTER=10000

export N_BURN=2000

export N_CHAIN=4

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --name_inhibitor "$INHIBITOR_NAMES" --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --last_run_dir $MAP_DIR --set_lognormal_dE --dE 0.05 --multi_var --multi_alpha

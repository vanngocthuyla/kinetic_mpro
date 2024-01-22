#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_analysis_fixed_ES.py"

export INPUT_FILE="/home/exouser/python/mers_mpro/input/ESI_Full.csv"

export MCMC_DIR="/home/exouser/python/mers_mpro/5.Convergence"

export OUT_DIR="/home/exouser/python/mers_mpro/5.Convergence_Analysis"

export E_ALPHA='/home/exouser/python/mers_mpro/5.ESI/sampling_1/map_sampling.pickle'

python $SCRIPT --kinetic_file $INPUT_FILE --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --dE_alpha_file $E_ALPHA
#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_MAP_combining.py"

export MCMC_DIR="/home/exouser/python/mers_mpro/3.ESI"

export OUT_DIR="/home/exouser/python/mers_mpro/4.Global/sampling_1"

python $SCRIPT --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR 
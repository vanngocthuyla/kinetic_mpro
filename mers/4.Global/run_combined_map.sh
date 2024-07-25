#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_MAP_combining.py"

export MCMC_DIR="/home/exouser/python/mpro/mers/3.ESI"

export OUT_DIR="/home/exouser/python/mpro/mers/4.Global/sampling_1"

python $SCRIPT --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR 
#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_pIC50.py"

export MCMC_DIR="/home/exouser/python/mers_mpro/6.pIC50/input"

export OUT_DIR="/home/exouser/python/mers_mpro/6.pIC50"

python $SCRIPT --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR 

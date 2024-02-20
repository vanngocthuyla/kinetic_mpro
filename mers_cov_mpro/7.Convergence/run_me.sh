#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_trace_pymbar.py"

export MCMC_DIR="/home/exouser/python/mers_mpro/7.ESI/"

export OUT_DIR="/home/exouser/python/mers_mpro/7.Convergence/"

export N_INTER=500

export N_CHAIN=4

export KEY="logK_I_M logK_I_D logK_I_DI logK_S_DI kcat_DSI"

export TRACE_NAME='traces'

python $SCRIPT --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --niters $N_INTER --nchain $N_CHAIN --key_to_check "$KEY" --multi_expt --exclude_first_trace --converged_trace_name $TRACE_NAME

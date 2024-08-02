#!/bin/bash

export DIR='/home/exouser/python'

export SCRIPT='/kinetic_mpro/scripts/run_analysis_global_sars.py'

export MCMC_FILE='/kinetic_mpro/sars/1.global/traces.pickle'

export SETTING='/kinetic_mpro/sars/1.global/setting.pickle'

export OUT_DIR='/kinetic_mpro/sars/1.global_analysis'

python $DIR$SCRIPT --mcmc_file $DIR$MCMC_FILE --setting $DIR$SETTING --out_dir $DIR$OUT_DIR
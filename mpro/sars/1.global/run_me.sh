#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_mcmc_sars.py"

export OUT_DIR="/home/exouser/python/mpro/sars/1.global"

export PRIOR='/home/exouser/python/mpro/sars/1.global/Prior.json'

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

python $SCRIPT --out_dir $OUT_DIR --prior_infor $PRIOR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN  --fit_mutant_kinetics  --fit_mutant_AUC  --fit_mutant_ICE  --fit_wildtype_Nashed  --fit_E_S  --fit_E_I  --multi_var_wt
#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/run_mcmc_sars_CRC.py"

export OUT_DIR="/home/exouser/python/mpro/sars/2.global_CRC"

export PRIOR='/home/exouser/python/mpro/sars/2.global_CRC/Prior.json'

export SHARED_PARAMS='/home/exouser/python/mpro/sars/2.global_CRC/Shared_params.json'

export N_INTER=500

export N_BURN=20

export N_CHAIN=4

python $SCRIPT --out_dir $OUT_DIR --prior_infor $PRIOR --shared_params_infor $SHARED_PARAMS --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN  --fit_mutant_kinetics  --fit_mutant_AUC  --fit_mutant_ICE  --fit_wildtype_Nashed  --fit_E_S  --fit_E_I  --multi_var_wt --multi_var_CRC 

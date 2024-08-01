#!/bin/bash

export DIR='/home/exouser/python'

export SCRIPT='/kinetic_mpro/scripts/run_mcmc_sars_CRC.py'

export OUT_DIR='/kinetic_mpro/sars/2.global_CRC'

export PRIOR='/kinetic_mpro/sars/2.global_CRC/Prior.json'

export SHARED_PARAMS='/kinetic_mpro/sars/2.global_CRC/Shared_params.json'

export N_INTER=500

export N_BURN=20

export N_CHAIN=4

python $DIR$SCRIPT --out_dir $DIR$OUT_DIR --prior_infor $DIR$PRIOR --shared_params_infor $DIR$SHARED_PARAMS --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN  --fit_mutant_kinetics  --fit_mutant_AUC  --fit_mutant_ICE  --fit_wildtype_Nashed  --fit_E_S  --fit_E_I  --multi_var_wt --multi_var_CRC 

#!/bin/bash

export DIR='/ocean/projects/mcb160011p/sophie92/python'

export SCRIPT='/kinetic_mpro/scripts/run_mcmc_sars.py'

export OUT_DIR='/kinetic_mpro/sars/1.global'

export PRIOR='/kinetic_mpro/sars/1.global/Prior.json'

export N_INTER=10000

export N_BURN=2000

export N_CHAIN=4

python $DIR$SCRIPT --out_dir $DIR$OUT_DIR --prior_infor $DIR$PRIOR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN  --fit_mutant_kinetics  --fit_mutant_AUC  --fit_mutant_ICE  --fit_wildtype_Nashed  --fit_E_S  --fit_E_I  --multi_var_wt
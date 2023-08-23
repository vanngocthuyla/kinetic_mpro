#!/bin/bash

export SCRIPT="/ocean/projects/mcb160011p/sophie92/python/mpro/scripts/submit_mcmc_adjustable.py"

export OUT_DIR="/ocean/projects/mcb160011p/sophie92/python/mpro/1.adjustable"

export N_INTER=10000

export N_BURN=2000

export N_CHAIN=4

python $SCRIPT --out_dir $OUT_DIR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_mutant_kinetics --fit_mutant_AUC --fit_mutant_ICE --fit_wildtype_Nashed --fit_E_S --fit_E_I --multi_var_wt
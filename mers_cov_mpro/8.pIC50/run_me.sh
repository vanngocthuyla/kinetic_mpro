#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_pIC50_fixed_ES.py"

export INHIBITOR_FILE="/home/exouser/python/mers_mpro/input/CDD_20231205_normalized_data.csv"

export MCMC_DIR="/home/exouser/python/mers_mpro/7.Convergence"

export OUT_DIR="/home/exouser/python/mers_mpro/8.pIC50"

export LOGK_FILE="/home/exouser/python/mers_mpro/7.ESI/map_sampling.pickle"

export EXCLUDE='ASAP-0000733 ASAP-0000738 ASAP-0011059 ASAP-0011359 ASAP-0013262 ASAP-0013266 ASAP-0013301 ASAP-0013412 ASAP-0013414 ASAP-0013890 ASAP-0014753 ASAP-0014764 ASAP-0014770 ASAP-0000144 ASAP-0000184 ASAP-0000185 ASAP-0000207 ASAP-0000208 ASAP-0000209 ASAP-0000225 ASAP-0008258 ASAP-0000225 ASAP-0000523 ASAP-0000580 ASAP-0008258 ASAP-0008452 ASAP-0008642 ASAP-0008675 ASAP-0010743 ASAP-0011124 ASAP-0011133 ASAP-0011137 ASAP-0011138 ASAP-0011139 ASAP-0011140 ASAP-0011141 ASAP-0011347 ASAP-0013299 ASAP-0014750 ASAP-0014888 ASAP-0014925 ASAP-0015637 ASAP-0000370 ASAP-0013719 ASAP-0013720 ASAP-0013721 ASAP-0013722 ASAP-0013731 ASAP-0015637 ASAP-0000161 ASAP-0000164 ASAP-0008315 ASAP-0013249 ASAP-0014786 ASAP-0014925'

python $SCRIPT --inhibitor_file $INHIBITOR_FILE --mcmc_dir $MCMC_DIR --out_dir $OUT_DIR --exclude_experiments "$EXCLUDE" --logK_dE_alpha $LOGK_FILE
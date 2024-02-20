#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/submit_mcmc_ESI_multiple.py"

export INPUT="/home/exouser/python/mers_mpro/input/CDD_20231205_normalized_data.csv"

export OUT_DIR="/home/exouser/python/mers_mpro/7.ESI/sampling_1"

export MAP_FILE="/home/exouser/python/mers_mpro/7.ESI/map_sampling.pickle"

export RUNNING_SCRIPT="/home/exouser/python/mers_mpro/scripts/run_CRC_fitting_ESI.py"

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

export NAME='ASAP-0000144 ASAP-0000184 ASAP-0000185 ASAP-0000207 ASAP-0000208 ASAP-0000209 ASAP-0000225 ASAP-0008258 ASAP-0000225 ASAP-0000523 ASAP-0000580 ASAP-0008258'

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --fit_E_S --fit_E_I --dE 0 --multi_var --multi_alpha --name_inhibitor "$NAME" --map_file $MAP_FILE --split_by 2

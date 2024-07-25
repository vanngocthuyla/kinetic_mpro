#!/bin/bash

export SCRIPT="/home/exouser/python/mpro/scripts/submit_CRC_mers.py"

export RUNNING_SCRIPT="/home/exouser/python/mpro/scripts/run_CRC_fitting_pIC50_estimating.py"

export INPUT="/home/exouser/python/mpro/mers/input/CDD_20240406_normalized_data.csv"

export PRIOR='/home/exouser/python/mpro/mers/7.CRC_outlier/Prior.json'

export MAP_FILE="/home/exouser/python/mpro/mers/7.CRC_outlier/map_sampling.pickle"

export OUT_DIR="/home/exouser/python/mpro/mers/7.CRC_outlier"

export N_INTER=1000

export N_BURN=200

export N_CHAIN=4

export SPLIT_BY=8

export NAME='ASAP-0015679 ASAP-0015677 ASAP-0015672 ASAP-0015666 ASAP-0015663 ASAP-0015660 ASAP-0015657 ASAP-0015656 ASAP-0015652 ASAP-0015648 ASAP-0015638 ASAP-0015637 ASAP-0015517 ASAP-0015514 ASAP-0015512 ASAP-0015508 ASAP-0015376 ASAP-0015294 ASAP-0015242 ASAP-0015239 ASAP-0015229 ASAP-0015227 ASAP-0015225 ASAP-0015223 ASAP-0015108 ASAP-0015099 ASAP-0015096 ASAP-0015092 ASAP-0015085 ASAP-0014973'

python $SCRIPT --input_file $INPUT --out_dir $OUT_DIR --running_script $RUNNING_SCRIPT  --prior_infor $PRIOR --niters $N_INTER --nburn $N_BURN --nchain $N_CHAIN --set_lognormal_dE --dE 0.1 --multi_var --name_inhibitor "$NAME" --map_file $MAP_FILE --split_by $SPLIT_BY --outlier_removal --exclude_first_trace
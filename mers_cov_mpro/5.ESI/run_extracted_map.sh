#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_MAP_extraction.py"

export DATA_FILE="/home/exouser/python/mers_mpro/input/ESI_mean_drop.csv"

export MAP="/home/exouser/python/mers_mpro/4.Global_Analysis/3.MAP_finding/map.pickle"

export OUT_DIR="/home/exouser/python/mers_mpro/5.ESI/sampling_1"

python $SCRIPT --data_file $DATA_FILE --out_dir $OUT_DIR --map_file $MAP
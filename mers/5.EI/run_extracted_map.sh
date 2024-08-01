#!/bin/bash

FILE=$(cd ../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export SCRIPT='/kinetic_mpro/scripts/run_MAP_extraction.py'

export DATA_FILE='/kinetic_mpro/mers/input/ESI_mean_drop.csv'

export MAP='/kinetic_mpro/mers/4.Global_Analysis/3.MAP_finding/map.pickle'

export OUT_DIR='/kinetic_mpro/mers/5.EI/sampling_1'

python $DIR$SCRIPT --data_file $DIR$DATA_FILE --out_dir $DIR$OUT_DIR --map_file $DIR$MAP
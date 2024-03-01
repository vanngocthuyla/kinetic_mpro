#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_bf.py"

export DATA="/home/exouser/python/mers_mpro/BF/data.pickle"

export FIRST="/home/exouser/python/mers_mpro/BF/KSDS_KSD"

export SECOND="/home/exouser/python/mers_mpro/BF/both"

export OUTFILE='bf_KSDS_KSD-both.txt'

python $SCRIPT --data_file $DATA --first_model_dir $FIRST --second_model_dir $SECOND --out_file $OUTFILE
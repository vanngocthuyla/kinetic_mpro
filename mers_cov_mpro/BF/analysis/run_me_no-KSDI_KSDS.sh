#!/bin/bash

export SCRIPT="/home/exouser/python/mers_mpro/scripts/run_bf.py"

export DATA="/home/exouser/python/mers_mpro/BF/data.pickle"

export FIRST="/home/exouser/python/mers_mpro/BF/KSDI_KSDS"

export SECOND="/home/exouser/python/mers_mpro/BF/no_constraint"

export OUTFILE='bf_KSDI_KSDS.txt'

python $SCRIPT --data_file $DATA --first_model_dir $FIRST --second_model_dir $SECOND --out_file $OUTFILE
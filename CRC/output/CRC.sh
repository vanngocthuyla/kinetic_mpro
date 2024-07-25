#!/bin/bash
    conda activate mpro 
cd /home/exouser/python/mpro/CRC/output
 date 
((python /home/exouser/python/mpro/scripts/run_CRC_fitting.py --name_inhibitor ASAP-0014973 --input_file /home/exouser/python/mpro/CRC/input/CDD_20240222_normalized_data.csv --prior_infor /home/exouser/python/mpro/CRC/input/Prior.json  --fit_E_S  --fit_E_I  --initial_values /home/exouser/python/mpro/CRC/input/map_sampling.pickle --out_dir /home/exouser/python/mpro/CRC/output --multi_var   --set_lognormal_dE  --dE 0.10000    --niters 1000  --nburn 200  --nthin 1  --nchain 4  --random_key 0  --outlier_removal 

) 2>&1) | tee /home/exouser/python/mpro/CRC/output/CRC.log
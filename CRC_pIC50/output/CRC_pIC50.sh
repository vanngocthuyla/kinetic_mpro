#!/bin/bash
    conda activate mpro 
cd /home/exouser/python/mpro/CRC_pIC50/output
 date 
((python /home/exouser/python/mpro/scripts/run_CRC_fitting_pIC50_estimating.py --name_inhibitor ASAP-0014973 --input_file /home/exouser/python/mpro/CRC_pIC50/input/CDD_20240222_normalized_data.csv --prior_infor /home/exouser/python/mpro/CRC_pIC50/input/Prior.json  --fit_E_S  --fit_E_I  --initial_values /home/exouser/python/mpro/CRC_pIC50/input/map_sampling.pickle --out_dir /home/exouser/python/mpro/CRC_pIC50/output --multi_var   --set_lognormal_dE  --dE 0.10000    --niters 1000  --nburn 200  --nthin 1  --nchain 4  --random_key 0  --outlier_removal  --exclude_first_trace  --converged_samples 500  --enzyme_conc_nM 100  --substrate_conc_nM 1350 

) 2>&1) | tee /home/exouser/python/mpro/CRC_pIC50/output/CRC_pIC50.log
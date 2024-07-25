## Assumption:

All datasets are shared the same conditions to we can utilize the MAP of posterior distributions of the enzyme-substrate parameters from the global fitting of MERS-CoV MPro as fixed values and estimate the enzyme-inhibitor parameters, along with the enzyme concentration uncertainty and the normalization factor.

## Input

- The input data should be under .csv format and contain the information about the plate of experiment, the concentration of the species (nM), ID of the inhibitor, response of the CRC. The final column ("Drop") to let the code know if we keep or drop a specific data point. 

- *Prior.json*: this file provides information about the prior distribution of the parameters.

- *map_sampling.pickle*: this file is copied from the global fitting and contains the MAP of posterior distributions of the enzyme-substrate parameters.

## Run_me.sh

You can run the model fitting directly by: 

*python mpro/scripts/run_CRC_fitting.py --name_inhibitor $ID --input_file $INPUT --prior_infor mpro/CRC/input/Prior.json --fit_E_S  --fit_E_I --initial_values mpro/CRC/input/map_sampling.pickle --out_dir /mpro/CRC/output --multi_var  --set_lognormal_dE  --dE 0.10 --niters 1000 --nburn 200  --nchain 4 --outlier_removal*

Or you can submit job by adjusting the code in **mpro/scripts/submit_CRC.py**. There are some arguments for this model fitting: 

- **args.input_file**: specifies the input file, which should be a CSV file containing the kinetic data. In the last column, the value drop=1 indicates that the corresponding data points will be excluded from the fitting process.

- **args.name_inhibitor**: a list of the names of inhibitors. If this argument is left empty, the model will estimate parameters for all inhibitors found in the args.input_file.

- **args.prior_infor**: provides the prior information for parameters.

- **args.shared_params_infor**: provides the information for shared parameters among multiple variants of enzyme/substrate/inhibitor. This only can be used for global fitting of multiple CRCs. 

- **args.initial_values**: allows the use of parameter values from the maximum a posteriori (MAP) estimation from the previous step as initial values for the model fitting in the next step. This also contains the values of posterior distributions of the enzyme-substrate parameters.

- **args.last_run_dir**: if additional samples need to be collected, this parameter declares the directory of the previous run that contains the information of the last state.

- **args.out_dir**: specifies the directory where the results will be saved.

- **args.fit_E_S**: if set to *True*, the model will estimate all enzyme-substrate parameters.

- **args.fit_E_I**: if set to *True*, the model will estimate all enzyme-inhibitor parameters.

- **args.multi_var**: if set to *True*, each dataset will have a measurement error estimated from the model. Otherwise, datasets from the same experiment or plate will share the measurement error.

- **args.multi_alpha**: if set to *True*, each dataset will have a normalization factor estimated from the model. Otherwise, datasets from the same plate will share the normalization factor. The **args.multi_var** and **args.multi_alpha** are only useful when we fit more than one CRC. 

- **args.set_lognormal_dE**: If set to *True*, a lognormal prior will be assigned for the enzyme concentration uncertainty. Otherwise, a uniform prior will be used.

- **args.dE**: This parameter specifies the uncertainty for the enzyme concentration.

- **args.set_K_S_DS_equal_K_S_D**: sets two specified parameters equal to each other.

- **args.set_K_S_DI_equal_K_S_DS**: sets two specified parameters equal to each other.

- **args.niters**, **args.nburn**, **args.nthin**, **args.nchain**: These specify the number of MCMC samples, the number of burning samples, the thinning rate, and the number of chains for Bayesian regression, respectively.

- **args.random_key**: sets the random key for Bayesian regression.

- **args.outlier_removal**: if set to *True*, the code will check for any outliers in the CRC and remove them before parameter estimation.


For example, if we aim to estimate kinetic parameters for a multiple CRCs, we would set **args.fit_E_S** = *True* and **args.fit_E_I** = *True*. Each dataset will have a measurement error, so **args.multi_var** = *True*, while some datasets from the same plate will share the alpha, hence **args.multi_alpha** = *False*. For the enzyme concentration uncertainty, we will assign a lognormal prior with an uncertainty of 10%, thus **args.set_lognormal_dE** = *True* and **args.dE** = *0.1*.

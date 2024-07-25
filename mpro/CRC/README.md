# CRC Fitting

## Assumption

All datasets share the same conditions, allowing us to utilize the Maximum A Posteriori (MAP) of posterior distributions of the enzyme-substrate parameters from the global fitting of MERS-CoV MPro as fixed values. This enables the estimation of enzyme-inhibitor parameters, along with the enzyme concentration uncertainty and the normalization factor.

## Input

- The input data should be in .csv format and contain information about the experiment plate, the concentration of the species (nM), the ID of the inhibitor, and the response of the CRC. The final column ("Drop") indicates whether a specific data point should be kept or dropped.

- *Prior.json*: provides information about the prior distribution of the parameters.

- *map_sampling.pickle*: this file, copied from the global fitting, contains the MAP of posterior distributions of the enzyme-substrate parameters.

## Run_me.sh

You can run the model fitting directly by executing:

    python /mpro/scripts/run_CRC_fitting.py --name_inhibitor $ID --input_file $INPUT --prior_infor /mpro/CRC/input/Prior.json --fit_E_S  --fit_E_I --initial_values /mpro/CRC/input/map_sampling.pickle --out_dir /mpro/CRC/output --multi_var  --set_lognormal_dE  --dE 0.10 --niters 1000 --nburn 200  --nchain 4 --outlier_removal

Alternatively, you can submit a job by adjusting the code in /mpro/scripts/submit_CRC.py. The arguments for this model fitting are: 

- **args.input_file**: Specifies the input file, a CSV file containing the kinetic data. In the last column, the value *1* indicates that the corresponding data points will be excluded from the fitting process.

- **args.name_inhibitor**: A list of the names of inhibitors. If this argument is left empty, the model will estimate parameters for all inhibitors found in the **args.input_file**.

- **args.prior_infor**: Provides the prior information for parameters.

- **args.shared_params_infor**: Provides the information for shared parameters among multiple variants of enzyme/substrate/inhibitor. This can only be used for global fitting of multiple CRCs. 

- **args.initial_values**: Allows the use of parameter values from the MAP estimation from the previous step as initial values for the model fitting in the next step. This also contains the values of posterior distributions of the enzyme-substrate parameters.

- **args.last_run_dir**: If additional samples need to be collected, this parameter specifies the directory of the previous run that contains the information of the last state.

- **args.out_dir**: Specifies the directory where the results will be saved.

- **args.fit_E_S**: if set to *True*, the model will estimate all enzyme-substrate parameters.

- **args.fit_E_I**: if set to *True*, the model will estimate all enzyme-inhibitor parameters.

- **args.multi_var**: if set to *True*, each dataset will have a measurement error estimated from the model. Otherwise, datasets from the same experiment or plate will share the measurement error.

- **args.multi_alpha**: if set to *True*, each dataset will have a normalization factor estimated from the model. Otherwise, datasets from the same plate will share the normalization factor. The **args.multi_var** and **args.multi_alpha** options are only useful when fitting more than one CRC. 

- **args.set_lognormal_dE**: If set to *True*, a lognormal prior will be assigned for the enzyme concentration uncertainty. Otherwise, a uniform prior will be used.

- **args.dE**: Specifies the uncertainty for the enzyme concentration.

- **args.set_K_S_DS_equal_K_S_D**: Sets two specified parameters equal to each other.

- **args.set_K_S_DI_equal_K_S_DS**: Sets two specified parameters equal to each other.

- **args.niters**, **args.nburn**, **args.nthin**, **args.nchain**: Specify the number of MCMC samples, the number of burn-in samples, the thinning rate, and the number of chains for Bayesian regression, respectively.

- **args.random_key**: Sets the random key for Bayesian regression.

- **args.outlier_removal**: if set to *True*, the code will check for any outliers in the CRC and remove them before parameter estimation.


For example, if we aim to estimate kinetic parameters for multiple CRCs, we would set **args.fit_E_S** = *True* and **args.fit_E_I** = *True*. Each dataset will have a measurement error, so **args.multi_var** = *True*, while some datasets from the same plate will share the normalization factor, hence **args.multi_alpha** = *False*. For the enzyme concentration uncertainty, we will assign a lognormal prior with an uncertainty of 10%, thus **args.set_lognormal_dE** = *True* and **args.dE** = *0.1*.

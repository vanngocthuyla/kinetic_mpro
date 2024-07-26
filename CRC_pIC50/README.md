# CRC fitting and pIC50 estimation

## Assumption

All datasets share the same conditions, allowing us to utilize the Maximum A Posteriori (MAP) of posterior distributions of the enzyme-substrate parameters from the global fitting of MERS-CoV MPro as fixed values. This enables the estimation of enzyme-inhibitor parameters, along with the enzyme concentration uncertainty and the normalization factor.

## Input

- The input data should be in .csv format and contain information about the experiment plate, the concentration of the species (nM), the ID of the inhibitor, and the response of the CRC. The final column ("Drop") indicates whether a specific data point should be kept or dropped.

- *Prior.json*: provides information about the prior distribution of the parameters.

- *map_sampling.pickle*: this file, copied from the global fitting, contains the MAP of posterior distributions of the enzyme-substrate parameters.

## Run_me.sh

Assuming `DIR` is your working directory, you can run the model fitting directly by executing:

      python $DIR/kinetic_mpro/scripts/run_CRC_fitting.py --name_inhibitor "ASAP-0014973" --input_file $DIR/kinetic_mpro/CRC_pIC50/input/input_data.csv --prior_infor $DIR/kinetic_mpro/CRC_pIC50/input/Prior.json --fit_E_S  --fit_E_I --initial_values $DIR/kinetic_mpro/CRC_pIC50/input/map_sampling.pickle --out_dir $DIR/kinetic_mpro/CRC_pIC50/output --multi_var --set_lognormal_dE --dE 0.10 --niters 1000 --nburn 200 --nchain 4 --outlier_removal --exclude_first_trace --converged_samples 500 --enzyme_conc_nM 100 --substrate_conc_nM 1350

Alternatively, you can submit a job by adjusting the code in **mpro/scripts/submit_CRC_pIC50.py**. 

There are same arguments to CRC fitting:

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

Other arguments in this analysis:

- **args.exclude_first_trace**: if set to *True*, the MAP from the 1st fitting will be used as initial values for the 2nd fitting to aid in better convergence. Otherwise, the last state of the 1st fitting will be used to collect more samples.

- **args.key_to_check**: Specifies a list of parameters to be checked for model convergence. If not specified, all parameters will be checked.

-	**args.converged_samples**: Specifies the number of converged samples required to estimate the pIC50.

-	**args.enzyme_conc_nM**: Sets the enzyme concentration (in nM) assumed for cellular conditions, which will be used to generate the CRC and estimate the pIC50.

-	**args.substrate_conc_nM**: Sets the substrate concentration (in nM) assumed for cellular conditions, which will be used to generate the CRC and estimate the pIC50.

## Different fitting scenarios

1) We want to collect 500 converged samples and estimate the pIC50, so we set `nburn = 200`, `niters = 1000`, `converged_samples = 500`. The model will undergo multiple fittings:

    * First fitting: This involves running 1200 MCMC samples, excluding the first 200 samples. The convergence of the remaining 1000 samples is checked. If there are at least 500 samples in the equilibration region, the pIC50 can be estimated and reported. Otherwise, a second fitting is performed.

    * Second fitting: This time, 1000 MCMC samples are run using the last state from the first fitting. The convergence of the combined 2000 samples from the first and second fittings is then checked. If there are at least 500 samples in the equilibration region, the pIC50 can be estimated and reported. Otherwise, additional fittings will be performed until enough converged samples are obtained or until the maximum number of fittings (10) is reached.

2) We want to collect 500 converged samples and estimate the pIC50, but we don't want to collect the samples from the first fitting, so we will have additional setting `exclude_first_trace = True`. 

    * First fitting: This involves running 1200 MCMC samples, excluding the first 200 samples. The convergence of the remaining 1000 samples is checked. If there are at least 500 samples in the equilibration region, the pIC50 can be estimated and reported. Otherwise, a second fitting is performed.
    
    * Second fitting: Another 1200 MCMC samples are run using the MAP from the first fitting as the initial values. Again, the first 200 samples are excluded for equilibration, and the convergence of the remaining 1000 samples is checked. If there are at least 500 samples in the equilibration region, the pIC50 can be estimated and reported. Otherwise, a third fitting is performed.
    
    * Third fitting: Similar to the second fitting in the first scenario, another attempt is made to achieve the required number of converged samples.

## Output

The results can be found in the *Convergence* folder. In this folder, all the MCMC samples from multiple fittings are combined, and samples in the equilibration region are extracted. Useful metrics, such as autocorrelation plots, trace plots, and summaries of MCMC convergence, are provided to assess the convergence.

Additionally, the log file contains important information: the number of fittings performed; the estimated pIC50 and its standard deviation if a sufficient number of samples are available. If `outlier_removal = True` and the curve exhibits any specific characteristics, such as being downward or upward, or if it is noisy, this information will also be reported in the log file.

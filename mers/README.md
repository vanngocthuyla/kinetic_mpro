# MERS CoV MPro fitting

The entire procedure comprises two main parts:

- Global Fitting: This step utilizes a subset of the data to estimate shared parameters across all datasets. These shared parameters include the binding affinities between the enzyme and the substrate.
- CRC Fitting and pIC50 Estimation: The shared parameters estimated from the global fitting are used to estimate local parameters for each CRC.

## Global Fitting

The global fitting included 14 sets of experiments: one set of enzyme-substrate (ES), which includes three datasets at three enzyme concentrations, and thirteen sets of enzyme-substrate-inhibitor (ESI), each including four datasets at different experimental conditions.

Due to the computational expense of global fitting, each ES/ESI dataset is fitted individually first, and the posterior distributions of parameters from these fittings are used to assign prior information for the global fitting of all ES/ESI datasets.

This part comprises fitting from step 1 to step 6: 

1. Fitting of ES datasets
2. Fitting 2 curves of each ESI datasets
3. Fitting 4 curves of each ESI datasets
4. Global fitting and analysis
5. Adjusting fitting of each ESI based on posterior distribution of the global fitting
6. Additional analysis

## CRC Fitting and pIC50 estimation

Using information on shared parameters estimated from the global fitting, each CRC can be estimated, including the binding affinities between the enzyme and the inhibitor. The code will automatically generate samples until the number of converged samples is sufficient to estimate the pIC50. An example can be found in /mpro/CRC_pIC50.

7. Outlier detection and fitting each CRC
8. pIC50 estimation for each CRC
9. Additional analysis

# Set up directory

To repeat the whole procedure, simply edit the root directory in `main_dir.txt`. This will update the directory for steps 1 to 9.

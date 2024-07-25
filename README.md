This project focuses on the fitting of concentration-response curves (CRC) from biochemical assays of Coronavirus (CoV) main protease (MPro).

The entire procedure comprises two main steps:

-	Global Fitting: This step utilizes a subset of the data to estimate shared parameters across all datasets. The shared parameters include the binding affinities between the enzyme and the substrate.
-	CRC Fitting and pIC50 Estimation: The shared parameters estimated from the global fitting are used to estimate local parameters for each CRC.

We used Bayesian regression to fit the datasets. Knowledge about kinetic variables is used to assign the prior information to estimate the parameters. Otherwise, uninformative priors can be used instead.


## Global Fitting

The global fitting of SARS-CoV-2 MPro or MERS-CoV MPro can be found in mpro/sars or mpro/mers, respectively.

## CRC Fitting

Using information on shared parameters estimated from the global fitting, each CRC can be estimated, including the binding affinities between the enzyme and the inhibitor. This can be found in mpro/CRC.

## CRC Fitting and pIC50 estimation

The code will automatically generate samples until the number of converged samples is sufficient to estimate the pIC50. This can be found in mpro/CRC_pIC50.

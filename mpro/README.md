This project focuses on the fitting of concentration-respose curve (CRC) from biochemical assay of Coronavirus main protease (MPro).

The entire procedure comprises two main steps:
-	Global Fitting: This step utilizes a subset of the data to estimate shared parameters across all datasets. The shared parameters include the binding affinities between the enzyme and the substrate.
-	CRC Fitting and pIC50 Estimation: The shared parameters estimated from the global fitting are used to estimate local parameters for each CRC.

We used Bayesian regression to fit the datasets. Knowledge about kinetic variable should be used to assign the prior information to estimate the parameters.


# Global Fitting

The global fitting of SARS-CoV-2 MPro or MERS-CoV MPro can be found in mpro/sars or mpro/mers, respectively.

# CRC Fitting

Using information of shared parameters estimated from the global fitting, each CRC can be estimated, and binding affinities between the enzyme and the inhibitor. This can be found in mpro/CRC.

# CRC Fitting and pIC50 estimation

The code will automatically generate samples until the number of converged samples is enough to estimate the pIC50. This can be found in mpro/CRC_pIC50.

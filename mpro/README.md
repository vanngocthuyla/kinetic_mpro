This project focuses on the fitting of concentration-response curves (CRC) from biochemical assays of Coronavirus (CoV) main protease (MPro).

The entire procedure comprises two main steps:

-	Global Fitting: This step utilizes a subset of the data to estimate shared parameters across all datasets. The shared parameters include the binding affinities between the enzyme and the substrate.
-	CRC Fitting and pIC50 Estimation: The shared parameters estimated from the global fitting are used to estimate local parameters for each CRC.

We used Bayesian regression to fit the datasets. Knowledge about kinetic variables is used to assign the prior information to estimate the parameters. Otherwise, uninformative priors can be used instead.


## Global Fitting

The global fitting of SARS-CoV-2 MPro or MERS-CoV MPro can be found in /mpro/sars or /mpro/mers, respectively.

## CRC Fitting

Using information on shared parameters estimated from the global fitting, each CRC can be estimated, including the binding affinities between the enzyme and the inhibitor. An example can be found in /mpro/CRC.

## CRC Fitting and pIC50 estimation

The code will automatically generate samples until the number of converged samples is sufficient to estimate the pIC50. An example can be found in /mpro/CRC_pIC50.

# Set up the running environment

To run the Bayesian regression, we need some python packages. 

  * jax v0.4.13
  * jaxlib v0.4.13
  * numpyro v0.4.13
  * pickle-mixin >=v1.0.2
  * arviz >=0.15.1

If a higher version of JAX and JAXlib is installed, we need to check whether x64 `jax.numpy` can be used by executing the following code without any errors:

    from jax.config import config
    config.update("jax_enable_x64", True)
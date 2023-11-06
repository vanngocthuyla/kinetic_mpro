import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
import jax
import jax.numpy as jnp
import numpy as np


def uniform_prior(name, lower, upper):
    """
    Parameters:
    ----------
    name: string, name of variable
    lower: float, lower value of uniform distribution
    upper: float, upper value of uniform distribution
    ----------
    return numpyro.Uniform
    """
    name = numpyro.sample(name, dist.Uniform(low=lower, high=upper))
    return name


def normal_prior(name, mu, sigma):
    """
    Parameters:
    ----------
    name : string, name of variable
    mu   : float, mean of distribution
    sigma: float, std of distribution
    ----------
    return numpyro.Normal
    """
    name = numpyro.sample(name, dist.Normal(loc=mu, scale=sigma))
    return name


def lognormal_prior(name, stated_value, uncertainty):
    """
    Define a numpyro prior for a deimensionless quantity
    :param name: str
    :param stated_value: float
    :uncertainty: float
    :rerurn: numpyro.Lognormal
    """
    m = stated_value
    v = uncertainty ** 2
    name = numpyro.sample(name, dist.LogNormal(loc=jnp.log(m / jnp.sqrt(1 + (v / (m ** 2)))), 
                                               scale=jnp.sqrt(jnp.log(1 + v / (m**2) )) ))

    return name


def logsigma_guesses(response):
    """
    Parameters:
    ----------
    response: jnp.array, observed data of concentration-response dataset
    ----------
    return range of log of sigma
    """
    log_sigma_guess = jnp.log(response.std()) # jnp.log(response.std())
    log_sigma_min = log_sigma_guess - 10 #log_sigma_min.at[0].set(log_sigma_guess - 10)
    log_sigma_max = log_sigma_guess + 5 #log_sigma_max.at[0].set(log_sigma_guess + 5)
    return log_sigma_min, log_sigma_max
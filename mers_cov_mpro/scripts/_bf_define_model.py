import jax
import jax.numpy as jnp

from _prior_check import convert_prior_from_dict_to_list, check_prior_group
from _MAP_finding_mers_concs import map_finding


class kinetic_model:
    """
    Create the kinetic model information with sampling data, prior, posterior and running setting
    """
    def __init__(self, data, prior, trace, setting):
        """
        Parameters:
        ----------
        data    : dict, kinetic experiment information
        prior   : dict, prior distribution for all parameters
        trace   : dict, mcmc posterior
        setting : dict, model running information
        ----------

        """
        self.data = data
        self.prior = prior
        self.trace = trace
        self.vars = trace.keys()

        self.setting = setting


    def logp(self, trace_map=None):
        """
        Parameters:
        ----------
        trace_map   : dict, mcmc posterior that would be used for finding MAP
        ----------
        Log posterior distribution
        """
        prior_infor = convert_prior_from_dict_to_list(self.prior, self.setting['fit_E_S'], self.setting['fit_E_I'])
        prior_infor_update = check_prior_group(prior_infor, len(self.data))

        if trace_map is None:
            print("Using trace from model")
            trace_map = self.trace.copy()

        [map_index, map_params, log_probs] = map_finding(mcmc_trace=trace_map, experiments=self.data,
                                                         prior_infor=prior_infor_update,
                                                         set_lognormal_dE=self.setting['set_lognormal_dE'], dE=self.setting['dE'],
                                                         set_K_S_DS_equal_K_S_D=self.setting['set_K_S_DS_equal_K_S_D'],
                                                         set_K_S_DI_equal_K_S_DS=self.setting['set_K_S_DI_equal_K_S_DS'],
                                                         show_progress=False)
        return log_probs
import os
import pickle
import json5 as json
import pandas as pd

from _prior_check import check_prior_group, convert_prior_from_dict_to_list, prior_group_multi_enzyme, define_uniform_prior_group


class ModelArgs:
    """
    Custom class to hold model arguments
    """
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            raise AttributeError(f"'ModelArgs' has no attribute '{attr}'")


class Model:
    """
    Create the model information with prior information, shared parameter information,
    running setting and some other constraints

    Example:
        args = ...  # Provide your arguments
        n_enzymes = ...  # Provide the number of enzymes
        model = Model(n_enzymes)
        model.check_model(args)
    """
    def __init__(self, n_enzymes, prior=None):
        self.n_enzymes = n_enzymes
        self.prior_infor = prior
        self.shared_params = None
        self.init_values = None
        self.args = None


    def check_model(self, input_args):
        """
        Attributes in input_args:
            
            traces_name     : str, name of mcmc trace
            initial_values  : dict, initial values for parameters
            last_run_dir    : str, last running directory
            out_dir         : str, output directory
            nburn           : int, number of burn-in
            niters          : int, number of samples
            nthin           : int, number of thinning
            nchain          : int, number of chains
            random_key      : int, random key
            lnKd_min        : float, lower values of uniform distribution for prior of dissociation constants
            lnKd_max        : float, upper values of uniform distribution for prior of dissociation constants
            kcat_min        : float, lower values of uniform distribution for prior of kcat
            kcat_max        : float, upper values of uniform distribution for prior of kcat
            multi_alpha     : boolean, normalization factor
                              If True, setting one alpha for each dataset.
                              If False, multiple datasets having the same plate share alpha
            alpha_min       : float, lower values of uniform distribution for prior of alpha
            alpha_max       : float, upper values of uniform distribution for prior of alpha
            set_lognormal_dE: boolean, using lognormal prior or uniform prior for enzyme concentration uncertainnty
            dE              : float, enzyme concentration uncertainty
            log_sigmas      : dict, measurement error of multiple experiments under log scale
            fixing_log_sigmas       : boolean, if initial values of log_sigmas are provided, and fixing_log_sigmas=True, 
                                      the model won't estimate these parameters
            nsamples_MAP    : int, number of checked samples when finding MAP
            set_equal_      : boolean, different model constraints
        """
        
        # Initial values
        if os.path.isfile(input_args.initial_values):
            init_values = pickle.load(open(input_args.initial_values, "rb"))
            self.init_values = init_values
            print("\nInitial values:", self.init_values, "\n")
        else:
            init_values = None

        # Model arguments
        if init_values is not None and getattr(input_args, 'fixing_log_sigmas', False):
            log_sigmas = {}
            for keys in init_values.keys():
                if key.startswith('log_sigma'):
                    log_sigmas[key] = init_values[key]
        else:
            log_sigmas = None

        alpha_list = {}
        E_list = {}
        if init_values is not None:
            for key in init_values.keys():
                if key.startswith('alpha'):
                    alpha_list[key] = init_values.get(key, None)
            for key in ['dE:100', 'dE:50', 'dE:25']:
                E_list[key] = init_values.get(key, None)
  
        self.args = ModelArgs({
            'fit_E_S'    :      getattr(input_args, 'fit_E_S', True),
            'fit_E_I'    :      getattr(input_args, 'fit_E_I', True),
            'traces_name':      "traces" if input_args.fit_E_S and input_args.fit_E_I else ("traces_E_S" if input_args.fit_E_S else "traces_E_I"),
            'last_run_dir':     getattr(input_args, 'last_run_dir', ""),
            'out_dir':          getattr(input_args, 'out_dir', ""),
            'nburn':            getattr(input_args, 'nburn', 200),
            'niters':           getattr(input_args, 'niters', 1000),
            'nthin':            getattr(input_args, 'nthin', 1),
            'nchain':           getattr(input_args, 'nchain', 4),
            'random_key':       getattr(input_args, 'random_key', 0),
            'lnKd_min':         getattr(input_args, 'lnKd_min', -20.73),
            'lnKd_max':         getattr(input_args, 'lnKd_max', 0),
            'kcat_min':         getattr(input_args, 'kcat_min', 0),
            'kcat_max':         getattr(input_args, 'kcat_max', 10),
            'multi_alpha':      getattr(input_args, 'multi_alpha', False),
            'alpha_min':        getattr(input_args, 'alpha_min', 0),
            'alpha_max':        getattr(input_args, 'alpha_max', 2.),
            'set_lognormal_dE': getattr(input_args, 'set_lognormal_dE', False),
            'dE':               getattr(input_args, 'dE', 0.),
            'alpha_list':       alpha_list,
            'E_list':           E_list,
            'log_sigmas':       log_sigmas,
            'nsamples_MAP':                 getattr(input_args, 'nsamples_MAP', None),
            'set_K_I_M_equal_K_S_M':        getattr(input_args, 'set_K_I_M_equal_K_S_M', False), 
            'set_K_S_DS_equal_K_S_D':       getattr(input_args, 'set_K_S_DS_equal_K_S_D', False),
            'set_K_S_DI_equal_K_S_DS':      getattr(input_args, 'set_K_S_DI_equal_K_S_DS', False),
            'set_kcat_DSS_equal_kcat_DS':   getattr(input_args, 'set_kcat_DSS_equal_kcat_DS', False),
            'set_kcat_DSI_equal_kcat_DS':   getattr(input_args, 'set_kcat_DSI_equal_kcat_DS', False), 
            'set_kcat_DSI_equal_kcat_DSS':  getattr(input_args, 'set_kcat_DSI_equal_kcat_DSS', False),
        })

        # Check model parameters
        assert self.args.lnKd_min < self.args.lnKd_max, "lnKd_min must be lower than lnKd_max."
        assert self.args.lnKd_max <= 0, "Ln(K) should be lower than 0."

        assert self.args.kcat_min < self.args.kcat_max, "kcat_min must be lower than kcat_max."
        assert self.args.kcat_max >= 0, "kcat_min should be larger than 0."

        # Prior
        if self.prior_infor is None:
            if os.path.isfile(input_args.prior_infor):
                prior = json.load(open(input_args.prior_infor))
            else:
                assert len(input_args.shared_params_infor) == 0, print("Incorrect prior information. Please double check the .json file!")
                print("\nUsing default prior information.")
                prior = define_uniform_prior_group(self.args.lnKd_min, self.args.lnKd_max, self.args.kcat_min, self.args.kcat_max)
        else:
            prior = self.prior_infor

        if not os.path.isfile(os.path.join(input_args.out_dir, 'Prior.json')):
            with open(os.path.join(input_args.out_dir, 'Prior.json'), 'w', encoding='utf-8') as f:
                json.dump(prior, f, ensure_ascii=False, indent=4)

        if self.args.set_K_I_M_equal_K_S_M: 
            del prior['logK_I_M']
        if self.args.set_K_S_DS_equal_K_S_D: 
            del prior['logK_S_DS']
        if self.args.set_K_S_DI_equal_K_S_DS: 
            del prior['logK_S_DI']

        if self.args.set_kcat_DSS_equal_kcat_DS: 
            del prior['kcat_DSS']
        if self.args.set_kcat_DSI_equal_kcat_DS: 
            del prior['kcat_DSI']
        if self.args.set_kcat_DSI_equal_kcat_DSS: 
            del prior['kcat_DSI']

        init_prior_infor = convert_prior_from_dict_to_list(prior, input_args.fit_E_S, input_args.fit_E_I)
        self.prior_infor = check_prior_group(init_prior_infor, self.n_enzymes)
        pd.DataFrame(self.prior_infor).to_csv("Prior_infor.csv", index=False)
        print("\nPrior information: \n", pd.DataFrame(self.prior_infor))

        # Shared parameter information
        if os.path.isfile(input_args.shared_params_infor):
            try:
                self.shared_params = json.load(open(input_args.shared_params_infor))
                for key in self.shared_params.keys():
                    assigned_idx = self.shared_params[key]['assigned_idx']
                    shared_idx = self.shared_params[key]['shared_idx']
                    print(f'{key}:{assigned_idx} will not be estimated. Shared parameter: {key}:{shared_idx}')
            except:
                pass
        else:
            print("\nNo shared parameter.")
        
        if (self.shared_params is not None) and (not os.path.isfile(os.path.join(input_args.out_dir,'shared_params.json'))):
            with open('shared_params.json', 'w', encoding='utf-8') as f:
                json.dump(self.shared_params, f, ensure_ascii=False, indent=4)
import numpy as np
from glob import glob
import os

import pickle
import arviz as az
from pymbar import timeseries

import numpy as np

class TraceAdjustment:
    
    def __init__(self, trace, shared_params=None):
        """
        Parameters:
        ----------
        trace : dict
            Trace samples obtained from MCMC (group_by_chain=False).
        shared_params : dict, optional
            Information for shared parameters. If provided, the trace will be adjusted based on these parameters.
        """
        self.trace = trace
        self.shared_params = shared_params if shared_params is not None else {}
        self.trace = self.adjust_by_shared_params()
        self.no_expt = None
        self.logK_names = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
        self.kcat_names = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']
        self.other_params_name = ['alpha', 'dE']

    def adjust_by_shared_params(self):
        """
        Adjusts the trace by shared parameter information.

        For each parameter, both "assigned_idx" and "shared_idx" are required for indices of shared parameter.
        For example, if enzymes A and B share the same dimerization logKd, and we have: "logKd": "assigned_idx": 0, "shared_idx": 1.
        This method will adjust the trace accordingly.

        Returns:
        ----------
        dict
            The trace adjusted by shared parameter information.
        """
        trace_update = self.trace.copy()
        if self.shared_params:
            for name, param in self.shared_params.items():
                assigned_idx = param['assigned_idx']
                shared_idx = param['shared_idx']
                trace_update[f'{name}:{assigned_idx}'] = self.trace.get(f'{name}:{shared_idx}', None)
        return trace_update

    def adjust_idx_for_multi_expts(self):
        """
        Return the trace for multiple experiments.

        If there are multiple experiments, adjust the trace keys to include experiment indices.
        For example, if there are two experiments sharing parameters except for logKd, the trace will have
        keys like 'logKd:0', 'logKd:1', 'logK_S_M:0', 'logK_S_M:1', 'logK_S_D:0', 'logK_S_D:1'.

        If there is only one experiment, update keys to include index 0.
        """
        idx = 0
        trace_update = {}
        flag_check = True

        while flag_check:
            # Check if there are any keys with the current idx
            check = np.sum([1 for key in self.logK_names + self.kcat_names if f'{key}:{idx}' in self.trace])
            
            if check == 0:
                flag_check = False
                self.no_expt = idx
            else:
                for key in self.logK_names + self.kcat_names:
                    if f'{key}:{idx}' in self.trace:
                        trace_update[f'{key}:{idx}'] = self.trace[f'{key}:{idx}']
                    else:
                        trace_update[f'{key}:{idx}'] = self.trace.get(key, None)
                idx += 1

        if not flag_check and idx == 0:
            for key in self.trace.keys():
                trace_update[f'{key}:0'] = self.trace[key]
            self.no_expt = 1

        return trace_update

    def adjust_extract_ith_params(self, ith):
        """
        Parameters:
        ----------
        ith : int, index of the experiment to extract.
        ----------

        Returns: dict, trace for the ith experiment.
            
        """
        trace_update = self.adjust_idx_for_multi_expts()
        trace_update_ith = {}
        if ith < self.no_expt:
            for key in self.logK_names + self.kcat_names:
                trace_update_ith[f'{key}'] = trace_update.get(f'{key}:{ith}', None)
        return trace_update_ith


class TraceConverter:
    
    def __init__(self, trace, nchain=4):
        """
        Parameters:
        ----------
        trace  : mcmc.get_samples(group_by_chain=False)
        nchain : int, number of chains (default is 4)
        """
        self.trace = trace
        self.nchain = nchain

    def ln_to_log(self, update_key_log10=False):
        """
        Convert kinetic parameters in the trace from ln scale to log10 scale.

        Returns:
        ----------
        trace_log : dict, kinetic parameters in the log10 scale
        """
        trace_log = {}
        for key in self.trace.keys():
            if key.startswith('log'):
                if update_key_log10:
                    new_key = 'log10' + key[3:]
                else:
                    new_key = key
                trace_log[new_key] = np.log10(np.exp(self.trace[key]))
            else:
                trace_log[key] = self.trace[key]
        return trace_log

    def ln_to_DeltaG(self):
        """
        Convert kinetic parameters in the trace from ln scale to DeltaG in kcal/mol.

        Returns:
        ----------
        trace_DeltaG : dict, kinetic parameters in kcal/mol
        """
        trace_DeltaG = {}
        RT = 8.314E-3 * 300 / 4.184  # 1 calorie = 4.184 Joules
        
        for key in self.trace.keys():
            if key.startswith('logK'): 
                if key == 'logKd':
                    new_key = 'DeltaG_' + key[4:]
                else:
                    new_key = 'DeltaG_' + key[5:]
                trace_DeltaG[new_key] = RT * self.trace[key]
            else:
                trace_DeltaG[key] = self.trace[key]
        return trace_DeltaG

    def one_to_nchain(self):
        """
        Convert the trace from group_by_chain=False to group_by_chain=True.

        Returns:
        ----------
        trace_group : dict or np.ndarray, trace with samples grouped by chain
        """
        if isinstance(self.trace, dict):
            trace_group = {}
            for key in self.trace.keys():
                trace_group[key] = np.reshape(self.trace[key], (self.nchain, int(len(self.trace[key]) / self.nchain)))
        elif isinstance(self.trace, np.ndarray):
            trace_group = np.reshape(self.trace, (self.nchain, int(len(self.trace) / self.nchain)))
        return trace_group

    def convert_name(self, change_names=None, name_startswith=''):
        """
        Convert trace parameter names to more readable names using LaTeX formatting.

        Parameters:
        ----------
        change_names : dict, optional
            Dictionary to specify custom name mappings. If provided, only these names will be used.

        Returns:
        ----------
        dict
            Updated trace with converted names.
        """
        default_change_names = {'kcat_DS':      '$k_{cat,DS}$',
                                'kcat_DSI':     '$k_{cat,DSI}$',
                                'kcat_DSS':     '$k_{cat,DSS}$',
                                'logK_S_M':     '$logK_{S,M}$',
                                'logK_S_D':     '$logK_{S,D}$',
                                'logK_S_DS':    '$logK_{S,DS}$',
                                'logK_I_M':     '$logK_{I,M}$',
                                'logK_I_D':     '$logK_{I,D}$',
                                'logK_I_DI':    '$logK_{I,DI}$',
                                'logK_S_DI':    '$logK_{S,DI}$',
                                'logK_I_M':     '$logK_{I,M}$',
                                'logK_I_D':     '$logK_{I,D}$',
                                'logK_I_DI':    '$logK_{I,DI}$',
                                'logK_S_DI':    '$logK_{S,DI}$',
                                'logKd':        '$logK_d$',
                                'DeltaG_S_M':   '$\Delta G_{S,M}$',
                                'DeltaG_S_D':   '$\Delta G_{S,D}$',
                                'DeltaG_S_DS':  '$\Delta G_{S,DS}$',
                                'DeltaG_I_M':   '$\Delta G_{I,M}$',
                                'DeltaG_I_D':   '$\Delta G_{I,D}$',
                                'DeltaG_I_DI':  '$\Delta G_{I,DI}$',
                                'DeltaG_S_DI':  '$\Delta G_{S,DI}$',
                                'DeltaG_I_M':   '$\Delta G_{I,M}$',
                                'DeltaG_I_D':   '$\Delta G_{I,D}$',
                                'DeltaG_I_DI':  '$\Delta G_{I,DI}$',
                                'DeltaG_S_DI':  '$\Delta G_{S,DI}$',
                                'DeltaG_Kd':        '$logK_d$',
                                'alpha:E100_S1350':  '$\\alpha^{E:100nM, S:1350nM}$',
                                'alpha:E100_S50':    '$\\alpha^{E:100nM, S:50nM}$',
                                'alpha:E100_S750':   '$\\alpha^{E:100nM, S:750nM}$',
                                'alpha:E50_S150':    '$\\alpha^{E:50nM, S:150nM}$',
                                'alpha:E50_S550':    '$\\alpha^{E:50nM, S:550nM}$',
                                'dE:100':       '$\Delta E^{100nM}$',
                                'dE:50':        '$\Delta E^{50nM}$',
                                'dE:25':        '$\Delta E^{25nM}$',
                                'log_sigma:ES'  :       'log$\sigma^{ES}$',
                                'log_sigma:ES:0':       'log$\sigma^{ES}$',
                                'log_sigma:ESI:0':      'log$\sigma^{E:100nM, S:50nM}$',
                                'log_sigma:ESI:1':      'log$\sigma^{E:100nM, S:50nM}$',
                                'log_sigma:ESI:2':      'log$\sigma^{E:100nM, S:750nM}$',
                                'log_sigma:ESI:3':      'log$\sigma^{E:50nM, S:150nM}$',
                           }
        if change_names is None:
            change_names = {}
        change_names_full = {**default_change_names, **change_names}

        trace_update = {}
        for key, new_key in change_names_full.items():
            if key in self.trace.keys() and key.startswith(name_startswith):
                trace_update[new_key] = self.trace.get(key, None)

        return trace_update


class TraceExtraction:
    
    def __init__(self, trace):
        """
        Parameters:
        ----------
        trace : mcmc.get_samples(group_by_chain=False)
        """
        self.trace = trace

    def extract_samples_from_trace(self, params, burn=0, thin=0):
        """
        Parameters:
        ----------
        params : list of parameter that would be extracted
        burn   : int, the number of initial samples that would be removed
        thin   : int, picking separated points from the sample, at each k-th step

        return two lists of kinetics parameters from trace and prior information
        """
        extract_samples = {}
        for var in params:
            try:
                samples = np.concatenate(np.array(self.trace.posterior[var]))
                if burn != 0:
                    samples = samples[burn:]
                if thin != 0:
                    samples = samples[::thin]
                extract_samples[var] = samples
            except Exception as e:
                continue
        return extract_samples

    def extract_shared_params(self):
        """
        Extract parameters that are shared among all experiments
        """
        logK_names = ['logKd', 'logK_S_M', 'logK_S_D', 'logK_S_DS', 'logK_I_M', 'logK_I_D', 'logK_I_DI', 'logK_S_DI']
        kcat_names = ['kcat_MS', 'kcat_DS', 'kcat_DSI', 'kcat_DSS']
        trace_update = {}
        for key in logK_names+kcat_names:
            if key in self.trace:
                trace_update[key] = trace[key]
        return trace_update
        

    def extract_params_from_trace_and_prior(self, prior_info, estimator='mean'):
        """
        Parameters:
        ----------
        prior_info : list of dict of assigned prior distribution for kinetics parameters
        estimator  : str, 'mean' or 'median'. Return the mean or median of MCMC trace

        Example of prior_info:
            [{'type': 'logKd', 'name': 'logKd',  'fit': 'local', 'dist': None, 'value': array([-11, -14])},
             {'type': 'logK','name': 'logK_S_M','fit': 'global','dist': 'uniform','lower': -20.0,'upper': 0.0}]
        ----------
        return two lists of kinetics parameters from trace and prior information
        """
        assert estimator in ['mean', 'median'], "Wrong argument! Estimator is 'mean' or 'median'"

        data = az.convert_to_inference_data(self.trace)
        params_name_kcat = [name for name in self.trace.keys() if name.startswith('kcat')]
        params_name_logK = [name for name in self.trace.keys() if name.startswith('logK')]

        samples_logK = self.extract_samples_from_trace(params_name_logK)
        samples_kcat = self.extract_samples_from_trace(params_name_kcat)
        params_logK = {}
        params_kcat = {}

        if estimator == 'mean':
            for name in params_name_logK:
                params_logK[name] = np.mean(samples_logK[name])
            for name in params_name_kcat:
                params_kcat[name] = np.mean(samples_kcat[name])
        else:
            for name in params_name_logK:
                params_logK[name] = np.median(samples_logK[name])
            for name in params_name_kcat:
                params_kcat[name] = np.median(samples_kcat[name])

        for prior in prior_info:
            if prior['dist'] is None:
                name = prior['name']
                if name.startswith('logK'):
                    if prior['fit'] == 'local':
                        for n in range(len(prior['value'])):
                            params_logK[f"{name}:{n}"] = prior['value'][n]
                    else:
                        params_logK[name] = prior['value']
                else:
                    if prior['fit'] == 'local':
                        for n in range(len(prior['value'])):
                            params_kcat[f"{name}:{n}"] = prior['value'][n]
                    else:
                        params_kcat[name] = prior['value']

        params_kcat = {k: (v if v is not None else 0.0) for k, v in params_kcat.items()}

        return params_logK, params_kcat

    def extract_params_from_map_and_prior(self, map_idx, prior_info):
        """
        Parameters:
        ----------
        map_idx    : index of map from mcmc
        prior_info : list of dict of assigned prior distribution for kinetics parameters

        Example of prior_info:
            [{'type': 'logKd', 'name': 'logKd',  'fit': 'local', 'dist': None, 'value': array([-11, -14])},
             {'type': 'logK','name': 'logK_S_M','fit': 'global','dist': 'uniform','lower': -20.0,'upper': 0.0}]
        ----------
        return two lists of kinetics parameters from map of trace and prior information
        """
        data = az.convert_to_inference_data(self.trace)
        params_name_kcat = [name for name in self.trace.keys() if name.startswith('kcat')]
        params_name_logK = [name for name in self.trace.keys() if name.startswith('logK')]

        params_logK = {}
        params_kcat = {}

        for name in self.trace.keys():
            if name in params_name_logK:
                params_logK[name] = self.trace[name][map_idx]
            elif name in params_name_kcat:
                params_kcat[name] = self.trace[name][map_idx]

        for prior in prior_info:
            if prior['dist'] is None:
                name = prior['name']
                if name.startswith('logK'):
                    if prior['fit'] == 'local':
                        for n in range(len(prior['value'])):
                            params_logK[f"{name}:{n}"] = prior['value'][n]
                    else:
                        params_logK[name] = prior['value']
                else:
                    if prior['fit'] == 'local':
                        for n in range(len(prior['value'])):
                            params_kcat[f"{name}:{n}"] = prior['value'][n]
                    else:
                        params_kcat[name] = prior['value']

        params_kcat = {k: (v if v is not None else 0.0) for k, v in params_kcat.items()}

        return params_logK, params_kcat

    def extract_by_start_expected_nsample(self, start=0, nchain=4, expected_nsample=0):
        """
        Parameters:
        ----------
        start            : integer, starting point to extract
        nchain           : number of chain from mcmc trace
        expected_nsample : integer, number of samples expected for the output traces.pickle
        ----------
        Return extracted trace, and reporting if there is enough samples.
        """
        assert expected_nsample > 0, "Please provide the expected_nsample for this flag."

        keys = list(self.trace.keys())
        _nsample = int(len(self.trace[keys[0]][start:]) / nchain)

        if _nsample < expected_nsample:
            converged_trace = self.trace
            flag = False
            message = f"There are still not enough {expected_nsample} samples. Only {_nsample} available."
        else:
            converged_trace = {key: self.trace[key][start:(start + nchain * expected_nsample)] for key in self.trace.keys()}
            flag = True
            message = f'There are {expected_nsample} samples available.'

        return converged_trace, flag, message


## Trace rhat ## --------------------------------------------------------------------

def _rhat(trace, nchain=4):
    """
    Parameters:
    ----------
    trace           : mcmc trace, can be dictionary of multiple variables 
    nchain          : number of chain from mcmc trace
    ----------
    Estimating rhat of the mcmc trace
    """
    trace_group = {}
    if type(trace) is dict:
        for key in trace.keys():
            trace_group = _trace_one_to_nchain(trace, nchain=nchain)
        _rhats = az.rhat(trace_group).to_dict()['data_vars']
        rhat = {}
        for key in _rhats.keys():
            rhat[key] = _rhats[key]['data']
    elif type(trace) is np.ndarray:
        trace_group = _trace_one_to_nchain(trace, nchain=nchain)
        rhat = az.rhat(trace_group)
    return rhat


def _convergence_rhat_all_chain(trace, nchain=4, digit=1):
    """
    Parameters:
    ----------
    trace           : mcmc trace, can be dictionary of multiple variables 
    nchain          : number of chain from mcmc trace
    digit           : number of decimal places to round to for rhat
    ----------
    Return the True if rhat approximate 1. Otherwise, return False
    """
    r = _rhat(trace, nchain=nchain)
    flag = True
    if type(trace) is dict:
        for key in trace.keys():
            if round(r[key], digit) != 1:
                flag = False
                return flag
    elif type(trace) is np.ndarray:
        if round(r, digit) != 1:
            flag = False
    return flag


def _convergence_rhat_one_chain_removal(trace, nchain=4, digit=1):
    """
    Parameters:
    ----------
    trace           : mcmc trace, can be dictionary of multiple variables 
    nchain          : number of chain from mcmc trace
    digit           : number of decimal places to round to for rhat
    ----------
    Checking if one chain of mcmc trace can be stuck in local minimum and removing that chain can result a converged trace. 

    Return the True if rhat approximate 1. Otherwise, return False.
    """
    flag = True
    idx_chain = []

    if type(trace) is dict:
        trace_group = _trace_one_to_nchain(trace, nchain=nchain)
        for key in trace_group.keys():
            for i in range(nchain):
                extracted_row = [row for row in range(nchain) if row != i]
                r = az.rhat(trace_group[key][extracted_row])
                if round(r, digit) != 1:
                    flag = False
                else:
                    idx_chain.append(i)

            # If one parameter unconverged, escape loop and return false. We don't need to check all parameters.
            if flag is False:
                return [flag, None]

    elif type(trace) is np.ndarray:
        trace_group = _trace_one_to_nchain(trace, nchain=nchain)
        
        for i in range(nchain):
            extracted_row = [row for row in range(nchain) if row != i]
            r = az.rhat(trace_group[extracted_row])
            if round(r, digit) != 1:
                flag = False
            else:
                idx_chain.append(i)
    
    if len(np.unique(idx_chain)) == 1:
        return [True, idx_chain[0]]
    else:
        return [False, None]


def _convergence_rhat(trace, nchain=4, digit=1, one_chain_removal=False):
    """
    Parameters:
    ----------
    trace           : mcmc trace, can be dictionary of multiple variables 
    nchain          : number of chain from mcmc trace
    digit           : number of decimal places to round to for rhat
    ----------
    Return the True if rhat approximate 1. Otherwise, return False
    """
    flag = _convergence_rhat_all_chain(trace, nchain=nchain, digit=digit)
    if not flag and one_chain_removal:
        print(f"Checking convergence of {nchain-1} chains.")
        return _convergence_rhat_one_chain_removal(trace, nchain=nchain, digit=digit)
    else:
        return [flag, None]


## Trace - convergence ## --------------------------------------------------------------------


def _trace_pymbar(trace, nskip=100, nchain=4, key_to_check=""):
    """
    Parameters:
    ----------
    trace           : mcmc trace, which is a dictionary of multiple variables 
    nskip           : nskip in timeseries.detect_equilibration
    nchain          : number of chain from mcmc trace
    key_to_check    : parameters that would be check
    ----------
    Return the converged trace, converged point t0, and nsample after convergence.
    """
    if len(key_to_check)>0:
        for key in key_to_check:
            assert key in trace.keys(), "Please provide the correct parameter name."
    else:
        key_to_check = trace.keys()

    t0 = 0
    for key in key_to_check:
        trace_t = trace[key]
        _t0, g, Neff_max = timeseries.detect_equilibration(trace_t, nskip=nskip)
        if _t0 > t0:
            t0 = _t0

    converged_nsample = int(len(trace_t[t0:])/nchain)
    converged_trace={}
    for key in trace.keys():
        converged_trace[key] = trace[key][t0:]

    return [converged_trace, t0, converged_nsample]


def _trace_convergence(mcmc_files, out_dir=None, nskip=100, nchain=4, expected_nsample=0,
                       key_to_check="", converged_trace_name='Converged_trace',
                       one_chain_removal=False, digit=1):
    """
    Parameters:
    ----------
    mcmc_files          : list of string, all traces.pickle files from different directory
    out_dir             : string, directory to save output
    nskip               : integer, nskip in timeseries.detect_equilibration
    nchain              : integer, number of chain from mcmc trace
    expected_nsample    : integer, number of samples expected for the output traces.pickle
    key_to_check        : list, parameters that would be check
    one_chain_removal   : boolean, checking if removing one chain can lead to converged mcmc trace
    digit               : number of decimal places to round to for rhat
    ----------
    
    Checking and saving the converged trace. If the expected_nsample is given, 
    reporting if there is not enough samples after the converged point ('t0').

    If one_chain_removal=True, checking if one chain of mcmc trace can be stuck in local minimum.
    In such case, removing that chain and checking the convergence again. 
    
    """
    if out_dir is not None and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for mcmc_file in mcmc_files: 
        mes = f'Loading {mcmc_file}'
        print(mes)
        if out_dir is not None:
            with open(os.path.join(out_dir, "log.txt"), "a") as f:
                print(mes, file=f)

    # Combining multiple traces and checking convergence by pymbar
    multi_trace = _combining_multi_trace(mcmc_files, nchain)
    [_trace, t0, _nsample] = _trace_pymbar(multi_trace, nskip=nskip, nchain=nchain, key_to_check=key_to_check)

    nchain_update = nchain
    # if expected_nsample is 0, extracting the converged trace by pymbar
    if expected_nsample==0:
        trace = _trace
        convergence_flag = "Please provide the expected_nsample for this flag."
    else:
        [trace, convergence_flag, mes] = TraceExtraction(trace=multi_trace).extract_by_start_expected_nsample(start=t0, nchain=nchain, expected_nsample=expected_nsample)
        
        # if expected_nsample larger than 0, and the number of converged samples is not enough
        if not convergence_flag:
            if one_chain_removal:

                # Find the different chain by rhat
                rhat_flag, idx = _convergence_rhat(trace=multi_trace, nchain=nchain, digit=digit, one_chain_removal=one_chain_removal)
                
                if rhat_flag and (idx is not None):
                    extracted_row = [row for row in range(nchain) if row != idx]
                    print(extracted_row)
                    trace_group = _trace_one_to_nchain(multi_trace, args.nchain)
                    extracted_trace = {key: trace_group[key][extracted_row].flatten() for key in trace_group.keys()}
                    
                    #Using pymbar to check the convergence and extract trace again
                    [_trace, t0, _] = _trace_pymbar(trace=extracted_trace, nchain=nchain, key_to_check=key_to_check)
                    [trace, convergence_flag, mes] = TraceExtraction(trace=_trace).extract_by_start_expected_nsample(start=t0, nchain=nchain, expected_nsample=expected_nsample)
                    nchain_update = nchain-1

    print(mes)
    
    if out_dir is not None:
        with open(os.path.join(out_dir, "log.txt"), "a") as f:
            print(mes, file=f)

        if convergence_flag:
            pickle.dump(trace, open(os.path.join(out_dir, converged_trace_name+".pickle"), "wb"))
            trace_group = _trace_one_to_nchain(trace, nchain=nchain_update)
            az.summary(trace_group).to_csv(os.path.join(out_dir, "Converged_summary.csv"))

    return [trace, convergence_flag, nchain_update]


def _combining_multi_trace(mcmc_files, nchain=4, nsample=None, params_names=None,
                           out_dir=None, combined_trace_name='Combined_trace'):
    """
    Parameters:
    ----------
    mcmc_files      : all traces.pickle files from different directory
    nchain          : number of chain from mcmc trace
    nsample         : number of samples expected for the output traces.pickle
    params_names    : list of parameter names to extract
    out_dir         : directory to save output
    ----------
    Combining mutiple pickle files into 1 file
    """
    if params_names is None:
        assert len(mcmc_files)>0, "Please provide at least one file of MCMC trace."
        trace = pickle.load(open(mcmc_files[0], "rb"))
        params_names = trace.keys()

    multi_trace = {}
    trace_group = {}
    for params_name in params_names:
        trace = np.array([[], [], [], []])
        for mcmc_file in mcmc_files:
            _trace = pickle.load(open(mcmc_file, "rb"))[params_name]
            if nsample is None:
                _nsample = int(len(_trace)/nchain)
            else:
                _nsample = nsample
            _trace = np.reshape(np.array(_trace), (nchain, _nsample)) #4*10000 matrix
            trace = np.hstack((trace, _trace))
        multi_trace[params_name] = trace.flatten()
        trace_group[params_name] = np.reshape(trace, (nchain, int(len(trace.flatten())/nchain)))

    if out_dir is not None:
        pickle.dump(multi_trace, open(os.path.join(out_dir, combined_trace_name+".pickle"), "wb"))
        az.summary(trace_group).to_csv(os.path.join(out_dir, "Combined_summary.csv"))

    return multi_trace
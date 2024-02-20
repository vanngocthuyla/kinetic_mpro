import numpy as np
from glob import glob
import os

import pickle
import arviz as az
from pymbar import timeseries

<<<<<<< HEAD
=======
def _trace_ln_to_log(trace, group_by_chain=False, nchain=4):
    """
    Parameters:
    ----------
    trace        : mcmc.get_samples(group_by_chain=True)

    return kinetic parameters in the log10 scale
    """
    trace_log = {}
    for key in trace.keys():
        if key.startswith('log') or key.startswith('ln'):
            if group_by_chain: trace_log[key] = np.reshape(np.log10(np.exp(trace[key])), (nchain, int(len(trace[key])/nchain)))
            else: trace_log[key] = np.log10(np.exp(trace[key]))
        else:
            if group_by_chain: trace_log[key] = np.reshape(trace[key], (nchain, int(len(trace[key])/nchain)))
            else: trace_log[key] = trace[key]
    return trace_log

>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0

def extract_samples_from_trace(trace, params, burn=0, thin=0):
    """
    Parameters:
    ----------
    trace        : mcmc.get_samples(group_by_chain=True)
    params       : list of parameter that would be extracted
    burn         : int, the number of initial samples that would be removed
    thin         : int, picking separated points from the sample, at each k-th step

    return two lists of kinetics parameters from trace and prior information
    """
    extract_samples = {}
    for var in params:
      try:
          samples = np.concatenate(np.array(trace.posterior[var]))
          if burn!=0: 
              samples = samples[burn:]
          if thin!=0:
              samples = samples[::thin]
          extract_samples[var] = samples
      except:
          continue
    return extract_samples


def extract_params_from_trace_and_prior(trace, prior_infor, estimator='mean'):
    """
    Parameters:
    ----------
    trace        : mcmc.get_samples(group_by_chain=True)
    prior_infor  : list of dict of assigned prior distribution for kinetics parameters
    estimator    : str, 'mean' or 'median'. Return the mean of median of MCMC trace

    Example of prior_infor:
        [{'type': 'logKd', 'name': 'logKd',  'fit': 'local', 'dist': None, 'value': array([-11, -14])},
         {'type': 'logK','name': 'logK_S_M','fit': 'global','dist': 'uniform','lower': -20.0,'upper': 0.0}]
    ----------
    return two lists of kinetics parameters from trace and prior information
    """
    assert estimator in ['mean', 'median'], print("Wrong argument! Estimator is mean or median")

    data = az.convert_to_inference_data(trace)
    params_name_kcat = []
    params_name_logK = []
    for name in trace.keys():
        if name.startswith('logK'):
            params_name_logK.append(name)
        if name.startswith('kcat'):
            params_name_kcat.append(name)

    samples_logK = extract_samples_from_trace(data, params_name_logK)
    samples_kcat = extract_samples_from_trace(data, params_name_kcat)
    params_logK = {}
    params_kcat = {}

    # Extract params from trace
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

    # Extract params from prior information
    for prior in prior_infor:
        if prior['dist'] is None:
            name = prior['name']
            if name.startswith('logK'):
                if prior['fit'] == 'local':
                    for n in range(len(prior['value'])):
                        params_logK[name+':'+str(n)] = prior['value'][n]
                else: params_logK[name] = prior['value']
            else:
                if prior['fit'] == 'local':
                    for n in range(len(prior['value'])):
                        params_kcat[name+':'+str(n)] = prior['value'][n]
                else: params_kcat[name] = prior['value']

    for n, params in enumerate(params_kcat):
        if params is None:
            params_kcat[n] = 0.

    return params_logK, params_kcat


# def list_kinetics_params(init_params, update_params):
#     results = init_params
#     for name_params in update_params:
#         results[name_params] = update_params[name_params]
#     return results


def extract_params_from_map_and_prior(trace, map_idx, prior_infor):
    """
    Parameters:
    ----------
    trace        : mcmc.get_samples(group_by_chain=False)
    map_idx      : index of map from mcmc
    prior_infor  : list of dict of assigned prior distribution for kinetics parameters
    
    Example of prior_infor:
        [{'type': 'logKd', 'name': 'logKd',  'fit': 'local', 'dist': None, 'value': array([-11, -14])},
         {'type': 'logK','name': 'logK_S_M','fit': 'global','dist': 'uniform','lower': -20.0,'upper': 0.0}]
    ----------
    return two lists of kinetics parameters from map of trace and prior information
    """

    data = az.convert_to_inference_data(trace)
    params_name_kcat = []
    params_name_logK = []
    for name in trace.keys():
        if name.startswith('logK'):
            params_name_logK.append(name)
        if name.startswith('kcat'):
            params_name_kcat.append(name)

    params_logK = {}
    params_kcat = {}

    # Extract params from trace[map_idx]
    for name in trace.keys():
        if name in params_name_logK:
            params_logK[name] = trace[name][map_idx]
        elif name in params_name_kcat:
            params_kcat[name] = trace[name][map_idx]

    # Extract params from prior information
    for prior in prior_infor:
        if prior['dist'] is None:
            name = prior['name']
            if name.startswith('logK'):
                if prior['fit'] == 'local':
                    for n in range(len(prior['value'])):
                        params_logK[name+':'+str(n)] = prior['value'][n]
                else: params_logK[name] = prior['value']
            else:
                if prior['fit'] == 'local':
                    for n in range(len(prior['value'])):
                        params_kcat[name+':'+str(n)] = prior['value'][n]
                else: params_kcat[name] = prior['value']

    for n, params in enumerate(params_kcat):
        if params is None:
            params_kcat[n] = 0.

    return params_logK, params_kcat


<<<<<<< HEAD
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


def _trace_ln_to_log(trace, group_by_chain=False, nchain=4):
    """
    Parameters:
    ----------
    trace        : mcmc.get_samples(group_by_chain=True)

    return kinetic parameters in the log10 scale
    """
    trace_log = {}
    for key in trace.keys():
        if key.startswith('log') or key.startswith('ln'):
            if group_by_chain: trace_log[key] = np.reshape(np.log10(np.exp(trace[key])), (nchain, int(len(trace[key])/nchain)))
            else: trace_log[key] = np.log10(np.exp(trace[key]))
        else:
            if group_by_chain: trace_log[key] = np.reshape(trace[key], (nchain, int(len(trace[key])/nchain)))
            else: trace_log[key] = trace[key]
    return trace_log


def _trace_one_to_nchain(trace, nchain=4):
    """
    Parameters:
    ----------
    trace        : mcmc.get_samples(group_by_chain=False)

    return trace as mcmc.get_samples(group_by_chain=True)
    """
    if type(trace) is dict:
        trace_group = {}
        for key in trace.keys():
            trace_group[key] = np.reshape(trace[key], (nchain, int(len(trace[key])/nchain)))
    elif type(trace) is np.ndarray:
        trace_group = np.reshape(trace, (nchain, int(len(trace)/nchain)))
    return trace_group


def _trace_extract_by_start_expected_nsample(trace, start=0, nchain=4, expected_nsample=0):
    """
    Parameters:
    ----------
    trace           : mcmc trace, which is a dictionary of multiple variables 
    start           : integer, starting point to extract
    nchain          : number of chain from mcmc trace
    expected_nsample: integer, number of samples expected for the output traces.pickle
    ----------
    Return extracted trace, and reporting if there is enough samples.
    """
    assert expected_nsample>0, print("Please provide the expected_nsample for this flag.")
    
    keys = list(trace.keys())
    _nsample = int(len(trace[keys[0]][start:])/nchain)

    if _nsample < expected_nsample:
        converged_trace = trace
        flag = False
        message = f"There are still not enough {expected_nsample} samples. Only {_nsample} available."
    else: 
        converged_trace = {}
        for key in trace.keys():
            converged_trace[key] = trace[key][start:(start+nchain*expected_nsample)]
        flag = True
        message = f'There are {expected_nsample} samples available.'

    return [converged_trace, flag, message]


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
=======
def _trace_convergence(mcmc_files, out_dir, nskip=100, nchain=4, nsample=10000,
                       key_to_check="", converged_trace_name='Converged_trace'):
    """
    Parameters:
    ----------
    mcmc_files      : all traces.pickle files from different directory
    out_dir         : directory to save output
    nskip           : nskip in timeseries.detect_equilibration
    nchain          : number of chain from mcmc trace
    nsample         : number of samples expected for the output traces.pickle
    key_to_check    : parameters that would be check
    ----------
    If all the chains converged and the number of nsample lower than expected, save the new trace.
    Otherwise, reporting that there is not enough samples after the converged point ('t0').
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for mcmc_file in mcmc_files: 
        with open(os.path.join(out_dir, "log.txt"), "a") as f:
            print(f'Loading {mcmc_file}', file=f)

    # Combining multiple traces
    multi_trace = _combining_multi_trace(mcmc_files, nchain)

    if len(key_to_check)>0:
        for key in key_to_check:
            assert key in multi_trace.keys(), "Please provide the correct parameter name."
    else:
        key_to_check = multi_trace.keys()
    # print("Checking parameters:", key_to_check)

    t0 = 0
    for key in key_to_check:
        trace_t = multi_trace[key]
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
        _t0, g, Neff_max = timeseries.detect_equilibration(trace_t, nskip=nskip)
        if _t0 > t0:
            t0 = _t0

<<<<<<< HEAD
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
        [trace, convergence_flag, mes] = _trace_extract_by_start_expected_nsample(trace=multi_trace, start=t0, nchain=nchain, expected_nsample=expected_nsample)
        
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
                    [trace, convergence_flag, mes] = _trace_extract_by_start_expected_nsample(trace=_trace, start=t0, nchain=nchain, expected_nsample=expected_nsample)
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
=======
    _nsample = int(len(trace_t[t0:])/nchain)
    trace = {}
    if _nsample < nsample:
        print("There is still not enough", nsample, "samples. Only", _nsample, "available.")
        convergence_flag = False
        for key in multi_trace.keys():
            trace[key] = multi_trace[key][t0:]
        with open(os.path.join(out_dir, "log.txt"), "a") as f:
            print(f"\nThere is still not enough {nsample} samples. Only {_nsample} available.", file=f)
    else: 
        convergence_flag = True
        for key in multi_trace.keys():
            trace[key] = multi_trace[key][t0:(t0+nchain*nsample)]
        with open(os.path.join(out_dir, "log.txt"), "a") as f:
            print(f'\nThere is {nsample} samples available.', file=f)

    pickle.dump(trace, open(os.path.join(out_dir, converged_trace_name+".pickle"), "wb"))

    trace_group = {}
    for key in trace.keys():
        trace_group[key] = np.reshape(trace[key], (nchain, int(len(trace[key])/nchain)))
    az.summary(trace_group).to_csv(os.path.join(out_dir, "Converged_summary.csv"))

    return trace, convergence_flag
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0


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
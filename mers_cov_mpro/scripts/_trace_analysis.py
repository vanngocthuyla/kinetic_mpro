import numpy as np
from glob import glob
import os

import pickle
import arviz as az
from pymbar import timeseries

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
        _t0, g, Neff_max = timeseries.detect_equilibration(trace_t, nskip=nskip)
        if _t0 > t0:
            t0 = _t0

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
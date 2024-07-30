# Fitting Bayesian model for Mpro given some constraints on parameters
import os
import pickle

import arviz as az

import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from _plotting import plotting_trace
from _model import global_fitting

from _kinetics import ReactionRate
from _prior_distribution import uniform_prior, normal_prior, logsigma_guesses, lognormal_prior
from _params_extraction import extract_logK_n_idx, extract_kcat_n_idx
from _prior_check import check_prior_group, prior_group_multi_enzyme, define_uniform_prior_group
from _model import _dE_priors, _dE_find_prior, _alpha_priors, _alpha_find_prior, fitting_each_dataset
from _pIC50 import scaling_data
from _plotting import plotting_trace 

import matplotlib.pyplot as plt


def _run_mcmc_CRC(expts, prior_infor, shared_params, init_values, last_run_dir, out_dir, args):
    """
    Parameters:
    ----------
    expts           : list of dict of multiple enzymes
        Each enzymes dataset contains multiple experimental datasets, including data_rate, data_AUC, data_ICE
        Each data_rate/data_AUC/data_ICE contains response, logMtot, lotStot, logItot
        Notice that for each data_rate/data_AUC/data_ICE, there may be one more datasets (noise for different dataset).
    prior_infor     : list of dict to assign prior distribution for kinetics parameters
    shared_params   : dict, information for shared parameters
    init_values     : dict, initial value for model fitting
    last_run_dir    : str, directory of the last running mcmc
    out_dir         : str, directory of the output 
    args            : dict, other model arguments
    ----------
    Fitting the Bayesian model to estimate the kinetics parameters and noise of all datasets
    Return mcmc.trace

    This function is modified from _model_fitting/_run_mcmc
    """
    rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))
    os.chdir(out_dir)
    traces_name = args.traces_name

    if not os.path.isfile(traces_name+'.pickle'):
        if not init_values is None:
            kernel = NUTS(model=global_fitting, init_strategy=init_to_value(values=init_values))
        else:
            kernel = NUTS(global_fitting)

        if os.path.isfile(os.path.join(last_run_dir, "Last_state.pickle")):
            last_state = pickle.load(open(os.path.join(last_run_dir, "Last_state.pickle"), "rb"))
            print("\nKeep running from last state.")
            mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
            mcmc.post_warmup_state = last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, experiments=expts, 
                     prior_infor=prior_infor, shared_params=shared_params, args=args)
        else:
            mcmc = MCMC(kernel, num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchain, progress_bar=True)
            mcmc.run(rng_key_, experiments=expts, prior_infor=prior_infor, shared_params=shared_params, args=args)
        
        mcmc.print_summary()

        print("Saving last state.")
        mcmc.post_warmup_state = mcmc.last_state
        pickle.dump(jax.device_get(mcmc.post_warmup_state), open("Last_state.pickle", "wb"))

        trace = mcmc.get_samples(group_by_chain=False)
        pickle.dump(trace, open(os.path.join(traces_name+'.pickle'), "wb"))

        ## Trace and autocorrelation plots
        plotting_trace(trace=trace, out_dir=out_dir, nchain=args.nchain)

        trace = mcmc.get_samples(group_by_chain=True)
        az.summary(trace).to_csv(traces_name+"_summary.csv")

        trace = mcmc.get_samples(group_by_chain=False)
    else:
        trace = pickle.load(open(traces_name+'.pickle', "rb"))

    return trace


def _CRC_check_noise(response, logItot, Z=2.5, plotting=False, scaling_plot=False, OUTFILE=''):
    """
    Parameters:
    ----------
    response  : np.array, response of the CRC dataset
    logItot   : np.array, log concentration of inhibitor
    Z         : integer, Z factor to detect outliers, default = 2.5
    plotting  : optional, boolean for plotting the figure
    OUTFILE   : optional, string, saving plot file
    ----------

    return [filtered_v, filtered_logItot, count_noisy_points] after outlier detection/removal
    """
    scaled_r = scaling_data(response, min(response), max(response))

    var = np.var(scaled_r)/len(scaled_r) #variance of the mean
    std = np.sqrt(var) #std of the mean

    # Calculated mean of each concentration and save the mean array as mean_r
    mean_r = []
    mean_response = []
    for logI in np.unique(logItot):
        mean_r.append(np.mean(scaled_r[logItot==logI]))
        mean_response.append(np.mean(response[logItot==logI]))
    mean_r = np.array(mean_r)
    mean_response = np.array(mean_response)

    filter_logI = []
    filter_r_scale = []
    filter_r = []
    outlier_pos = []

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for idx, logI in enumerate(np.unique(logItot)):
        r_each_logI = scaled_r[logItot==logI]
        for _r in r_each_logI:
            if _r>mean_r[idx]-Z*std and _r<mean_r[idx]+Z*std:
                filter_logI.append(logItot[scaled_r==_r])
                filter_r_scale.append(scaled_r[scaled_r==_r])
                filter_r.append(response[scaled_r==_r])
                outlier_pos.append(False)
            else:
                if plotting:
                    if scaling_plot:
                        ax.plot(np.log10(np.exp(logI)), _r, 'rx', label='Outlier', markersize=16)
                    else:
                        ax.plot(np.log10(np.exp(logI)), response[scaled_r==_r], 'rx', label='Outlier',  markersize=12)
                    handles, labels = ax.get_legend_handles_labels()
                outlier_pos.append(True)

        if plotting:
            if scaling_plot:
                ax.plot(np.log10(np.exp(logI)), mean_r[idx], "g^", label='Mean of each conc')
            else:
                ax.plot(np.log10(np.exp(logI)), mean_response[idx], "g^", label='Mean of each conc')
            handles, labels = ax.get_legend_handles_labels()

    if plotting:
        if scaling_plot:
            ax.plot(np.log10(np.exp(logItot)), scaled_r, 'b.', label='Observed data')
            ax.set_ylabel("% Activity")
        else:
            ax.plot(np.log10(np.exp(logItot)), response, 'b.', label='Observed data')
            ax.set_ylabel("Response")
        handles, labels = ax.get_legend_handles_labels()
        ax.set_xlabel("Log$_{10}$[I]")

        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.tight_layout()

    filter_logI = np.concatenate(filter_logI)
    filter_r_scale = np.concatenate(filter_r_scale)
    filter_r = np.concatenate(filter_r)

    outlier_pos = np.flip(np.array(outlier_pos))

    if np.sum(outlier_pos)>0 and len(OUTFILE)>0:
        plt.savefig(OUTFILE, bbox_inches='tight')

    return [filter_logI, filter_r, outlier_pos]


def _expt_check_noise_trend(expts, OUT_DIR=''):
    """
    Parameters:
    ----------
    expts     : list of experiment
    OUT_DIR   : optional, string, directory for saving plot file
    ----------

    Return a updated set of experiment similar to output of load_data_mers 
    after outlier detection and removal.
    """
    expts_update = []
    outliers = []
    mes_noise = []
    mes_trend = []
    for expt in expts:
        expt_update = {}
        for key in expt.keys():
            if not key in ['kinetics', 'AUC', 'ICE']:
                expt_update[key] = expt[key]
            else:
                data = expt[key]
                if data is not None:
                    if type(data) is dict:
                        data_update = {}
                        for i in range(len(data)):
                            [r, logMtot, logStot, logItot] = data[i]
                            if len(OUT_DIR)>0:
                                output = os.path.join(OUT_DIR, expt['index']+'_'+key+'_'+str(i))
                            else:
                                output = ''
                            [filter_logItot, filter_r, outlier_pos] = _CRC_check_noise(r, logItot, plotting=True, OUTFILE=output)
                            outliers.append(outlier_pos)

                            if np.sum(outlier_pos)==0:
                                data_update[i] = data[i]
                            else:
                                filter_logMtot = np.repeat(np.unique(logMtot), len(filter_logItot))
                                filter_logStot = np.repeat(np.unique(logStot), len(filter_logItot))
                                data_update[i] = [filter_r, filter_logMtot, filter_logStot, filter_logItot]
                        
                                percent_noise = np.sum(outlier_pos)/len(logItot)
                                mes_noise.append(_CRC_report_noise(percent_noise, expt['index']+'_'+key+'_'+str(i)))

                            mes = _CRC_report_trend(*_CRC_check_trend(filter_logItot, filter_r, scaling=True), "Curve "+expt['index']+'_'+key+'_'+str(i))
                            if len(mes)>0:
                                mes_trend.append(mes)

                        expt_update[key] = data_update
                        del data_update
                    
                    else:
                        data_update = []
                        [r, logMtot, logStot, logItot] = data
                        if len(OUT_DIR)>0:
                            output = os.path.join(OUT_DIR, expt['index']+'_'+key)
                        else:
                            output = ''
                        [filter_logItot, filter_r, outlier_pos] = _CRC_check_noise(r, logItot, plotting=True, OUTFILE=output)
                        outliers.append(outlier_pos)

                        if np.sum(outlier_pos)==0:
                            data_update = data
                        else:
                            filter_logMtot = np.repeat(np.unique(logMtot), len(filter_logItot))
                            filter_logStot = np.repeat(np.unique(logStot), len(filter_logItot))
                            data_update = [filter_r, filter_logMtot, logStot, filter_logItot]
                        
                            percent_noise = np.sum(outlier_pos)/len(logItot)
                            mes_noise.append(_CRC_report_noise(percent_noise))

                        mes = _CRC_report_trend(*_CRC_check_trend(filter_logItot, filter_r, scaling=True), "Curve")
                        if len(mes)>0:
                            mes_trend.append(mes)

                        expt_update[key] = data_update

        expts_update.append(expt_update)

    if len(mes_noise)==0:
        mes_noise = None

    if len(mes_trend)==0:
        mes_trend = None

    return [expts_update, outliers, mes_noise, mes_trend]


def _CRC_check_trend(logconcs, response, scaling=False):
    """
    Parameters:
    ----------
    logconcs  : np.array, log concentration of the CRC dataset
    response  : np.array, response of the CRC dataset
    scaling   : boolean, if True, convert the response to the %Activity or %Inhibition
    ----------
    Report if curve is upward (flag_up) or downward (flag_down)
    """
    if scaling:
        Rs = scaling_data(response, min(response), max(response))
    else: 
        Rs = response

    # Calculating mean of response (mean_Rs) for each log10 of concentration (xs):
    mean_Rs = []
    for (key, group) in itertools.groupby(sorted(list(zip(logconcs, Rs))), lambda x : x[0]):
        Rgs = np.array(list(group))
        mean_Rgs = np.mean(Rgs[:,1])
        mean_Rs.append(mean_Rgs)
    mean_Rs = np.array(mean_Rs)

    check_up = np.around(np.gradient(mean_Rs, mean_Rs.size), 1)>=0
    check_down = np.around(np.gradient(mean_Rs, mean_Rs.size), 1)<=0

    if sum(check_up)==len(mean_Rs) and sum(check_down)==0:
        flag_up = True
        flag_down = False
    elif sum(check_up)==0 and sum(check_down)==len(mean_Rs):
        flag_up = False
        flag_down = True
    else:
        flag_up = False
        flag_down = False
    
    return [flag_up, flag_down]


def _CRC_report_trend(flag_up, flag_downn, CRC_index='Curve'):
    """
    Parameters:
    ----------
    flag_up  : boolean, flag of upward curve
    flag_up  : boolean, flag of downward curve
    ----------
    Reported message if curve is upward (flag_up) or downward (flag_down)
    """
    if flag_up and (not flag_downn): 
        mes = CRC_index + ' is a upward curve.'
    elif (not flag_up) and flag_downn:
        mes = CRC_index + ' is a downward curve.'
    else:
        mes = ''
    return mes


def _CRC_report_noise(percent_noise, CRC_index=''):
    """
    Parameters:
    ----------
    percent_noise   : percentage of noise in the curve
    ----------
    Reported message if curve is noisy
    """
    if type(percent_noise) == np.ndarray:
        mes_noise = []
        for _percent in percent_noise:
            if _percent>0.5: 
                mes_noise.append(f"Should not use this results. More than {round(percent_noise*100,2)}% of the curve {CRC_index} was noisy.")
            elif _percent>0:
                mes_noise.append(f'{round(percent_noise*100,2)}% of the curve {CRC_index} was noisy.')
    else:
        if percent_noise>0.5: 
            mes_noise = f"Should not use this results. More than {round(percent_noise*100,2)}% of the curve {CRC_index} was noisy."
        elif percent_noise>0:
            mes_noise = f'{round(percent_noise*100,2)}% of the data was noisy.'
        else:
            mes_noise = None 

    return mes_noise
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd


def load_data_no_inhibitor(df, multi_var=False):
    """
    Parameters:
    ----------
    df          : pandas dataframe, each row containing the following information
        Plate           : plate name of experiment
        Enzyme (nM)     : enzyme concentration under nM
        Substrate (nM)  : substrate concentration under nM
        Inhibitor (nM)  : inhibitor concentration under nM
        v (nM.min^{-1}) : reaction rate under nM/min
    multi_var   : optional, boolean, return the output that can be used to fit multiple variances for each plate
    ----------
    
    Return the list of dict, each dict contain the information of experiment. 
    This function is used for no inhibitior datasets.
    """ 
    plate_list = np.unique(df['Plate'])
    multi_experiments = []
    experiment = []
    data_rate = {}
    for i, plate_i in enumerate(plate_list):
        kinetics_logMtot = []
        kinetics_logStot = []
        kinetics_logItot = []
        rate = []

        unique_enzyme = np.unique(df['Enzyme (nM)'])
        for n, conc_enzyme in enumerate(unique_enzyme):
            dat = df[(df['Enzyme (nM)']==conc_enzyme)*df['Plate']==plate_i]
            Stot = np.array(dat['Substrate (nM)'])*1E-9
            Mtot = np.ones(len(Stot))*conc_enzyme*1E-9
            v = np.array(dat['v (nM.min^{-1})'])*1E-9

            logMtot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Mtot])
            logStot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Stot])

            if len(np.unique(Stot))>1:
                experiment.append({'type':'kinetics', 'enzyme': 'mers', 'plate': 'ES',
                                   'figure': plate_i[:-5], 'sub_figure': f'E:{conc_enzyme}nM',
                                   'logMtot': logMtot, # M
                                   'logStot': logStot, # M
                                   'logItot': None, #np.array([np.log(1E-20)]*len(Mtot)), #None
                                   'v': v, # M min^{-1}
                                   'x':'logStot'})

                kinetics_logMtot.append(logMtot)
                kinetics_logStot.append(logStot)
                kinetics_logItot.append(np.array([np.log(1E-30)]*len(Mtot)))
                rate.append(v)

        if len(kinetics_logMtot)>1:
            kinetics_logMtot = np.concatenate(kinetics_logMtot)
            kinetics_logStot = np.concatenate(kinetics_logStot)
            kinetics_logItot = np.concatenate(kinetics_logItot)
            rate = np.concatenate(rate)
        elif len(kinetics_logMtot)==1:
            kinetics_logMtot = kinetics_logMtot[0]
            kinetics_logStot = kinetics_logStot[0]
            kinetics_logItot = kinetics_logItot[0]
            rate = rate[0]

        data_rate[i] = [np.array(rate), np.array(kinetics_logMtot), np.array(kinetics_logStot), None]

    multi_experiments.append({'enzyme': 'mers', 'figure': 'No Inhibitor', 'index': 'ES',
<<<<<<< HEAD
                              'plate' : np.repeat('ES', len(plate_list)), 
                              'kinetics': data_rate, 'AUC': None, 'ICE': None,
=======
                              'kinetics': data_rate, 'AUC': None, 'ICE': None
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
                              })
    if multi_var:
        return multi_experiments, experiment
    else:
        dat = df
        Mtot = np.array(dat['Enzyme (nM)'])*1E-9
        Stot = np.array(dat['Substrate (nM)'])*1E-9
        Itot = np.ones(len(Mtot))*1E-30
        rate = np.array(dat['v (nM.min^{-1})'])*1E-9

        logMtot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Mtot])
        logStot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Stot])
        data_rate = [rate, logMtot, logStot, None] #np.log(Itot)]

        one_experiment = []
        one_experiment.append({'enzyme': 'mers', 'figure': 'No Inhibitor', 'index': 'ES',
<<<<<<< HEAD
                               'plate' : 'ES',
=======
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
                               'kinetics': data_rate, 'AUC': None, 'ICE': None
                               })

        return one_experiment, experiment


def load_data_one_inhibitor(df, multi_var=False, name=None, min_points=8):
    """
    Parameters:
    ----------
    df          : pandas dataframe, each row containing the following information
        Plate           : plate name of experiment
        Enzyme (nM)     : enzyme concentration under nM
        Substrate (nM)  : substrate concentration under nM
        Inhibitor (nM)  : inhibitor concentration under nM
        v (nM.min^{-1}) : reaction rate under nM/min
    multi_var   : optional, boolean, return the output that can be used to fit multiple variances for each plate
    name        : optional, string, name of inihbitor
    min_points  : minimum data points required to fit the model
    ----------
    
    Return the list of dict, each dict contain the information of experiment. 
    This function is used for inhibitior datasets.
    """ 
    plate_list = np.unique(df['Plate'])
    multi_experiments = []
    experiment = []

    if name is None:
        try: name = np.unique(df['Inhibitor_ID'])[0]
        except: name = 'Inhibitor'

    data_rate = {}
    for i, plate_i in enumerate(plate_list):
        kinetics_logMtot = []
        kinetics_logStot = []
        kinetics_logItot = []
        rate = []

        unique_enzyme = np.unique(df['Enzyme (nM)'])
        for n, conc_enzyme in enumerate(unique_enzyme):
            dat = df[(df['Enzyme (nM)']==conc_enzyme)*df['Plate']==plate_i]
            
            # if len(dat)<min_points:
            #     print(f"There was only {len(dat)} data points.")
            #     data_rate[i] = [None, None, None, None]
            #     break
            
            Mtot = np.array(dat['Enzyme (nM)'])*1E-9
            Stot = np.array(dat['Substrate (nM)'])*1E-9
            Itot = np.array(dat['Inhibitor (nM)'])*1E-9
            v = np.array(dat['v (nM.min^{-1})'])*1E-9

            logMtot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Mtot])
            logStot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Stot])
            logItot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Itot])

            if len(np.unique(Itot))>1:
<<<<<<< HEAD
                experiment.append({'type':'kinetics', 'enzyme': 'mers', 'plate': plate_i,
                                   'index': name[7:12], #'ESI',
=======
                experiment.append({'type':'kinetics', 'enzyme': 'mers', 'index': name[7:12], #'ESI',
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
                                   'figure': name, 'sub_figure': f'E:{conc_enzyme}nM, S:{int(np.unique(Stot)[0]*1E9)}nM',
                                   'logMtot': logMtot, # M
                                   'logStot': logStot, # M
                                   'logItot': logItot, # M
                                   'v': v, # M min^{-1}
                                   'x':'logItot'})
                kinetics_logMtot.append(logMtot)
                kinetics_logStot.append(logStot)
                kinetics_logItot.append(logItot)
                rate.append(v)

            elif len(np.unique(Itot))==1 and len(np.unique(Stot))>1: # Fixed inhibitor (or no inhbitor)
<<<<<<< HEAD
                experiment.append({'type':'kinetics', 'enzyme':'mers', 'plate': plate_i,
                                   'index':name[7:12], #'index':'ESI',
=======
                experiment.append({'type':'kinetics', 'enzyme': 'mers', 'index':name[7:12], #'index':'ESI',
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
                                   'figure': plate_i[:-5], 'sub_figure': f'E:{conc_enzyme}nM',
                                   'logMtot': logMtot, # M
                                   'logStot': logStot, # M
                                   'logItot': logItot, #None
                                   'v': v, # M min^{-1}
                                   'x':'logStot'})
                kinetics_logMtot.append(logMtot)
                kinetics_logStot.append(logStot)
                kinetics_logItot.append(logItot)
                rate.append(v)

        if len(kinetics_logMtot)>1:
            kinetics_logMtot = np.concatenate(kinetics_logMtot)
            kinetics_logStot = np.concatenate(kinetics_logStot)
            kinetics_logItot = np.concatenate(kinetics_logItot)
            rate = np.concatenate(rate)
        elif len(kinetics_logMtot)==1:
            kinetics_logMtot = kinetics_logMtot[0]
            kinetics_logStot = kinetics_logStot[0]
            kinetics_logItot = kinetics_logItot[0]
            rate = rate[0]

        data_rate[i] = [rate, kinetics_logMtot, kinetics_logStot, kinetics_logItot]

    multi_experiments.append({'enzyme': 'mers', 'figure': name, 'index':name[7:12], #'index':'ESI',
<<<<<<< HEAD
                              'plate' : plate_list, 
=======
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
                              'kinetics': data_rate, 'AUC': None, 'ICE': None
                              })
    if multi_var:
        return multi_experiments, experiment
    else:
        dat = df
        Mtot = np.array(dat['Enzyme (nM)'])*1E-9
        Stot = np.array(dat['Substrate (nM)'])*1E-9
        Itot = np.array(dat['Inhibitor (nM)'])*1E-9
        rate = np.array(dat['v (nM.min^{-1})'])*1E-9

        logMtot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Mtot])
        logStot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Stot])
        logItot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Itot])

        data_rate = [rate, logMtot, logStot, logItot]
        one_experiment = []
<<<<<<< HEAD
        one_experiment.append({'enzyme': 'mers', 'index': name[7:12], #'index':'ESI',
                               'figure': name, 'plate' : name[7:12],
                               'kinetics': data_rate, 'AUC': None, 'ICE': None
=======
        one_experiment.append({'enzyme': 'mers', 'index':name[7:12], #'index':'ESI',
                               'figure': name,
                               'kinetics': data_rate,
                               'AUC': None, 'ICE': None
>>>>>>> e16ad6bbbb4c64bfe977436054fce8235c94dbc0
                               })

        return one_experiment, experiment
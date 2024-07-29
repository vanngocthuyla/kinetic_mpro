import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

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

    data_CRC = {}
    for i, plate_i in enumerate(plate_list):
        CRC_logMtot = []
        CRC_logStot = []
        CRC_logItot = []
        response = []

        unique_enzyme = np.unique(df['Enzyme (nM)'])
        for n, conc_enzyme in enumerate(unique_enzyme):
            dat = df[(df['Enzyme (nM)']==conc_enzyme)*df['Plate']==plate_i]
            
            # if len(dat)<min_points:
            #     print(f"There was only {len(dat)} data points.")
            #     data_CRC[i] = [None, None, None, None]
            #     break
            
            Mtot = np.array(dat['Enzyme (nM)'])*1E-9
            Stot = np.array(dat['Substrate (nM)'])*1E-9
            Itot = np.array(dat['Inhibitor (nM)'])*1E-9
            v = np.array(dat['v (nM.min^{-1})'])*1E-9

            logMtot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Mtot])
            logStot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Stot])
            logItot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Itot])

            if len(np.unique(Itot))>1:
                experiment.append({'type':'CRC', 'enzyme': 'mers', 'plate': plate_i,
                                   'figure': name, 'sub_figure': f'E:{conc_enzyme}nM, S:{int(np.unique(Stot)[0]*1E9)}nM',
                                   'logMtot': logMtot, # M
                                   'logStot': logStot, # M
                                   'logItot': logItot, # M
                                   'v': v, # M min^{-1}
                                   'x':'logItot'})
                CRC_logMtot.append(logMtot)
                CRC_logStot.append(logStot)
                CRC_logItot.append(logItot)
                response.append(v)

            elif len(np.unique(Itot))==1 and len(np.unique(Stot))>1: # Fixed inhibitor (or no inhbitor)
                experiment.append({'type':'CRC', 'enzyme':'mers', 'plate': plate_i,
                                   'figure': plate_i[:-5], 'sub_figure': f'E:{conc_enzyme}nM',
                                   'logMtot': logMtot, # M
                                   'logStot': logStot, # M
                                   'logItot': logItot, #None
                                   'v': v, # M min^{-1}
                                   'x':'logStot'})
                CRC_logMtot.append(logMtot)
                CRC_logStot.append(logStot)
                CRC_logItot.append(logItot)
                response.append(v)

        if len(CRC_logMtot)>1:
            CRC_logMtot = np.concatenate(CRC_logMtot)
            CRC_logStot = np.concatenate(CRC_logStot)
            CRC_logItot = np.concatenate(CRC_logItot)
            response = np.concatenate(response)
        elif len(CRC_logMtot)==1:
            CRC_logMtot = CRC_logMtot[0]
            CRC_logStot = CRC_logStot[0]
            CRC_logItot = CRC_logItot[0]
            response = response[0]

        data_CRC[i] = [response, CRC_logMtot, CRC_logStot, CRC_logItot]

    multi_experiments.append({'enzyme': 'mers', 'figure': name,
                              'plate' : plate_list, 
                              'CRC': data_CRC, 'kinetics': None, 'AUC': None, 'ICE': None
                              })
    if multi_var:
        return multi_experiments, experiment
    else:
        dat = df
        Mtot = np.array(dat['Enzyme (nM)'])*1E-9
        Stot = np.array(dat['Substrate (nM)'])*1E-9
        Itot = np.array(dat['Inhibitor (nM)'])*1E-9
        response = np.array(dat['v (nM.min^{-1})'])*1E-9

        logMtot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Mtot])
        logStot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Stot])
        logItot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Itot])

        data_CRC = [response, logMtot, logStot, logItot]
        one_experiment = []
        one_experiment.append({'enzyme': 'mers',
                               'figure': name, 'plate' : name,
                               'CRC': data_CRC, 'kinetics': None, 'AUC': None, 'ICE': None
                               })

        return one_experiment, experiment
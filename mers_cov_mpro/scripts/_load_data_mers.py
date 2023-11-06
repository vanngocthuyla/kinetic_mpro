import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd


def load_data_no_inhibitor(df, multi_var=False):

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
                experiment.append({'type':'kinetics', 'enzyme': 'mers',
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
                              'kinetics': data_rate, 'AUC': None, 'ICE': None
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
                               'kinetics': data_rate, 'AUC': None, 'ICE': None
                               })

        return one_experiment, experiment


def load_data_one_inhibitor(df, multi_var=False, name=None):

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
            Mtot = np.array(dat['Enzyme (nM)'])*1E-9
            Stot = np.array(dat['Substrate (nM)'])*1E-9
            Itot = np.array(dat['Inhibitor (nM)'])*1E-9
            v = np.array(dat['v (nM.min^{-1})'])*1E-9

            logMtot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Mtot])
            logStot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Stot])
            logItot = np.array([np.log(conc) if conc > 0 else np.log(1E-30) for conc in Itot])

            if len(np.unique(Itot))>1:
                experiment.append({'type':'kinetics', 'enzyme': 'mers', 'index': 'ESI',
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
                experiment.append({'type':'kinetics', 'enzyme': 'mers', 'index':name[8:12], #'index':'ESI',
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

    multi_experiments.append({'enzyme': 'mers', 'figure': name, 'index':name[8:12], #'index':'ESI',
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
        one_experiment.append({'enzyme': 'mers', 'index':name[8:12], #'index':'ESI',
                               'figure': name,
                               'kinetics': data_rate,
                               'AUC': None, 'ICE': None
                               })

        return one_experiment, experiment
import numpy as np
import jax
import jax.numpy as jnp

def load_data(fit_mutant_kinetics=True, fit_mutant_AUC=False, fit_mutant_ICE=False, 
              fit_wildtype_Nashed=False, fit_wildtype_Vuong=False, 
              fit_E_S=True, fit_E_I=True):

    experiments = []
    if fit_mutant_kinetics:
        if fit_E_S:
            # Fig 3a red
            experiments.append({'type':'kinetics',
                                'enzyme': 'mutant',
                                'figure':'Mut-3a',
                                'logMtot': np.log(np.array([6, 10, 25, 50, 75, 100, 120])*1E-6), # M
                                'logStot': np.array([np.log(200E-6)]*7), # 5 uL of 5mM substrate diluted with 95 uL reaction mixture,
                                'logItot': np.array([np.log(1E-20)]*7), #None
                                'v': np.array([0.0195, 0.024, 0.081, 0.36, 0.722, 1.13, 1.64])*1E-6, # M min^{-1}
                                'x':'logMtot'})

            # Fig 3b
            experiments.append({'type':'kinetics',
                                'figure':'Mut-3b',
                                'logMtot': np.array([np.log(40E-6)]*8), # M
                                'logStot': np.log(np.array([25, 50, 75, 100, 125, 150, 175, 200])*1E-6),
                                'logItot': np.array([np.log(1E-30)]*8), # None
                                'v': np.array([0.0488, 0.09, 0.127, 0.161, 0.192, 0.206, 0.223, 0.227])*1E-6, # M min^{-1}
                                'x':'logStot'})
            
        if fit_E_I:
            # Fig 5b 
            experiments.append({'type':'kinetics',
                                'enzyme': 'mutant',
                                'figure':'Mut-5b',
                                'logMtot': np.array([np.log(10E-6)]*12),  # M
                                'logStot': np.array([np.log(0.25E-3)]*12), #5 uL of 5mM substrate diluted with 95 uL reaction mixture,
                                'logItot': np.log(np.array([0.25, 0.5, 1, 1.75, 2.5, 5, 10, 16.6, 33.3, 66.6, 100, 133.3])*1E-6), 
                                'v': np.array([0.1204, 0.1862, 0.243, 0.3234, 0.3816, 0.4788, 0.5302, 0.3887, 0.2751, 0.1694, 0.129, 0.0947])*1E-6, # M min^{-1}
                                'x': 'logItot'})
            
        if fit_E_S and fit_E_I:
            # Fig 5c
            experiments.append({'type':'kinetics',
                                'enzyme': 'mutant',
                                'figure':'Mut-5c',
                                'logMtot': np.array([np.log(10E-6)]*6),
                                'logStot': np.log(np.array([96, 64, 48, 32, 16, 8])*1E-6),
                                'logItot': np.array([np.log(10E-6)]*6), 
                                'v': np.array([0.269, 0.189, 0.135, 0.097, 0.0395, 0.0229])*1E-6, # M min^{-1}
                                'x':'logStot'})

    if fit_mutant_AUC and fit_E_I:
        # Fig 6b
        experiments.append({'type':'AUC',
                            'enzyme': 'mutant',
                            'figure':'Mut-6b',
                            'logMtot': np.array([np.log(7E-6)]*8), # 6-7 micromolar
                            'logStot': np.array([np.log(1E-20)]*8), #None
                            'logItot': np.log(np.array([1, 3, 6, 10, 12.5, 15, 20, 50])*1E-6), 
                            'M': np.array([7, 6, 5, 3.25, 2.75, 2.55, 1.85, 0.8])*1E-6, # M min^{-1}
                            'x':'logItot'})

    if fit_mutant_ICE and fit_E_S and fit_E_I:
        # Fig 6d
        experiments.append({'type':'catalytic_efficiency',
                            'enzyme': 'mutant',
                            'figure':'Mut-6d',
                            'logMtot': np.array([np.log(10E-6)]*5),
                            'logStot': np.array([np.log(1E-20)]*5), # Not used
                            'logItot': np.log((np.array([0, 23.3, 56.6, 90, 123]) + 10)*1E-6),
                            'Km_over_kcat':np.array([2170, 7640, 17800, 29700, 36600])/10.,
                            'x':'logItot'})

    if fit_mutant_kinetics:
        # Collate all kinetics experiments
        kinetics_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'kinetics'])
        kinetics_logStot = np.hstack([experiment['logStot'] for experiment in experiments if experiment['type'] == 'kinetics'])
        kinetics_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'kinetics'])
        v = np.hstack([experiment['v'] for experiment in experiments if experiment['type'] == 'kinetics'])
        data_rate_mut = [v, kinetics_logMtot, kinetics_logStot, kinetics_logItot]
    else: 
        data_rate_mut = None

    if fit_mutant_AUC and fit_E_I:
        # Collate AUC experiments
        AUC_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'AUC'])
        AUC_logStot = np.hstack([experiment['logStot'] for experiment in experiments if experiment['type'] == 'AUC'])
        AUC_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'AUC'])
        auc = np.hstack([experiment['M'] for experiment in experiments if experiment['type'] == 'AUC'])
        data_auc_mut = [auc, AUC_logMtot, AUC_logStot, AUC_logItot]
    else:
        data_auc_mut = None

    if fit_mutant_ICE and fit_E_S and fit_E_I:
        # Collate inverse catalytic efficiency experiments
        ice_logMtot = np.hstack([experiment['logMtot'] for experiment in experiments if experiment['type'] == 'catalytic_efficiency'])
        ice_logStot = None
        ice_logItot = np.hstack([experiment['logItot'] for experiment in experiments if experiment['type'] == 'catalytic_efficiency'])
        ice = np.hstack([experiment['Km_over_kcat'] for experiment in experiments if experiment['type'] == 'catalytic_efficiency'])
        data_ice_mut = [ice, ice_logMtot, ice_logStot, ice_logItot]
    else:
        data_ice_mut = None

    # Fig S2a red WT ##Condition similar to 3a Mutation
    experiments_wt = []
    if fit_wildtype_Nashed and fit_E_S:
        experiments_wt.append({'type':'kinetics',
                            'enzyme': 'wild-type',
                            'figure':'WT-2a',
                            'logMtot': np.log(np.array([0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0938, 0.0625, 0.0469])*1E-6), # M
                            'logStot': np.array([np.log(200E-6)]*9), # 5 uL of 5mM substrate diluted with 95 uL reaction mixture,
                            'logItot': np.array([np.log(1E-30)]*9), #None
                            'v': np.array([6.71, 3.17, 1.99, 0.89, 0.58, 0.41, 0.195, 0.147, 0.072])*1E-6, # M min^{-1}
                            'x':'logMtot'})

        #Fig S2b WT ##Condition similar to 3b Mutation
        experiments_wt.append({'type':'kinetics',
                            'enzyme': 'wild-type',
                            'figure':'WT-2b',
                            'logMtot': np.array([np.log(200E-9)]*7), # M
                            'logStot': np.log(np.array([16, 32, 64, 77, 102.4, 128, 154])*1E-6), #M
                            'logItot': np.array([np.log(1E-30)]*7), # None
                            'v': np.array([0.285, 0.54, 0.942, 0.972, 1.098, 1.248, 1.338])*1E-6, # M min^{-1}
                            'x':'logStot'})

        #Fig S2c WT ##Condition similar to 3b Mutation
        experiments_wt.append({'type':'kinetics',
                            'enzyme': 'wild-type',
                            'figure':'WT-2c',
                            'logMtot': np.log(np.array([2, 1, 0.5, 0.25, 0.125, 0.0625, 0.0313])*1E-6), # M
                            'logStot': np.array([np.log(200E-6)]*7), # M
                            'logItot': np.array([np.log(1E-30)]*7), # None
                            'v': np.array([30, 11.493, 4.8215, 1.4366, 0.46799, 0.10725, 0.021973])*1E-6, # M min^{-1}
                            #'v/Mtot': np.array([15, 11.4931, 9.643, 5.7465, 3.7439, 1.716, 0.702]), # min^{-1}
                            'x':'logMtot'})

    if fit_wildtype_Vuong and fit_E_S and fit_E_I:
        # Fig 1b WT SARS-Covid-2 #Vuong et al
        experiments_wt.append({'type':'kinetics',
                            'enzyme': 'wild-type',
                            'figure':'WT-1b',
                            'logMtot': np.array([np.log(80E-9)]*24), # M
                            'logStot': np.array([np.log(100E-6)]*24),
                            'logItot': np.log(10**(np.array([-8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5, -8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5, -8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5]))), #M
                            # 'logItot': np.array([-8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5, -8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5, -8 , -7.5 , -7 , -6.5 , -6 , -5.5 , -5 , -4.5]), #M
                            'v': np.array([9.32 , 9.15 , 9.2 , 2.42 , 0.45 , 0.25 , 0.12 , 0.05, 8.03 , 8.29 , 6.26 , 0.51 , 0.02 , 0.045 , 0 , 0.1, 7.93 , 8.21 , 7.26 , 1.26 , 0.312 , 0.15 , 0.09 , 0.08])*1E-6, # M min^{-1}
                            'x':'logItot'})
        
    if fit_wildtype_Nashed or fit_wildtype_Vuong:
        if fit_E_I and not fit_E_S:
            data_rate_wt = None
        else:
            # Collate all kinetics experiments
            kinetics_logMtot_wt = np.hstack([experiment['logMtot'] for experiment in experiments_wt if experiment['type'] == 'kinetics'])
            kinetics_logStot_wt = np.hstack([experiment['logStot'] for experiment in experiments_wt if experiment['type'] == 'kinetics'])
            kinetics_logItot_wt = np.hstack([experiment['logItot'] for experiment in experiments_wt if experiment['type'] == 'kinetics'])
            v_wt = np.hstack([experiment['v'] for experiment in experiments_wt if experiment['type'] == 'kinetics'])
            data_rate_wt = [v_wt, kinetics_logMtot_wt, kinetics_logStot_wt, kinetics_logItot_wt]
    else:
        data_rate_wt = None

    experiments_multi_enzyme = []

    if fit_mutant_kinetics:
        experiments_multi_enzyme.append({'enzyme': 'mutant', 
                                        'kinetics': data_rate_mut, 'AUC': data_auc_mut, 'ICE': data_ice_mut
                                        })
    if fit_wildtype_Nashed or fit_wildtype_Vuong:
        if fit_E_S:
            experiments_multi_enzyme.append({'enzyme': 'wild_type', 
                                            'kinetics': data_rate_wt, 'AUC': None, 'ICE': None
                                            })

    return experiments_multi_enzyme, experiments, experiments_wt
import os
import pickle 


def save_model_setting(args, OUTDIR='', OUTFILE='setting.pickle'):
    """
    Parameters:
    ----------
    args    : arguments of model fitting
    OUTDIR  : optional, string, output directory for saving file
    OUTFILE : optional, string, output file name
    ----------
    Saving setting information under .pickle file
    """
    setting = {}
    setting['fit_E_S'] = args.fit_E_S
    setting['fit_E_I'] = args.fit_E_I
    setting['nburn']   = args.nburn
    setting['niters']  = args.niters
    setting['nchain']  = args.nchain
    setting['nthin']   = args.nthin
    setting['nthin']   = args.random_key
    
    setting['lnKd_min']     = getattr(args, 'lnKd_min', None)
    setting['lnKd_max']     = getattr(args, 'lnKd_max', None)
    setting['kcat_min']     = getattr(args, 'kcat_min', None)
    setting['kcat_max']     = getattr(args, 'kcat_max', None)
    setting['multi_alpha']  = getattr(args, 'multi_alpha', None)
    setting['alpha_min']    = getattr(args, 'alpha_min', None)
    setting['alpha_max']    = getattr(args, 'alpha_max', None)
    
    try:
        setting['set_lognormal_dE'] = args.set_lognormal_dE
        setting['dE'] = args.dE
    except AttributeError:
        pass
    
    try:
        setting['set_K_I_M_equal_K_S_M']        = args.set_K_I_M_equal_K_S_M
        setting['set_K_S_DS_equal_K_S_D']       = args.set_K_S_DS_equal_K_S_D
        setting['set_K_S_DI_equal_K_S_DS']      = args.set_K_S_DI_equal_K_S_DS
        setting['set_kcat_DSS_equal_kcat_DS']   = args.set_kcat_DSS_equal_kcat_DS
        setting['set_kcat_DSI_equal_kcat_DS']   = args.set_kcat_DSI_equal_kcat_DS
        setting['set_kcat_DSI_equal_kcat_DSS']  = args.set_kcat_DSI_equal_kcat_DSS
    except AttributeError:
        pass

    # Save the settings to a pickle file
    pickle.dump(setting, open(os.path.join(OUTDIR, OUTFILE), "wb"))
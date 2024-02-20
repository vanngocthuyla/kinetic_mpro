import os
import pickle 

def save_model_setting(args, OUTDIR='', OUTFILE='setting.pickle'):
    """
    Parameters:
    ----------
    args 	: arguments of model fitting
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
    
    setting['set_lognormal_dE']         = args.set_lognormal_dE
    setting['dE']                       = args.dE
    
    setting['set_K_S_DS_equal_K_S_D']   = args.set_K_S_DS_equal_K_S_D
    setting['set_K_S_DI_equal_K_S_DS']  = args.set_K_S_DI_equal_K_S_DS

    pickle.dump(setting, open(os.path.join(OUTDIR, OUTFILE), "wb"))
import numpy as np

def extract_samples_from_trace(trace, params, burn=0, thin=0):
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


def list_kinetics_params(init_params, update_params):
    results = init_params
    for name_params in update_params:
        results[name_params] = update_params[name_params]
    return results
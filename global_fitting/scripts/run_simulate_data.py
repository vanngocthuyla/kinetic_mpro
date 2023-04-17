import warnings
import numpy as np
import sys
import os
import argparse
import pickle

import matplotlib.pyplot as plt
from _data_simulation import kinetics_data_simulation
from _plotting import plot_kinetics_data

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument( "--out_dir",               type=str, 				default="")
parser.add_argument( "--type_experiment",		type=str, 				default="AUC")
parser.add_argument( "--log_sigma",				type=float,				default=-15.68)
parser.add_argument( "--number_of_experiments",	type=int,				default=1)
parser.add_argument( "--seed",					type=int,				default=0)

args = parser.parse_args()

experiments = []

logItot = np.log(np.array([1, 3, 6, 10, 12.5, 15, 20, 50])*1E-6)
n_concs = len(logItot)
logMtot = np.array([np.log(7E-6)]*n_concs)
logStot = np.array([np.log(1E-20)]*n_concs)

# Dimerization
logKd = -0.005 
# Substrate binding
logK_S_M = -6.015
logK_S_D = -9.312
logK_S_DS = -13.853
# Inhibitor binding
logK_I_M = 0
logK_I_D = -17.669
logK_I_DI = -18.141
# Binding of substrate and inhibitor
logK_S_DI = -13.967

kcat_MS = 0
kcat_DS = 0
kcat_DSI = 0.45
kcat_DSS = 0.38

params_logK = [logKd, logK_S_M, logK_S_D, logK_S_DS, logK_I_M, logK_I_D, logK_I_DI, logK_S_DI]
params_kcat = [kcat_MS, kcat_DS, kcat_DSI, kcat_DSS]
sigma = np.exp(args.log_sigma)

for n in range(args.number_of_experiments):
	data = kinetics_data_simulation(logMtot, logStot, logItot, params_logK, params_kcat,
	                                sigma, args.type_experiment, args.seed)
	experiments.append(data)

pickle.dump(experiments, open(os.path.join(args.out_dir, 'simulated_data.pickle'), "wb"))
plot_kinetics_data(experiments, params_logK, params_kcat, args.type_experiment)
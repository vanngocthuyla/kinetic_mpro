import sys
import os
import argparse

import numpy as np
import pickle

import jax
import jax.numpy as jnp

from _bf import bayes_factor_v1, bayes_factor_v2
from _bf_define_model import kinetic_model

parser = argparse.ArgumentParser()

parser.add_argument( "--data_file",                     type=str,               default="")

parser.add_argument( "--first_model_dir",               type=str,               default="")
parser.add_argument( "--second_model_dir",              type=str,               default="")
parser.add_argument( "--out_file",                      type=str,               default="")

parser.add_argument("--estimator_version",              type=int,               default=2)
parser.add_argument("--aug_with",                       type=str,               default="GaussMix") #"Normal", "Uniform", "GaussMix"
parser.add_argument("--n_components",                   type=int,               default=1)
parser.add_argument("--covariance_type",                type=str,               default="full") # 'full', 'tied', 'diag', 'spherical'
# read here for explanation: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

parser.add_argument("--sigma_robust",                   action="store_true",    default=False)
parser.add_argument("--bootstrap",                      type=int,               default=None)
parser.add_argument("--random_state",                   type=int,               default=0)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)

if args.estimator_version == 1:
    bayes_factor = bayes_factor_v1
elif args.estimator_version == 2:
    bayes_factor = bayes_factor_v2
else:
    raise ValueError("Unknown version: %d" % args.estimator_version)

def _load_model_information(model_dir):
    prior = pickle.load(open(os.path.join(model_dir, 'MAP_prior.pickle'), "rb"))
    trace = pickle.load(open(os.path.join(model_dir, 'MAP_trace.pickle'), "rb"))
    setting = pickle.load(open(os.path.join(model_dir, 'setting.pickle'), "rb"))

    return [prior, trace, setting]

data = pickle.load(open(os.path.join(args.data_file), "rb"))
first_model = kinetic_model(data, *_load_model_information(args.first_model_dir))
second_model = kinetic_model(data, *_load_model_information(args.second_model_dir))

first_sample = _load_model_information(args.first_model_dir)[1]
first_sample['logp'] = pickle.load(open(os.path.join(args.first_model_dir, 'log_probs.pickle'), "rb"))
second_sample = _load_model_information(args.second_model_dir)[1]
second_sample['logp'] = pickle.load(open(os.path.join(args.second_model_dir, 'log_probs.pickle'), "rb"))

bf, bf_err = bayes_factor(model_ini=first_model, sample_ini=first_sample, 
                          model_fin=second_model, sample_fin=second_sample, 
                          aug_with=args.aug_with, sigma_robust=args.sigma_robust, 
                          n_components=args.n_components, covariance_type=args.covariance_type,
                          bootstrap=args.bootstrap)

with open(args.out_file, "w") as f:
    if args.bootstrap is None:
        print("log10(bf) = %0.5f" % (bf*np.log10(np.e)), file=f)
    else:
        print("Running %d bootstraps to estimate error." % bootstrap, file=f)
        print("log10(bf) = %0.5f +/- %0.5f" % (bf * np.log10(np.e), bf_err * np.log10(np.e)), file=f)
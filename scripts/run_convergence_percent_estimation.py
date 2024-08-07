"""
calculate convergence of percentiles for parameters in traces
"""

import os
import argparse
import pickle

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_file",          type=str,           default="traces.pickle")
parser.add_argument( "--out_dir",            type=str,           default="")

parser.add_argument( "--vars",               type=str,           default="")
parser.add_argument( "--percentiles",        type=str,           default="5 25 50 75 95")
parser.add_argument( "--sample_proportions", type=str,           default="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
parser.add_argument( "--repeats",            type=int,           default=100)
parser.add_argument( "--random_state",       type=int,           default=0)

args = parser.parse_args()

def percentiles(x, q, nsamples, repeats):
    perce = []
    for _ in range(repeats):
        rnd_x = np.random.choice(x, size=nsamples, replace=True)
        p = np.percentile(rnd_x, q)
        perce.append(p)

    perce = np.array(perce)
    p_mean = perce.mean(axis=0)
    p_err = perce.std(axis=0)

    return p_mean, p_err


def print_percentiles(p_mean, p_err):
    if isinstance(p_mean, float) and isinstance(p_err, float):
        return "%12.5f%12.5f" % (p_mean, p_err)
    else:
        p_str = "".join(["%12.5f%12.5f" % (p_m, p_e) for p_m, p_e in zip(p_mean, p_err)])
        return p_str

if os.path.isfile(args.mcmc_file):
    print("Loading " + args.mcmc_file)
    sample = pickle.load(open(args.mcmc_file, 'rb'))
else:
    print(args.mcmc_file, "doesn't exist.")

np.random.seed(args.random_state)

qs = [float(s) for s in args.percentiles.split()]
qs_str = "".join(["%10.1f-th %10.1f-error " % (q, q) for q in qs])
print("qs:", qs_str)

sample_proportions = [float(s) for s in args.sample_proportions.split()]
print("sample_proportions:", sample_proportions)

if len(args.vars)>0:
    vars = args.vars.split()
else:
    vars = list(sample.keys())
print("vars:", vars)

if len(args.out_dir)>0:
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)
	os.chdir(args.out_dir)
else:
	if not os.path.exists("Convergence"):
		os.mkdir("Convergence")
	os.chdir("Convergence")

all_vars = sample.keys()

for v in vars:
    if v not in all_vars:
        raise KeyError(v + " not a valid var name.")

for var in vars:

    x = sample[var]
    nsamples = len(x)

    out_file_handle = open(var + ".dat", "w")
    out_file_handle.write("proportion   nsamples" + qs_str + "\n")

    for samp_pro in sample_proportions:
        nsamp_pro = int(nsamples * samp_pro)
        p_mean, p_err = percentiles(x, qs, nsamp_pro, args.repeats)

        out_str = "%10.5f%10d" % (samp_pro, nsamp_pro) + print_percentiles(p_mean, p_err) + "\n"

        out_file_handle.write(out_str)

    out_file_handle.close()

print("DONE")
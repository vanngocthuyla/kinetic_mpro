"""
to plot convergence curve for the percentiles of posteriors
"""

import argparse
import os
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_file",      type=str,               default="traces.pickle")
parser.add_argument( "--data_dir",       type=str,               default="")
parser.add_argument( "--out_dir",        type=str,               default="")

parser.add_argument( "--combine_plot",   action="store_true",    default=False)

parser.add_argument( "--vars",           type=str,               default="")
parser.add_argument( "--percentiles",    type=str,               default="5 25 50 75 95")
parser.add_argument( "--xlabel",         type=str,               default="Sample proportion")

parser.add_argument( "--font_scale",     type=float,             default=0.75)

args = parser.parse_args()

sns.set(font_scale=args.font_scale)

if len(args.vars)>0:
    vars = args.vars.split()
else:
    if os.path.exists(args.mcmc_file):
        sample = pickle.load(open(args.mcmc_file, 'rb'))
        vars = list(sample.keys())
    else:
        print("Please provide list or variables or MCMC file.")
print("vars:", vars)

ylabels = {}
ylabels["logKd:0"]                = "log$K_d$ (M) - $MPro^{Mut}$"
ylabels["logKd:1"]                = "log$K_d$ (M) - $MPro^{WT}$"
ylabels["logK_S_M"]               = "log$K_{S,M}$ (M)"
ylabels["logK_S_D"]               = "log$K_{S,D}$ (M)"
ylabels["logK_S_DS"]              = "log$K_{S,DS}$ (M)"
ylabels["logK_I_M"]               = "log$K_{I,M}$ (M)"
ylabels["logK_I_D"]               = "log$K_{I,D}$ (M)"
ylabels["logK_I_DI"]              = "log$K_{I,DI}$ (M)"
ylabels["logK_S_DI"]              = "log$K_{S,DI}$ (M)"

ylabels["kcat_DS:0"]              = "$kcat_{DS}$ $(min^{-1})$ - $MPro^{Mut}$"
ylabels["kcat_DS:1"]              = "$kcat_{DS}$ $(min^{-1})$ - $MPro^{WT}$"
ylabels["kcat_DSI:0"]             = "$kcat_{DSI}$ $(min^{-1})$ - $MPro^{Mut}$"
ylabels["kcat_DSI:1"]             = "$kcat_{DSI}$ $(min^{-1})$ - $MPro^{WT}$"
ylabels["kcat_DSS:0"]             = "$kcat_{DSS}$ $(min^{-1})$ - $MPro^{Mut}$"
ylabels["kcat_DSS:1"]             = "$kcat_{DSS}$ $(min^{-1})$ - $MPro^{WT}$"

ylabels["log_sigma_rate:mut"]     = "log$\sigma$ - Kinetic $MPro^{Mut}$"
ylabels["log_sigma_auc:mut"]      = "log$\sigma$ - AUC $MPro^{Mut}$"
ylabels["log_sigma_ice:mut"]      = "log$\sigma$ - ICE $MPro^{Mut}$"
ylabels["log_sigma_rate:wt_1:0"]  = "log$\sigma$ - Kinetic $MPro^{WT,0}$"
ylabels["log_sigma_rate:wt_1:1"]  = "log$\sigma$ - Kinetic $MPro^{WT,1}$"

xlabel = args.xlabel

qs = [float(s) for s in args.percentiles.split()]
data_cols = ["%0.1f-th" % q for q in qs]
print("data_cols:", data_cols)
err_cols = ["%0.1f-error" % q for q in qs]
print("err_cols:", err_cols)
legends = ["%d-th" % q for q in qs]

colors = ["b", "g", "r", "c", "m"]
#line_styles = ["solid", "dotted", "dashed", "dashdot", "solid"]
line_styles = ["solid" for _ in range(5)]
markers = ["o", "s", "d", "^", "v"]

if len(args.out_dir)>0:
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    os.chdir(args.out_dir)
else:
    if not os.path.exists("Plot"):
        os.mkdir("Plot")
    os.chdir("Plot")

def plot_convergence_posterior(var, inp_file, xlabel, ylabel,
                               percentiles="5 25 50 75 95", colors=["b", "g", "r", "c", "m"],
                               line_styles=["solid" for _ in range(5)], markers = ["o", "s", "d", "^", "v"],
                               figure_size=(3.2, 2.2), dpi=80, out_file=None, ax=None):

    qs = [float(s) for s in percentiles.split()]
    data_cols = ["%0.1f-th" % q for q in qs]
    err_cols  = ["%0.1f-error" % q for q in qs]
    legends   = ["%d-th" % q for q in qs]

    if ax is None:
        plt.figure(figsize=figure_size)
        ax = plt.axes()

    data = pd.read_csv(inp_file, sep="\s+")
    x = data["proportion"]

    for i, data_col in enumerate(data_cols):
        err_col = err_cols[i]
        color = colors[i]
        line_style = line_styles[i]
        marker = markers[i]
        legend = legends[i]

        y = data[data_col]
        yerr = data[err_col]

        ax.errorbar(x, y, yerr=yerr, linestyle=line_style, c=color, marker=marker, markersize=5, label=legend)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file, dpi=dpi)
    else:
        return ax

if not args.combine_plot:
    for var in vars:
        if var in ylabels:
            ylabel = ylabels[var]
        else:
            ylabel = var
        inp_file = os.path.join(args.data_dir, var + ".dat")
        plot_convergence_posterior(var, inp_file, xlabel, ylabel,
                                   out_file=os.path.join(args.out_dir, var))
                                  #  out_file=os.path.join(args.out_dir, var+'.pdf'))
else:
    nrow = int((len(vars)+1)/2)

    fig, axes = plt.subplots(ncols=2, nrows=nrow, sharex=True, figsize=(6.4, 2.2*nrow))
    plt.subplots_adjust(wspace=0., hspace=0.)
    axes = axes.flatten()

    for var, ax in zip(vars, axes):
        if var in ylabels:
            ylabel = ylabels[var]
        else:
            ylabel = var
        inp_file = os.path.join(args.data_dir, var + ".dat")
        data = pd.read_csv(inp_file, sep="\s+")
        x = data["proportion"]

        for i, data_col in enumerate(data_cols):
            err_col = err_cols[i]
            color = colors[i]
            line_style = line_styles[i]
            marker = markers[i]
            legend = legends[i]

            y = data[data_col]
            yerr = data[err_col]

            ax.errorbar(x, y, yerr=yerr, linestyle=line_style, c=color, marker=marker, markersize=5, label=legend)

        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel(xlabel)
    axes[-2].set_xlabel(xlabel)

    if (len(vars)%2)==1:
        axes[len(vars)].axis('off')

    fig.tight_layout()
    fig.savefig('Convergence Plot.pdf', dpi=300)

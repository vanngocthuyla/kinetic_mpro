import warnings
import numpy as np
import sys
import os
import itertools
import argparse

import pickle
import arviz as az
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _pIC50 import _correct_ID
from _pIC50_correlation import _df_pIC50_pIC90, corr_pearsonr_N_sample, corr_bootstrap_matrix, corr_leave_p_out_matrix

parser = argparse.ArgumentParser()

parser.add_argument( "--out_dir",                       type=str,               default="")

parser.add_argument( "--cellular_pIC50_file",           type=str,               default="")
parser.add_argument( "--control_pIC50_file",            type=str,               default="")
parser.add_argument( "--inhibit_pIC50_file",            type=str,               default="")
parser.add_argument( "--dimer_pIC50_file",              type=str,               default="")

parser.add_argument( "--include_experiments",           type=str,               default="")
parser.add_argument( "--exclude_experiments",           type=str,               default="")

parser.add_argument( "--bootstrapping",                 action="store_true",    default=False)
parser.add_argument( "--leave_p_out_CV",                action="store_true",    default=False)
parser.add_argument( "--p_out_CV",                      type=int,               default="10")

args = parser.parse_args()

## Extracting cellular pIC50 -----------------------------------------------------------------------------------------
print(f"Extracting cellular pIC values from {args.cellular_pIC50_file}")
_df_cell = pd.read_csv(args.cellular_pIC50_file)
_df_cell = _df_cell.rename(columns={'Molecule Name': "ID", 'IC50 (uM)': 'IC50_uM', 'Hill': 'neg_hill'})
_df_cell.insert(len(_df_cell.columns), 'hill', -_df_cell['neg_hill'])
df_cell = _df_pIC50_pIC90(_df_cell)
df_cell = df_cell.rename(columns={'pIC50': 'cell_pIC50', 'pIC90': 'cell_pIC90'})


## Extracting control pIC50 --------------------------------------------------------------------------------------
print(f"Extracting cellular pIC values from {args.control_pIC50_file}")
_df_ctrl = pd.read_csv(args.control_pIC50_file)
_df_ctrl = _df_ctrl.rename(columns={'Molecule Name': "ID", 'IC50 (uM)': 'IC50_uM', 'Hill': 'hill'})
df_ctrl = _df_pIC50_pIC90(_df_ctrl)
df_ctrl = df_ctrl.rename(columns={'pIC50': 'control_pIC50', 'pIC90': 'control_pIC90'})


## Extracting inhibition pIC50 --------------------------------------------------------------------------------------
print(f"Extracting cellular pIC values from {args.inhibit_pIC50_file}")
_df_inhib = pd.read_csv(args.inhibit_pIC50_file)
_df_inhib = _df_inhib.rename(columns={'Molecule Name': "ID", 'Hill': 'hill'})
df_inhib = _df_pIC50_pIC90(_df_inhib)
df_inhib = df_inhib.rename(columns={'pIC50': 'inhibit_pIC50', 'pIC90': 'inhibit_pIC90'})


## Extracting dimer-only pIC50 --------------------------------------------------------------------------------------
print(f"Extracting dimer-only pIC values from {args.dimer_pIC50_file}")
_df_dimer = pd.read_csv(args.dimer_pIC50_file)
_df_dimer = _df_dimer.rename({"Unnamed: 0": "ID"}, axis=1)
_df_dimer = _df_dimer[['ID', 'pIC50', 'pIC50_std', 'pIC90', 'pIC90_std', 'hill']]
df_dimer = _correct_ID(_df_dimer, 'drop')
df_dimer = df_dimer.rename(columns={'pIC50': 'dimer_pIC50', 'pIC90': 'dimer_pIC90'})


## Combining data for the correlogram -------------------------------------------------------------------------------
dat = pd.merge(df_inhib[['ID', 'inhibit_pIC50', 'inhibit_pIC90']], df_dimer, on='ID', how='inner')
dat = pd.merge(dat, df_ctrl[['ID', 'control_pIC50', 'control_pIC90']], on='ID', how='inner')
dat = pd.merge(dat, df_cell[['ID', 'cell_pIC50', 'cell_pIC90']], on='ID', how='inner')

_inhibitor_list = args.include_experiments.split()
exclude_experiments = args.exclude_experiments.split()
if len(_inhibitor_list)>0:
    inhibitor_list = [name for name in _inhibitor_list if (name not in exclude_experiments)]
    dat_expts = pd.DataFrame([inhibitor_list], ['ID']).T
    dat = pd.merge(dat, dat_expts, on='ID', how='inner')

assay_name_50 = ['Inhibition pIC50', 'Control pIC50', 'Dimer-only pIC50', 'Cellular pEC50']
assay_name_90 = ['Inhibition pIC90', 'Control pIC90', 'Dimer-only pIC90', 'Cellular pEC90']
IC50_keys = ['inhibit_pIC50', 'control_pIC50', 'dimer_pIC50', 'cell_pIC50']
IC90_keys = ['inhibit_pIC90', 'control_pIC90', 'dimer_pIC90', 'cell_pIC90']
change_names_50 = dict(([i, assay_name_50[i]]) for i in range(4))
change_names_90 = dict(([i, assay_name_90[i]]) for i in range(4))

frameon=False
tick_size = 30

#Set the white background, Zoom the font scale
sns.set(style="white", font_scale=2.5)

for keys, assay_name, change_names in zip([IC50_keys, IC90_keys], [assay_name_50, assay_name_90], [change_names_50,change_names_90]):
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(18, 16), constrained_layout=True)
    # fig.suptitle('Correlogram of pIC50 - Fitting Without Control', fontsize = 'x-large', fontweight = 'bold')
    for i in range(0,n):
        for j in range(0, n):
            if i>j:
                sns.scatterplot(ax=axes[i, j],
                                data=dat[[keys[i], keys[j]]],
                                x=keys[j], y=keys[i], palette=['blue'], legend=True, s=60)
                # corr_pearsonr_N_sample(dat[keys[i]], dat[keys[j]], ax=axes[i, j])
                # handles, labels = axes[i, j].get_legend_handles_labels()
                # axes[i, j].get_legend().remove()
                axes[i, j].set_xlim(4, 8)
                axes[i, j].set_ylim(4, 8)
                axes[i, j].set(xticks=[4, 5, 6, 7, 8], yticks=[4, 5, 6, 7, 8])
                axes[i, j].grid(True)
                axes[i, j].tick_params(axis='x', labelsize=tick_size)
                axes[i, j].tick_params(axis='y', labelsize=tick_size)
                if j == 0:
                    plt.setp(axes[i, j], xlabel='')
                    plt.setp(axes[i, j], ylabel=assay_name[i])
                if i == (n-1):
                    plt.setp(axes[i, j], xlabel=assay_name[j])
                    plt.setp(axes[i, j], ylabel='')
                if j == 0 and i==(n-1):
                    plt.setp(axes[i, j], ylabel=assay_name[i])
                    plt.setp(axes[i, j], xlabel=assay_name[j])
                if j != 0 and i !=(n-1):
                    plt.setp(axes[i, j], xlabel='')
                    plt.setp(axes[i, j], ylabel='')
            else:
                fig.delaxes(axes[i][j])

    # by_label = dict(zip(labels, handles))
    # fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.65, 0.65), title='pIC50')
    plt.savefig(os.path.join(args.out_dir, f"Correlogram_{keys[i][-2:]}.jpg"), bbox_inches='tight', dpi=800);

    if args.bootstrapping:
        if not os.path.isdir(os.path.join(args.out_dir, "Bootstrap")):
            os.mkdir(os.path.join(args.out_dir, "Bootstrap"))
        os.chdir(os.path.join(args.out_dir, "Bootstrap"))

        for _method in ['pearsonr', 'spearmanr', 'kendall', 'RMSD', 'aRMSD']:
            table = corr_bootstrap_matrix(dat, keys, method=_method)
            table = table.rename(index=change_names)
            table.to_csv(f'pIC{keys[i][-2:]}_{_method}.csv')

    if args.leave_p_out_CV and args.p_out_CV>0: 
        if not os.path.isdir(os.path.join(args.out_dir, "LpOCV")):
            os.mkdir(os.path.join(args.out_dir, "LpOCV"))
        os.chdir(os.path.join(args.out_dir, "LpOCV"))

        for _method in ['pearsonr', 'spearmanr', 'kendall', 'RMSD', 'aRMSD']:
            table = corr_leave_p_out_matrix(dat, keys, method=_method)
            table = table.rename(index=change_names)
            table.to_csv(f'pIC{keys[i][-2:]}_{_method}.csv')
import ast
import configparser
import csv
import os
import sys
from itertools import cycle
from glob import glob

import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn as sns
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set_style('whitegrid')

if not (len(sys.argv) == 3 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file] [survival estimators file]'
          % sys.argv[0])
    sys.exit()

config = configparser.ConfigParser()
config.read(sys.argv[1])
n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
coverage_range = [0.7, 0.8, 0.9, 0.95]
output_dir = config['DEFAULT']['output_dir']
os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

assert len(datasets) == 3  # this script is hard-coded to work with 3 plots

survival_estimator_names = []
estimator_display_names = []
with open(sys.argv[2], 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if not line.startswith('#'):
            pieces = line.split(' : ')
            if len(pieces) == 2:
                survival_estimator_names.append(pieces[0])
                estimator_display_names.append(pieces[1])

# method for translating survival curve into single survival time estimate
survival_time_method = 'median'

figsize = (9, 3.8)

n_calib_fracs = 10
n_estimators = len(estimator_display_names)
n_datasets = len(datasets)

np.random.seed(0)
colors = pl.cm.tab20(np.linspace(0, 1, n_estimators))[
    np.random.permutation(n_estimators)]
for target_coverage in coverage_range:
    plt.figure(figsize=figsize)
    for dataset_idx, dataset in enumerate(datasets):
        centers = np.zeros((n_estimators, n_calib_fracs))
        spreads = np.zeros((n_estimators, n_calib_fracs))
        for estimator_idx, survival_estimator_name \
                in enumerate(survival_estimator_names):
            for calib_idx, calib_frac \
                    in enumerate(np.linspace(0.1, 1, n_calib_fracs)):
                matches = list(glob(
                    os.path.join(
                        output_dir,
                        'split_conformal_prediction',
                        '%s_%s*_%s_calib%f_coverages.txt'
                        % (survival_estimator_name, dataset,
                           survival_time_method, calib_frac))))
                if len(matches) != 1:
                    print("\n".join(matches))
                assert len(matches) == 1
                match = matches[0]
                coverages = np.loadtxt(match)
                for row in coverages:
                    if row[0] == target_coverage:
                        centers[estimator_idx, calib_idx] \
                            = np.mean(row[1:])
                        spreads[estimator_idx, calib_idx] \
                            = np.std(row[1:])

        lines = ["-", "--", "-.", ":"]
        linecycler = cycle(lines)

        ax = plt.subplot(1, n_datasets, dataset_idx + 1)
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        for x, xerr, name, color \
                in zip(centers, spreads, estimator_display_names, colors):
            plt.plot(np.linspace(0.1, 1, n_calib_fracs), x,
                     linestyle=next(linecycler),
                     label=name, color=color)
            plt.fill_between(np.linspace(0.1, 1, n_calib_fracs),
                             x - xerr, x + xerr, alpha=0.05,
                             facecolor=color)

        if dataset_idx == n_datasets - 1:
            lgd = plt.legend(loc='center left', frameon=True, bbox_to_anchor=(1, 0.5))
        if dataset_idx == 0:
            plt.ylabel("Empirical coverage (mean)")
        if dataset == 'metabric':
            plt.title('METABRIC', fontsize='medium')
        elif dataset == 'rotterdam-gbsg2':
            plt.title('Rotterdam/GBSG', fontsize='medium')
        elif dataset == 'support2_onehot':
            plt.title('SUPPORT', fontsize='medium')
        if dataset_idx == (n_datasets // 2):
            plt.xlabel('Fraction of calibration data used to construct '
                       'marginal prediction intervals (target coverage %.1f%%)'
                       % (target_coverage * 100))
    plt.tight_layout()
    plt.subplots_adjust(wspace=.3)
    plt.savefig(os.path.join(output_dir, 'plots',
                             'emp_coverage_vs_calib_frac_target%f_'
                             % target_coverage
                             +
                             'marginal_%s_survival_time.pdf'
                             % survival_time_method),
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()

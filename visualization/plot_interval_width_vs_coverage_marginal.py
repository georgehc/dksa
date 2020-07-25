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

experiment_idx = 0

config = configparser.ConfigParser()
config.read(sys.argv[1])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
coverage_range = \
    ast.literal_eval(config['DEFAULT']['conformal_prediction_CI_coverage'])
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

# use all calibration data
calib_frac = 1.0

figsize = (9, 10.5)

n_coverages = len(coverage_range)
n_estimators = len(estimator_display_names)
n_datasets = len(datasets)

np.random.seed(0)
colors = pl.cm.tab20(np.linspace(0, 1, n_estimators))[
    np.random.permutation(n_estimators)]

plt.figure(figsize=figsize)
for dataset_idx, dataset in enumerate(datasets):
    centers = np.zeros((n_estimators, n_coverages))
    spreads = np.zeros((n_estimators, n_coverages))
    for coverage_idx, target_coverage in enumerate(coverage_range):
        for estimator_idx, survival_estimator_name \
                in enumerate(survival_estimator_names):
            matches = list(glob(
                os.path.join(
                    output_dir,
                    'split_conformal_prediction',
                    '%s_%s_exp%d_*%s_calib%f_qhats.txt'
                    % (survival_estimator_name, dataset, experiment_idx,
                       survival_time_method, calib_frac))))
            if len(matches) != 1:
                print("\n".join(matches))
            assert len(matches) == 1
            match = matches[0]
            coverages = np.loadtxt(match)
            for row in coverages:
                if row[0] == target_coverage:
                    if dataset == 'support2_onehot':
                        centers[estimator_idx, coverage_idx] \
                            = np.mean(row[1:]) / 30.42 * 2
                        spreads[estimator_idx, coverage_idx] \
                            = np.std(row[1:]) / 30.42 * 2
                    else:
                        centers[estimator_idx, coverage_idx] \
                            = np.mean(row[1:]) * 2
                        spreads[estimator_idx, coverage_idx] \
                            = np.std(row[1:]) * 2

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

    if dataset_idx == 0:
        ax = plt.subplot2grid((6, 6), [0, 1], 3, 3)
    elif dataset_idx == 1:
        ax = plt.subplot2grid((6, 6), [3, 0], 3, 3)
    elif dataset_idx == 2:
        ax = plt.subplot2grid((6, 6), [3, 3], 3, 3)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    for x, xerr, name, color \
            in zip(centers, spreads, estimator_display_names, colors):
        plt.plot(coverage_range, x,
                 linestyle=next(linecycler),
                 label=name, color=color)
        plt.fill_between(coverage_range,
                         x - xerr, x + xerr, alpha=0.2,
                         facecolor=color)

    if dataset_idx == 0:
        plt.legend(loc='center left', frameon=True, bbox_to_anchor=(1, 0.5))
    if dataset == 'metabric':
        plt.title('METABRIC', fontsize='medium')
    elif dataset == 'rotterdam-gbsg2':
        plt.title('Rotterdam/GBSG', fontsize='medium')
    elif dataset == 'support2_onehot':
        plt.title('SUPPORT', fontsize='medium')
    plt.xlabel("Target coverage level $1-\\alpha$")
    plt.ylabel("Prediction interval width (months)")

plt.tight_layout()
plt.subplots_adjust(wspace=.9, hspace=.7)
plt.savefig(os.path.join(output_dir, 'plots',
                         'interval_width_vs_coverage_'
                         +
                         'marginal_mean_std_%s_survival_time.pdf'
                         % survival_time_method),
            bbox_inches='tight')
# plt.show()

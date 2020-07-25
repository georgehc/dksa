
import ast
import configparser
import os
import sys
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
# coverage_range = [0.6, 0.7, 0.8, 0.9, 0.95]
output_dir = config['DEFAULT']['output_dir']
coverage_range = np.array(coverage_range)
coverage_range = coverage_range[coverage_range >= 0.6]

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

figsize = (9, 5)

def fix_estimator_names(estimator_display_names):
    new_estimator_display_names = []
    for estimator_display_name in estimator_display_names:
        if estimator_display_name == 'NKS-MLP':
            new_estimator_display_names.append('NKS-MLP (init: standard)')
        else:
            new_estimator_display_names.append(estimator_display_name)
    return new_estimator_display_names

n_estimators = len(survival_estimator_names)
n_datasets = len(datasets)
plt.figure(figsize=figsize)
for target_coverage in coverage_range:
    plt.figure(figsize=figsize)
    for dataset_idx, dataset in enumerate(datasets):
        all_data = []
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
                        all_data.append(row[1:] / 30.42 * 2)
                    else:
                        all_data.append(row[1:] * 2)

        plt.subplot(1, n_datasets, dataset_idx + 1)
        plt.violinplot(all_data[::-1], vert=False)
        if dataset_idx == 0:
            plt.yticks(range(1, n_estimators + 1),
                       fix_estimator_names(estimator_display_names[::-1]))
        else:
            plt.yticks(range(1, n_estimators + 1), [''] * n_estimators)
        if dataset == 'metabric':
            plt.title('METABRIC', fontsize='medium')
        elif dataset == 'rotterdam-gbsg2':
            plt.title('Rotterdam/GBSG', fontsize='medium')
        elif dataset == 'support2_onehot':
            plt.title('SUPPORT', fontsize='medium')
        if dataset_idx == (n_datasets // 2):
            plt.xlabel('Marginal prediction interval width (months) for '
                       'target coverage %0.1f%%' % (target_coverage * 100))
    plt.tight_layout()
    plt.subplots_adjust(wspace=.05)
    plt.savefig(os.path.join(output_dir, 'plots',
                             'interval_width_vs_methods_coverage%f.pdf'
                             % target_coverage),
                bbox_inches='tight')

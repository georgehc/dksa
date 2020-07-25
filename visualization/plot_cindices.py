#!/usr/bin/env python
import ast
import configparser
import csv
import os
import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
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
output_dir = config['DEFAULT']['output_dir']
os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

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

csv_filenames = []
for survival_estimator_name in survival_estimator_names:
    if survival_estimator_name == 'coxph':
        table_filename = \
            os.path.join(output_dir,
                         '%s_experiments%d_test_metrics_bootstrap.csv'
                         % (survival_estimator_name,
                            n_experiment_repeats))
    else:
        table_filename = \
            os.path.join(output_dir,
                         '%s_experiments%d_cv%d_test_metrics_bootstrap.csv'
                         % (survival_estimator_name,
                            n_experiment_repeats,
                            cross_val_n_folds))
    if not os.path.isfile(table_filename):
        print('File not found:', table_filename)
    assert os.path.isfile(table_filename)
    csv_filenames.append(table_filename)

results = {}

for filename in csv_filenames:
    with open(filename, 'r') as f:
        csv_file = csv.reader(f)
        header = True
        for row in csv_file:
            if header:
                assert row[0] == 'dataset'
                header = False
                continue

            dataset = row[0]
            experiment_idx = int(row[1])
            if experiment_idx > 0:
                continue
            estimator_display_name = row[2]

            cindex_td = float(row[3])
            cindex_td_CI_lower = float(row[4])
            cindex_td_CI_upper = float(row[5])
            cindex_td_CI_mean = float(row[6])

            metrics = [cindex_td,
                       cindex_td_CI_lower,
                       cindex_td_CI_upper,
                       cindex_td_CI_mean]

            key = (dataset, estimator_display_name)
            if key not in results:
                results[key] = [metrics]
            else:
                results[key].append(metrics)

for key in results:
    results[key] = np.array(results[key])

figsize = (9, 5)

def fix_estimator_names(estimator_display_names):
    new_estimator_display_names = []
    for estimator_display_name in estimator_display_names:
        if estimator_display_name == 'NKS-MLP':
            new_estimator_display_names.append('NKS-MLP (init: standard)')
        else:
            new_estimator_display_names.append(estimator_display_name)
    return new_estimator_display_names

n_estimators = len(estimator_display_names)
n_datasets = len(datasets)
plt.figure(figsize=figsize)
for dataset_idx, dataset in enumerate(datasets):
    all_cindex_td_scores = []
    for estimator_display_name in estimator_display_names:
        key = (dataset, estimator_display_name)

        cindex_td_scores = results[key][0]

        all_cindex_td_scores.append(cindex_td_scores)
    all_cindex_td_scores = np.array(all_cindex_td_scores)

    plt.subplot(1, n_datasets, dataset_idx + 1)
    plt.boxplot(all_cindex_td_scores[:, [1, 2]][::-1].T,
                usermedians=all_cindex_td_scores[:, 0][::-1],
                whis=(0, 100), showbox=False, vert=False)
    for idx in range(n_estimators):
        plt.hlines(idx + 1, all_cindex_td_scores[:, 1][::-1][idx],
                   all_cindex_td_scores[:, 2][::-1][idx], lw=1)

    if dataset_idx == (n_datasets // 2):
        plt.xlabel('Time-Dependent Concordance Index')

    if dataset_idx == 0:
        plt.yticks(range(1, n_estimators + 1),
                   fix_estimator_names(estimator_display_names[::-1]))
    else:
        plt.yticks(range(1, n_estimators + 1), [''] * n_estimators)

    if dataset == 'metabric':
        plt.title('METABRIC', fontsize='medium')
    elif dataset == 'support2_onehot':
        plt.title('SUPPORT', fontsize='medium')
    elif dataset == 'rotterdam-gbsg2':
        plt.title('Rotterdam/GBSG', fontsize='medium')

plt.tight_layout()
plt.subplots_adjust(wspace=.05)
plt.savefig(os.path.join(output_dir, 'plots', 'cindices.pdf'),
            bbox_inches='tight')
# plt.show()

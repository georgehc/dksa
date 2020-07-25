#!/usr/bin/env python
import ast
import configparser
import os
import pickle
import sys

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

experiment_idx = 0

config = configparser.ConfigParser()
config.read(sys.argv[1])
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
                # if pieces[0] != 'coxph':
                survival_estimator_names.append(pieces[0])
                estimator_display_names.append(pieces[1])

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
    all_times = []
    for survival_estimator_name in survival_estimator_names:
        timing_filename = \
            os.path.join(output_dir, 'timing',
                         '%s_%s_exp%d_cv%d_fitting_times.pkl'
                         % (survival_estimator_name, dataset,
                            experiment_idx, cross_val_n_folds))
        with open(timing_filename, 'rb') as pickle_file:
            times, _ = pickle.load(pickle_file)
            all_times.append(times)

    plt.subplot(1, n_datasets, dataset_idx + 1)
    plt.violinplot(all_times[::-1], vert=False)
    ax = plt.gca()
    ax.set_xscale('log')

    if dataset_idx == (n_datasets // 2):
        plt.xlabel('Training time per cross-validation model fit (seconds)')

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

plt.tight_layout()
plt.subplots_adjust(wspace=.05)
plt.savefig(os.path.join(output_dir, 'plots',
                         'cv_train_times_vs_methods.pdf'),
            bbox_inches='tight')

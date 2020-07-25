#!/usr/bin/env python
import ast
import configparser
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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
                if pieces[0].startswith('nks_'):
                    survival_estimator_names.append(pieces[0])
                    estimator_display_names.append(pieces[1])

figsize = (9, 11)

n_estimators = len(estimator_display_names)
n_datasets = len(datasets)
plt.figure(figsize=figsize)
for dataset_idx, dataset in enumerate(datasets):
    for estimator_idx, (survival_estimator_name, estimator_display_name) \
            in enumerate(zip(survival_estimator_names,
                             estimator_display_names)):
        timing_filename = \
            os.path.join(output_dir, 'timing',
                         '%s_%s_exp%d_cv%d_fitting_times.pkl'
                         % (survival_estimator_name, dataset,
                            experiment_idx, cross_val_n_folds))
        with open(timing_filename, 'rb') as pickle_file:
            _, times = pickle.load(pickle_file)
            num_durations_range = list(sorted(times.keys()))
            if num_durations_range[0] == 0:
                num_durations_range = num_durations_range[1:] + [0]
            all_times = [times[num_durations]
                         for num_durations in num_durations_range]
            labels = []
            for num_durations in num_durations_range:
                if num_durations != 0:
                    labels.append('%d time points' % num_durations)
                else:
                    labels.append('No discretization')

            plt.subplot(n_estimators, n_datasets,
                        estimator_idx * n_datasets + dataset_idx + 1)
            plt.violinplot(all_times[::-1], vert=False)
            ax = plt.gca()

            if dataset_idx == (n_datasets // 2) \
                    and estimator_idx == n_estimators - 1:
                plt.xlabel(
                    'Training time per cross-validation model fit (seconds)')
            if dataset_idx == 0:
                plt.ylabel(estimator_display_name)
                plt.yticks(range(1, len(num_durations_range) + 1),
                           labels[::-1])
            else:
                plt.yticks(range(1, len(num_durations_range) + 1),
                           [''] * len(labels))

            # plt.ylabel('Number of discretized time points')
            if estimator_idx == 0 and dataset == 'metabric':
                plt.title('METABRIC', fontsize='medium')
            elif estimator_idx == 0 and dataset == 'rotterdam-gbsg2':
                plt.title('Rotterdam/GBSG', fontsize='medium')
            elif estimator_idx == 0 and dataset == 'support2_onehot':
                plt.title('SUPPORT', fontsize='medium')

plt.tight_layout()
plt.subplots_adjust(wspace=.05, hspace=.3)
plt.savefig(os.path.join(output_dir, 'plots',
                         'n_durations_cv_train_times.pdf'),
            bbox_inches='tight')
# plt.show()

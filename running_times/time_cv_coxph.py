#!/usr/bin/env python
import ast
import configparser
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pickle
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
__package__ = 'running_times'
import time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold

from survival_datasets import load_dataset


survival_estimator_name = 'coxph'

if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit()

config = configparser.ConfigParser()
config.read(sys.argv[1])
n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
os.makedirs(os.path.join(output_dir, 'timing'), exist_ok=True)

for experiment_idx in range(n_experiment_repeats):
    for dataset in datasets:
        if dataset == 'rotterdam-gbsg2' and experiment_idx > 0:
            # for the Rotterdam/GBSG2 combo, we train on Rotterdam data and
            # test on GBSG2 data, i.e., there's only 1 train/test split
            continue

        print('[Dataset: %s, experiment: %d]' % (dataset, experiment_idx))
        print()

        X_train, y_train, X_test, y_test, feature_names, \
                compute_features_and_transformer, transform_features = \
            load_dataset(dataset, experiment_idx)

        if dataset == 'support2_onehot':
            feature_names = feature_names[:-1]

        cv_fitting_times = []
        kf = KFold(n_splits=cross_val_n_folds, shuffle=False)
        for cross_val_idx, (train_idx, val_idx) \
                in enumerate(kf.split(X_train)):
            fold_X_train = X_train[train_idx]
            fold_y_train = y_train[train_idx]

            fold_X_train_std, transformer = \
                    compute_features_and_transformer(fold_X_train)

            tic = time.time()
            if dataset == 'support2_onehot':
                # drop last column as it's for a one-hot encoding
                fold_X_train_std = fold_X_train_std[:, :-1]

            train_data_df = \
                pd.DataFrame(np.hstack((fold_X_train_std, fold_y_train)),
                             columns=feature_names + ['time', 'status'])
            surv_model = CoxPHFitter()
            surv_model.fit(train_data_df, duration_col='time', event_col='status',
                           show_progress=False, step_size=.1)
            elapsed = time.time() - tic
            print(elapsed)
            cv_fitting_times.append(elapsed)

        print()

        output_timing_filename \
            = os.path.join(output_dir, 'timing',
                           '%s_%s_exp%d_cv%d_fitting_times.pkl'
                           % (survival_estimator_name, dataset, experiment_idx,
                              cross_val_n_folds))

        with open(output_timing_filename, 'wb') as pickle_file:
            pickle.dump((cv_fitting_times,
                         None),
                        pickle_file)

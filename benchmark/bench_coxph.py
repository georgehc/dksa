#!/usr/bin/env python
import ast
import configparser
import csv
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
__package__ = 'benchmark'
import time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler

from survival_datasets import load_dataset


survival_estimator_name = 'coxph'
survival_estimator_display_name = 'Cox'

if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit()

config = configparser.ConfigParser()
config.read(sys.argv[1])
n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

output_table_filename = \
    os.path.join(output_dir,
                 '%s_nexp%d_test_metrics.csv'
                 % (survival_estimator_name,
                    n_experiment_repeats))
output_table_file = open(output_table_filename, 'w')
csv_writer = csv.writer(output_table_file)
csv_writer.writerow(['dataset',
                     'experiment_idx',
                     'method',
                     'cindex_td',
                     'cindex_td_CI_lower',
                     'cindex_td_CI_upper',
                     'cindex_td_CI_mean',
                     'integrated_brier',
                     'integrated_brier_CI_lower',
                     'integrated_brier_CI_upper',
                     'integrated_brier_CI_mean'])


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

        sorted_train_times = np.sort(y_train[:, 0])
        num_train_times = len(sorted_train_times)

        print('Testing...')
        time_elapsed_filename = \
            os.path.join(output_dir, 'models',
                         '%s_%s_%s_test_time.txt'
                         % (survival_estimator_name, dataset, experiment_idx))
        tic = time.time()
        X_train_standardized, transformer = \
                compute_features_and_transformer(X_train)
        X_test_std = transform_features(X_test, transformer)
        if dataset == 'support2_onehot':
            # drop last column as it's for a one-hot encoding
            X_train_standardized = X_train_standardized[:, :-1]
            X_test_std = X_test_std[:, :-1]
            feature_names = feature_names[:-1]

        train_data_df = \
            pd.DataFrame(np.hstack((X_train_standardized, y_train)),
                         columns=feature_names + ['time', 'status'])

        surv_model = CoxPHFitter()
        surv_model.fit(train_data_df, duration_col='time', event_col='status',
                       show_progress=False, step_size=.1)
        elapsed = time.time() - tic
        print('Time elapsed: %f second(s)' % elapsed)
        np.savetxt(time_elapsed_filename, np.array(elapsed).reshape(1, -1))

        # ---------------------------------------------------------------------
        # evaluation
        #

        sorted_y_test = np.unique(y_test[:, 0])
        surv_df = surv_model.predict_survival_function(X_test_std,
                                                       sorted_y_test)
        surv = surv_df.values.T
        ev = EvalSurv(surv_df, y_test[:, 0], y_test[:, 1], censor_surv='km')
        cindex_td = ev.concordance_td('antolini')
        print('c-index (td):', cindex_td)

        linear_predictors = \
            surv_model.predict_log_partial_hazard(X_test_std)
        cindex = concordance_index(y_test[:, 0],
                                   -linear_predictors,
                                   y_test[:, 1])
        print('c-index:', cindex)

        time_grid = np.linspace(sorted_y_test[0], sorted_y_test[-1], 100)
        integrated_brier = ev.integrated_brier_score(time_grid)
        print('Integrated Brier score:', integrated_brier, flush=True)

        test_set_metrics = [cindex_td, integrated_brier]

        rng = np.random.RandomState(bootstrap_random_seed)

        bootstrap_dir = os.path.join(output_dir, 'bootstrap',
                                     '%s_%s_exp%d_test'
                                     % (survival_estimator_name, dataset,
                                        experiment_idx))
        os.makedirs(bootstrap_dir, exist_ok=True)
        cindex_td_filename = os.path.join(bootstrap_dir,
                                          'cindex_td_scores.txt')
        integrated_brier_filename = os.path.join(bootstrap_dir,
                                                 'integrated_brier_scores.txt')
        if not os.path.isfile(cindex_td_filename) \
                or not os.path.isfile(integrated_brier_filename):
            bootstrap_cindex_td_scores = []
            bootstrap_integrated_brier_scores = []
            for bootstrap_idx in range(bootstrap_n_samples):
                bootstrap_sample_indices = \
                    rng.choice(X_test_std.shape[0], size=X_test_std.shape[0],
                               replace=True)

                X_test_std_bootstrap = X_test_std[bootstrap_sample_indices]
                y_test_bootstrap = y_test[bootstrap_sample_indices]
                sorted_y_test_bootstrap = np.unique(y_test_bootstrap[:, 0])

                surv_df = \
                    surv_model.predict_survival_function(X_test_std_bootstrap,
                                                         sorted_y_test_bootstrap)
                ev = EvalSurv(surv_df, y_test_bootstrap[:, 0],
                              y_test_bootstrap[:, 1], censor_surv='km')
                cindex_td = ev.concordance_td('antolini')

                linear_predictors = \
                    surv_model.predict_log_partial_hazard(X_test_std_bootstrap)
                cindex = concordance_index(y_test_bootstrap[:, 0],
                                           -linear_predictors,
                                           y_test_bootstrap[:, 1])

                time_grid = np.linspace(sorted_y_test_bootstrap[0],
                                        sorted_y_test_bootstrap[-1], 100)
                integrated_brier = ev.integrated_brier_score(time_grid)

                bootstrap_cindex_td_scores.append(cindex_td)
                bootstrap_integrated_brier_scores.append(integrated_brier)

            bootstrap_cindex_td_scores = \
                np.array(bootstrap_cindex_td_scores)
            bootstrap_integrated_brier_scores = \
                np.array(bootstrap_integrated_brier_scores)

            np.savetxt(cindex_td_filename, bootstrap_cindex_td_scores)
            np.savetxt(integrated_brier_filename,
                       bootstrap_integrated_brier_scores)
        else:
            bootstrap_cindex_td_scores = \
                np.loadtxt(cindex_td_filename).flatten()
            bootstrap_integrated_brier_scores = \
                np.loadtxt(integrated_brier_filename).flatten()
        print()

        bootstrap_mean_cindex_td = np.mean(bootstrap_cindex_td_scores)
        bootstrap_mean_integrated_brier = \
            np.mean(bootstrap_integrated_brier_scores)

        sorted_bootstrap_cindex_td_scores = \
            np.sort(bootstrap_cindex_td_scores)
        sorted_bootstrap_integrated_brier_scores = \
            np.sort(bootstrap_integrated_brier_scores)

        tail_prob = ((1. - bootstrap_CI_coverage) / 2)
        lower_end = int(np.floor(tail_prob * bootstrap_n_samples))
        upper_end = int(np.ceil((1. - tail_prob) * bootstrap_n_samples))
        print('%0.1f%% bootstrap confidence intervals:'
              % (100 * bootstrap_CI_coverage))
        print('c-index (td): (%0.8f, %0.8f), mean: %0.8f'
              % (sorted_bootstrap_cindex_td_scores[lower_end],
                 sorted_bootstrap_cindex_td_scores[upper_end],
                 bootstrap_mean_cindex_td))
        print('Integrated Brier score: (%0.8f, %0.8f), mean: %0.8f'
              % (sorted_bootstrap_integrated_brier_scores[lower_end],
                 sorted_bootstrap_integrated_brier_scores[upper_end],
                 bootstrap_mean_integrated_brier))
        print()
        test_set_metrics += \
            [sorted_bootstrap_cindex_td_scores[lower_end],
             sorted_bootstrap_cindex_td_scores[upper_end],
             bootstrap_mean_cindex_td,
             sorted_bootstrap_integrated_brier_scores[lower_end],
             sorted_bootstrap_integrated_brier_scores[upper_end],
             bootstrap_mean_integrated_brier]

        np.savetxt(os.path.join(bootstrap_dir, 'final_metrics.txt'),
                   np.array(test_set_metrics))

        csv_writer.writerow([dataset, experiment_idx,
                             survival_estimator_display_name,
                             test_set_metrics[0],
                             test_set_metrics[2],
                             test_set_metrics[3],
                             test_set_metrics[4],
                             test_set_metrics[1],
                             test_set_metrics[5],
                             test_set_metrics[6],
                             test_set_metrics[7]])

        print()
        print()

#!/usr/bin/env python
import ast
import configparser
import csv
import gc
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pickle
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
__package__ = 'benchmark'
import time

import numpy as np
import pandas as pd
from joblib import dump, load
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from sklearn.model_selection import KFold

from npsurvival_models import RandomSurvivalForest
from survival_datasets import load_dataset


survival_estimator_name = 'rsf'
survival_estimator_display_name = 'RSF'

if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit()

config = configparser.ConfigParser()
config.read(sys.argv[1])
n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
method_random_seed = int(config['method: %s'
                                % survival_estimator_name]['random_seed'])
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])
max_n_cores = int(config['DEFAULT']['max_n_cores'])
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

if max_n_cores <= 0:
    n_jobs = os.cpu_count()
else:
    n_jobs = min(max_n_cores, os.cpu_count())

hyperparams = \
    [(max_features, min_samples_leaf, use_km)
     for max_features
     in ast.literal_eval(config['method: %s'
                                % survival_estimator_name]['max_features'])
     for min_samples_leaf
     in ast.literal_eval(config['method: %s'
                                % survival_estimator_name]['min_samples_leaf'])
     for use_km
     in ast.literal_eval(config['method: %s'
                                % survival_estimator_name]['use_km'])]

output_test_table_filename \
    = os.path.join(output_dir,
                   '%s_experiments%d_cv%d_test_metrics_bootstrap.csv'
                   % (survival_estimator_name,
                      n_experiment_repeats,
                      cross_val_n_folds))
output_test_table_file = open(output_test_table_filename, 'w')
test_csv_writer = csv.writer(output_test_table_file)
test_csv_writer.writerow(['dataset',
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

        output_train_metrics_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_cv%d_train_metrics.txt'
                           % (survival_estimator_name, dataset, experiment_idx,
                              cross_val_n_folds))
        output_best_cv_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_cv%d_best_cv_hyperparams.pkl'
                           % (survival_estimator_name, dataset, experiment_idx,
                              cross_val_n_folds))
        if not os.path.isfile(output_train_metrics_filename) or \
                not os.path.isfile(output_best_cv_hyperparam_filename):
            print('Training...', flush=True)
            train_metrics_file = open(output_train_metrics_filename, 'w')
            best_cv_hyperparams = {}

            # load_dataset already shuffles; no need to reshuffle
            kf = KFold(n_splits=cross_val_n_folds, shuffle=False)
            max_cindex = -np.inf
            min_integrated_brier = np.inf
            arg_max_cindex = None
            arg_min_integrated_brier = None

            for hyperparam in hyperparams:
                max_features, min_samples_leaf, use_km = hyperparam
                cindex_scores = []
                integrated_brier_scores = []

                for cross_val_idx, (train_idx, val_idx) \
                        in enumerate(kf.split(X_train)):
                    tic = time.time()
                    fold_X_train = X_train[train_idx]
                    fold_y_train = y_train[train_idx]
                    fold_X_val = X_train[val_idx]
                    fold_y_val = y_train[val_idx]

                    # random survival forests don't need feature standardization

                    model_filename = \
                        os.path.join(output_dir, 'models',
                                     '%s_%s_exp%d_mf%d_msl%d_cv%d.pkl'
                                     % (survival_estimator_name, dataset,
                                        experiment_idx, max_features,
                                        min_samples_leaf, cross_val_idx))
                    time_elapsed_filename = model_filename[:-4] + '_time.txt'
                    if not os.path.isfile(model_filename):
                        surv_model = \
                            RandomSurvivalForest(
                                n_estimators=100,
                                max_features=max_features,
                                max_depth=None,
                                oob_score=False,
                                feature_importance=False,
                                min_samples_leaf=min_samples_leaf,
                                random_state=method_random_seed,
                                n_jobs=n_jobs)
                        surv_model.fit(fold_X_train, fold_y_train)
                        elapsed = time.time() - tic
                        print('Time elapsed: %f second(s)' % elapsed)
                        np.savetxt(time_elapsed_filename,
                                   np.array(elapsed).reshape(1, -1))
                        surv_model.save(model_filename)
                    else:
                        surv_model = RandomSurvivalForest.load(model_filename)
                        elapsed = float(np.loadtxt(time_elapsed_filename))
                        print('Time elapsed (from previous fitting): '
                              + '%f second(s)' % elapsed)

                    sorted_unique_fold_y_train = np.unique(fold_y_train[:, 0])
                    if use_km == 1:
                        use_kaplan_meier = True
                    else:
                        use_kaplan_meier = False
                    surv = \
                        surv_model.predict_surv(fold_X_val,
                                                sorted_unique_fold_y_train,
                                                presorted_times=True,
                                                use_kaplan_meier= \
                                                    use_kaplan_meier)

                    surv_df = pd.DataFrame(surv.T,
                                           columns=range(fold_X_val.shape[0]),
                                           index=sorted_unique_fold_y_train)
                    ev = EvalSurv(surv_df, fold_y_val[:, 0], fold_y_val[:, 1],
                                  censor_surv='km')
                    time_grid = np.linspace(fold_y_val[:, 0].min(),
                                            fold_y_val[:, 0].max(), 100)

                    sorted_fold_y_train = np.sort(fold_y_train[:, 0])

                    cindex_scores.append(ev.concordance_td('antolini'))
                    integrated_brier_scores.append(
                            ev.integrated_brier_score(time_grid))

                    del surv_model
                    gc.collect()

                cross_val_cindex = np.mean(cindex_scores)
                cross_val_integrated_brier = np.mean(integrated_brier_scores)
                print(hyperparam, ':', cross_val_cindex,
                      cross_val_integrated_brier, flush=True)
                print(hyperparam, ':', cross_val_cindex,
                      cross_val_integrated_brier, flush=True,
                      file=train_metrics_file)

                if cross_val_cindex > max_cindex:
                    max_cindex = cross_val_cindex
                    arg_max_cindex = hyperparam

                if cross_val_integrated_brier < min_integrated_brier:
                    min_integrated_brier = cross_val_integrated_brier
                    arg_min_integrated_brier = hyperparam

            train_metrics_file.close()

            best_cv_hyperparams['cindex_td'] = (arg_max_cindex, max_cindex)
            best_cv_hyperparams['integrated_brier'] \
                = (arg_min_integrated_brier, min_integrated_brier)

            with open(output_best_cv_hyperparam_filename, 'wb') as pickle_file:
                pickle.dump(best_cv_hyperparams, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading previous cross-validation results...', flush=True)
            with open(output_best_cv_hyperparam_filename, 'rb') as pickle_file:
                best_cv_hyperparams = pickle.load(pickle_file)
            arg_max_cindex, max_cindex = best_cv_hyperparams['cindex_td']
            arg_min_integrated_brier, min_integrated_brier \
                = best_cv_hyperparams['integrated_brier']

        print('Best hyperparameters for maximizing training c-index (td):',
              arg_max_cindex, '-- achieves score %5.4f' % max_cindex,
              flush=True)
        print('Best hyperparameters for minimizing training integrated ' +
              'Brier score:', arg_min_integrated_brier,
              '-- achieves score %5.4f' % min_integrated_brier, flush=True)

        best_hyperparams = [arg_max_cindex]
        if arg_min_integrated_brier not in best_hyperparams:
            best_hyperparams.append(arg_min_integrated_brier)

        print()
        print('Testing...', flush=True)
        final_test_scores = {}
        for hyperparam in best_hyperparams:
            max_features, min_samples_leaf, use_km = hyperparam
            tic = time.time()
            model_filename = \
                os.path.join(output_dir, 'models',
                             '%s_%s_exp%d_mf%d_msl%d_test.pkl'
                             % (survival_estimator_name, dataset,
                                experiment_idx, max_features,
                                min_samples_leaf))
            time_elapsed_filename = model_filename[:-4] + '_time.txt'
            if not os.path.isfile(model_filename):
                surv_model = \
                    RandomSurvivalForest(n_estimators=100,
                                         max_features=max_features,
                                         max_depth=None,
                                         oob_score=False,
                                         feature_importance=False,
                                         min_samples_leaf=min_samples_leaf,
                                         random_state=method_random_seed,
                                         n_jobs=n_jobs)
                surv_model.fit(X_train, y_train)
                elapsed = time.time() - tic
                print('Time elapsed: %f second(s)' % elapsed)
                np.savetxt(time_elapsed_filename,
                           np.array(elapsed).reshape(1, -1))
                surv_model.save(model_filename)
            else:
                surv_model = RandomSurvivalForest.load(model_filename)
                elapsed = float(np.loadtxt(time_elapsed_filename))
                print('Time elapsed (from previous fitting): %f second(s)'
                      % elapsed)


            sorted_unique_y_train = np.unique(y_train[:, 0])
            if use_km == 1:
                use_kaplan_meier = True
            else:
                use_kaplan_meier = False
            surv = surv_model.predict_surv(X_test, sorted_unique_y_train,
                                           presorted_times=True,
                                           use_kaplan_meier=use_kaplan_meier)

            surv_df = pd.DataFrame(surv.T,
                                   columns=range(X_test.shape[0]),
                                   index=sorted_unique_y_train)
            ev = EvalSurv(surv_df, y_test[:, 0], y_test[:, 1],
                          censor_surv='km')
            time_grid = np.linspace(y_test[:, 0].min(), y_test[:, 0].max(), 100)

            cindex_td = ev.concordance_td('antolini')
            integrated_brier = ev.integrated_brier_score(time_grid)

            sorted_y_train = np.sort(y_train[:, 0])

            print('Hyperparameter', hyperparam,
                  'achieves c-index (td) %5.4f,' % cindex_td,
                  'integrated Brier score %5.4f' % integrated_brier,
                  flush=True)

            test_set_metrics = [cindex_td, integrated_brier]

            rng = np.random.RandomState(bootstrap_random_seed)

            bootstrap_dir = \
                os.path.join(output_dir, 'bootstrap',
                             '%s_%s_exp%d_mf%d_msl%d_km%d_test'
                             % (survival_estimator_name, dataset,
                                experiment_idx, max_features,
                                min_samples_leaf, use_km))
            os.makedirs(bootstrap_dir, exist_ok=True)
            cindex_td_filename = os.path.join(bootstrap_dir,
                                              'cindex_td_scores.txt')
            integrated_brier_filename = \
                os.path.join(bootstrap_dir, 'integrated_brier_scores.txt')
            if not os.path.isfile(cindex_td_filename) \
                    or not os.path.isfile(integrated_brier_filename):
                bootstrap_cindex_td_scores = []
                bootstrap_integrated_brier_scores = []
                for bootstrap_idx in range(bootstrap_n_samples):
                    bootstrap_sample_indices = \
                        rng.choice(X_test.shape[0],
                                   size=X_test.shape[0],
                                   replace=True)

                    y_test_bootstrap = y_test[bootstrap_sample_indices]
                    surv_bootstrap = surv[bootstrap_sample_indices]

                    ev = EvalSurv(pd.DataFrame(surv_bootstrap.T,
                                               columns=range(X_test.shape[0]),
                                               index=sorted_unique_y_train),
                                  y_test_bootstrap[:, 0],
                                  y_test_bootstrap[:, 1], censor_surv='km')
                    time_grid = np.linspace(y_test_bootstrap[:, 0].min(),
                                            y_test_bootstrap[:, 0].max(), 100)

                    cindex_td = ev.concordance_td('antolini')
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

            final_test_scores[hyperparam] = tuple(test_set_metrics)

            del surv_model
            gc.collect()

        test_csv_writer.writerow(
            [dataset, experiment_idx,
             survival_estimator_display_name,
             final_test_scores[arg_max_cindex][0],
             final_test_scores[arg_max_cindex][2],
             final_test_scores[arg_max_cindex][3],
             final_test_scores[arg_max_cindex][4],
             final_test_scores[arg_min_integrated_brier][1],
             final_test_scores[arg_min_integrated_brier][5],
             final_test_scores[arg_min_integrated_brier][6],
             final_test_scores[arg_min_integrated_brier][7]])

        print()
        print()

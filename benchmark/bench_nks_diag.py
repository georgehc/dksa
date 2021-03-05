#!/usr/bin/env python
import ast
import configparser
import csv
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pickle
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
__package__ = 'benchmark'
import time
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split

import torchtuples as tt
from pycox.evaluation import EvalSurv
from neural_kernel_survival import NKS, NKSDiscrete, DiagonalScaler
from survival_datasets import load_dataset


survival_estimator_name = 'nks_diag'
survival_estimator_display_name = 'NKS-Diag'

if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit()

config = configparser.ConfigParser()
config.read(sys.argv[1])
n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
use_cross_val = int(config['DEFAULT']['use_cross_val']) > 0
use_early_stopping = int(config['DEFAULT']['use_early_stopping']) > 0
val_ratio = float(config['DEFAULT']['simple_data_splitting_val_ratio'])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
method_header = 'method: %s' % survival_estimator_name
method_random_seed = int(config[method_header]['random_seed'])
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

n_epochs_list = ast.literal_eval(config[method_header]['n_epochs'])
if use_early_stopping and not use_cross_val:
    n_epochs_list = [np.max(n_epochs_list)]
hyperparams = \
    [(batch_size, n_epochs, lr, num_durations)
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for n_epochs
     in n_epochs_list
     for lr
     in ast.literal_eval(config[method_header]['learning_rate'])
     for num_durations
     in ast.literal_eval(config[method_header]['num_durations'])]

if use_cross_val:
    val_string = 'cv%d' % cross_val_n_folds
else:
    val_string = 'vr%f' % val_ratio
    if use_early_stopping:
        val_string += '_earlystop'

output_test_table_filename \
    = os.path.join(output_dir,
                   '%s_nexp%d_%s_test_metrics.csv'
                   % (survival_estimator_name,
                      n_experiment_repeats,
                      val_string))
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
                           '%s_%s_exp%d_%s_train_metrics.txt'
                           % (survival_estimator_name, dataset, experiment_idx,
                              val_string))
        output_best_val_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_best_val_hyperparams.pkl'
                           % (survival_estimator_name, dataset, experiment_idx,
                              val_string))
        if not os.path.isfile(output_train_metrics_filename) or \
                not os.path.isfile(output_best_val_hyperparam_filename):
            print('Training...', flush=True)
            train_metrics_file = open(output_train_metrics_filename, 'w')
            best_val_hyperparams = {}

            # load_dataset already shuffles; no need to reshuffle
            if use_cross_val:
                kf = KFold(n_splits=cross_val_n_folds, shuffle=False)
                train_data_split = list(kf.split(X_train))
            else:
                train_data_split = [train_test_split(range(len(X_train)),
                                                     test_size=val_ratio,
                                                     shuffle=False)]

            max_cindex = -np.inf
            min_integrated_brier = np.inf
            arg_max_cindex = None
            arg_min_integrated_brier = None

            for hyperparam in hyperparams:
                batch_size, n_epochs, lr, num_durations = hyperparam
                cindex_scores = []
                integrated_brier_scores = []

                for fold_idx, (train_idx, val_idx) \
                        in enumerate(train_data_split):
                    fold_X_train = X_train[train_idx]
                    fold_y_train = y_train[train_idx].astype('float32')
                    fold_X_val = X_train[val_idx]
                    fold_y_val = y_train[val_idx].astype('float32')

                    fold_X_train_std, transformer = \
                            compute_features_and_transformer(fold_X_train)
                    fold_X_val_std = transform_features(fold_X_val, transformer)
                    fold_X_train_std = fold_X_train_std.astype('float32')
                    fold_X_val_std = fold_X_val_std.astype('float32')

                    tic = time.time()
                    torch.manual_seed(method_random_seed)
                    torch.cuda.manual_seed_all(method_random_seed)
                    np.random.seed(method_random_seed)

                    batch_norm = True
                    dropout = 0.
                    output_bias = False

                    optimizer = tt.optim.Adam(lr=lr)
                    net = nn.Sequential(
                        DiagonalScaler(fold_X_train_std.shape[1]))

                    if num_durations > 0:
                        labtrans = NKSDiscrete.label_transform(num_durations)
                        fold_y_train_discrete \
                            = labtrans.fit_transform(*fold_y_train.T)
                        fold_y_val_discrete \
                            = labtrans.transform(*fold_y_val.T)
                        surv_model = NKSDiscrete(net, optimizer,
                                                 duration_index=labtrans.cuts)
                    else:
                        surv_model = NKS(net, optimizer)

                    model_filename = \
                        os.path.join(output_dir, 'models',
                                     '%s_%s_exp%d_%s_bs%d_nep%d_'
                                     % (survival_estimator_name, dataset,
                                        experiment_idx, val_string, batch_size,
                                        n_epochs)
                                     +
                                     'lr%f_nd%d_fold%d.pt'
                                     % (lr, num_durations, fold_idx))
                    time_elapsed_filename = model_filename[:-3] + '_time.txt'
                    if not os.path.isfile(model_filename):
                        if use_cross_val:
                            print('*** Fitting with hyperparam:', hyperparam,
                                  '-- cross val index:', fold_idx, flush=True)
                        else:
                            print('*** Fitting with hyperparam:', hyperparam,
                                  flush=True)
                        if num_durations > 0:
                            if use_early_stopping and not use_cross_val:
                                surv_model.fit(fold_X_train_std,
                                               fold_y_train_discrete,
                                               batch_size, n_epochs,
                                               [tt.callbacks.EarlyStopping()],
                                               val_data=(fold_X_val_std,
                                                         fold_y_val_discrete),
                                               verbose=False)
                            else:
                                surv_model.fit(fold_X_train_std,
                                               fold_y_train_discrete,
                                               batch_size, n_epochs,
                                               verbose=False)
                        else:
                            if use_early_stopping and not use_cross_val:
                                surv_model.fit(fold_X_train_std,
                                               (fold_y_train[:, 0],
                                                fold_y_train[:, 1]), batch_size,
                                               n_epochs,
                                               [tt.callbacks.EarlyStopping()],
                                               val_data=(fold_X_val_std,
                                                         (fold_y_val[:, 0],
                                                          fold_y_val[:, 1])),
                                               verbose=False)
                            else:
                                surv_model.fit(fold_X_train_std,
                                               (fold_y_train[:, 0],
                                                fold_y_train[:, 1]), batch_size,
                                               n_epochs, verbose=False)
                        elapsed = time.time() - tic
                        print('Time elapsed: %f second(s)' % elapsed)
                        np.savetxt(time_elapsed_filename,
                                   np.array(elapsed).reshape(1, -1))
                        surv_model.save_net(model_filename)
                    else:
                        print('*** Loading ***', flush=True)
                        surv_model.load_net(model_filename)
                        elapsed = float(np.loadtxt(time_elapsed_filename))
                        print('Time elapsed (from previous fitting): '
                              + '%f second(s)' % elapsed)

                    if num_durations > 0:
                        surv_df = \
                            surv_model.interpolate(10).predict_surv_df(
                                fold_X_val_std)
                    else:
                        surv_df = surv_model.predict_surv_df(fold_X_val_std)
                    ev = EvalSurv(surv_df, fold_y_val[:, 0], fold_y_val[:, 1],
                                  censor_surv='km')

                    sorted_fold_y_val = np.sort(np.unique(fold_y_val[:, 0]))
                    time_grid = np.linspace(sorted_fold_y_val[0],
                                            sorted_fold_y_val[-1], 100)

                    surv = surv_df.to_numpy().T

                    cindex_scores.append(ev.concordance_td('antolini'))
                    integrated_brier_scores.append(
                            ev.integrated_brier_score(time_grid))
                    print('  c-index (td):', cindex_scores[-1])
                    print('  Integrated Brier score:',
                          integrated_brier_scores[-1])

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

            best_val_hyperparams['cindex_td'] = \
                (arg_max_cindex, max_cindex)
            best_val_hyperparams['integrated_brier'] = \
                (arg_min_integrated_brier, min_integrated_brier)

            with open(output_best_val_hyperparam_filename, 'wb') as pickle_file:
                pickle.dump(best_val_hyperparams, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading previous cross-validation results...', flush=True)
            with open(output_best_val_hyperparam_filename, 'rb') as pickle_file:
                best_val_hyperparams = pickle.load(pickle_file)
            arg_max_cindex, max_cindex = best_val_hyperparams['cindex_td']
            arg_min_integrated_brier, min_integrated_brier \
                = best_val_hyperparams['integrated_brier']

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
        X_train_std, transformer = \
                compute_features_and_transformer(X_train)
        X_test_std = transform_features(X_test, transformer)
        X_train_std = X_train_std.astype('float32')
        X_test_std = X_test_std.astype('float32')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
        final_test_scores = {}
        for hyperparam in best_hyperparams:
            batch_size, n_epochs, lr, num_durations = hyperparam
            tic = time.time()
            torch.manual_seed(method_random_seed)
            torch.cuda.manual_seed_all(method_random_seed)
            np.random.seed(method_random_seed)

            batch_norm = True
            dropout = 0.0
            output_bias = False

            optimizer = tt.optim.Adam(lr=lr)
            net = nn.Sequential(DiagonalScaler(X_train_std.shape[1]))

            if num_durations > 0:
                labtrans = NKSDiscrete.label_transform(num_durations)
                y_train_discrete = labtrans.fit_transform(*y_train.T)
                surv_model = NKSDiscrete(net, optimizer,
                                         duration_index=labtrans.cuts)
            else:
                surv_model = NKS(net, optimizer)

            model_filename = \
                os.path.join(output_dir, 'models',
                             '%s_%s_exp%d_%s_bs%d_nep%d_lr%f_nd%d_test.pt'
                             % (survival_estimator_name, dataset,
                                experiment_idx, val_string, batch_size,
                                n_epochs, lr, num_durations))
            time_elapsed_filename = model_filename[:-3] + '_time.txt'
            if not use_cross_val:
                val_model_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_%s_bs%d_nep%d_'
                                 % (survival_estimator_name, dataset,
                                    experiment_idx, val_string, batch_size,
                                    n_epochs)
                                 +
                                 'lr%f_nd%d_fold%d.pt'
                                 % (lr, num_durations, 0))
                val_time_elapsed_filename = \
                    val_model_filename[:-3] + '_time.txt'
                copyfile(val_model_filename, model_filename)
                copyfile(val_model_filename[:-3] + '_train_features.txt',
                         model_filename[:-3] + '_train_features.txt')
                copyfile(val_model_filename[:-3] + '_train_observed_times.txt',
                         model_filename[:-3] + '_train_observed_times.txt')
                copyfile(val_model_filename[:-3] + '_train_events.txt',
                         model_filename[:-3] + '_train_events.txt')
                copyfile(val_model_filename[:-3] + '_train_embeddings.txt',
                         model_filename[:-3] + '_train_embeddings.txt')
                copyfile(val_time_elapsed_filename,
                         time_elapsed_filename)
            if not os.path.isfile(model_filename):
                print('*** Fitting with hyperparam:', hyperparam, flush=True)
                if num_durations > 0:
                    surv_model.fit(X_train_std, y_train_discrete,
                                   batch_size, n_epochs, verbose=False)
                else:
                    surv_model.fit(X_train_std, (y_train[:, 0], y_train[:, 1]),
                                   batch_size, n_epochs, verbose=False)
                elapsed = time.time() - tic
                print('Time elapsed: %f second(s)' % elapsed)
                np.savetxt(time_elapsed_filename,
                           np.array(elapsed).reshape(1, -1))
                surv_model.save_net(model_filename)
            else:
                print('*** Loading ***', flush=True)
                surv_model.load_net(model_filename)
                elapsed = float(np.loadtxt(time_elapsed_filename))
                print('Time elapsed (from previous fitting): %f second(s)'
                      % elapsed)

            if num_durations > 0:
                surv_model_interp = surv_model.interpolate(10)
                surv_df = surv_model_interp.predict_surv_df(X_test_std)
            else:
                surv_df = surv_model.predict_surv_df(X_test_std)
            surv_index = surv_df.index
            ev = EvalSurv(surv_df, y_test[:, 0], y_test[:, 1], censor_surv='km')

            sorted_y_test = np.unique(y_test[:, 0])
            time_grid = np.linspace(sorted_y_test[0], sorted_y_test[-1], 100)

            cindex_td = ev.concordance_td('antolini')
            integrated_brier = ev.integrated_brier_score(time_grid)

            surv = surv_df.to_numpy().T

            print('Hyperparameter', hyperparam,
                  'achieves c-index (td) %5.4f,' % cindex_td,
                  'integrated Brier score %5.4f' % integrated_brier,
                  flush=True)

            test_set_metrics = [cindex_td, integrated_brier]

            rng = np.random.RandomState(bootstrap_random_seed)

            bootstrap_dir = \
                os.path.join(output_dir, 'bootstrap',
                             '%s_%s_exp%d_%s_bs%d_nep%d_lr%f_nd%d_test'
                             % (survival_estimator_name, dataset,
                                experiment_idx, val_string, batch_size,
                                n_epochs, lr, num_durations))
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
                        rng.choice(X_test_std.shape[0],
                                   size=X_test_std.shape[0],
                                   replace=True)

                    y_test_bootstrap = y_test[bootstrap_sample_indices]
                    surv_bootstrap = surv[bootstrap_sample_indices]

                    ev = EvalSurv(pd.DataFrame(surv_bootstrap.T,
                                               columns= \
                                                   range(X_test_std.shape[0]),
                                               index=surv_index),
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

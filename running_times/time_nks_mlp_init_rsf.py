#!/usr/bin/env python
import ast
import configparser
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pickle
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
__package__ = 'running_times'

import numpy as np

from survival_datasets import load_dataset


init_survival_estimator_name = 'rsf'
survival_estimator_name = 'nks_mlp_init_rsf_full_train'

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
os.makedirs(os.path.join(output_dir, 'timing'), exist_ok=True)

n_epochs_list = ast.literal_eval(config[method_header]['n_epochs'])
if use_early_stopping and not use_cross_val:
    n_epochs_list = [np.max(n_epochs_list)]
hyperparams = \
    [(batch_size, n_epochs, n_layers, n_nodes, lr, num_durations)
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for n_epochs
     in n_epochs_list
     for n_layers
     in ast.literal_eval(config[method_header]['n_layers'])
     for n_nodes
     in ast.literal_eval(config[method_header]['n_nodes'])
     for lr
     in ast.literal_eval(config[method_header]['learning_rate'])
     for num_durations
     in ast.literal_eval(config[method_header]['num_durations'])]

if use_cross_val:
    val_string = 'cv%d' % cross_val_n_folds
    init_val_string = val_string
    n_folds = cross_val_n_folds
else:
    val_string = 'vr%f' % val_ratio
    init_val_string = val_string
    if use_early_stopping:
        val_string += '_earlystop'
    n_folds = 1

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

        init_best_val_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_best_val_hyperparams.pkl'
                           % (init_survival_estimator_name, dataset,
                              experiment_idx,
                              init_val_string))
        assert os.path.isfile(init_best_val_hyperparam_filename)
        with open(init_best_val_hyperparam_filename, 'rb') as pickle_file:
            best_val_hyperparams = pickle.load(pickle_file)
        arg_max_cindex, max_cindex = best_val_hyperparams['cindex_td']
        max_features, min_samples_leaf = arg_max_cindex

        model_filename = \
            os.path.join(output_dir, 'models',
                         '%s_%s_exp%d_%s_mf%d_msl%d_test.pkl'
                         % (init_survival_estimator_name, dataset,
                            experiment_idx, init_val_string, max_features,
                            min_samples_leaf))
        time_elapsed_filename = model_filename[:-4] + '_time.txt'
        rsf_time = float(np.loadtxt(time_elapsed_filename))

        prox_filename = model_filename[:-4] + '_prox_matrix.txt'
        time_elapsed_filename = prox_filename[:-4] + '_time.txt'
        prox_matrix_time = float(np.loadtxt(time_elapsed_filename))

        mds_size = min(len(X_train), X_train.shape[1])
        mds_filename = model_filename[:-4] + '_mds%d.txt' % mds_size
        time_elapsed_filename = mds_filename[:-4] + '_time.txt'
        mds_time = float(np.loadtxt(time_elapsed_filename))

        print('RSF training time: %f' % rsf_time)
        print('Prox matrix computation time: %f' % prox_matrix_time)
        print('MDS computation time: %f' % mds_time)
        print('Total pre-processing excluding RSF hyperparameter sweep: %f'
              % (rsf_time + prox_matrix_time + mds_time))
        print()

        fitting_times_by_num_durations = {}
        fitting_times = []
        fine_tuning_times_by_num_durations = {}
        fine_tuning_times = []
        rsf_neural_approx_times = []
        for hyperparam in hyperparams:
            batch_size, n_epochs, n_layers, n_nodes, lr, num_durations \
                = hyperparam

            for fold_idx in range(n_folds):
                emb_model_filename = \
                    os.path.join(output_dir, 'models',
                                 'rsf_full_train_neural_approx_'
                                 +
                                 '%s_exp%d_%s_mf%d_msl%d_'
                                 % (dataset, experiment_idx,
                                    init_val_string, max_features,
                                    min_samples_leaf)
                                 +
                                 'bs%d_nep%d_nla%d_nno%d_'
                                 % (batch_size, 100, n_layers, n_nodes)
                                 +
                                 'lr%f_fold%d.pt'
                                 % (lr, fold_idx))
                time_elapsed_filename = emb_model_filename[:-3] + '_time.txt'
                rsf_neural_approx_time = float(np.loadtxt(time_elapsed_filename))

                rsf_neural_approx_times.append(rsf_neural_approx_time)

                model_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_%s_mf%d_msl%d_'
                                 % (survival_estimator_name, dataset,
                                    experiment_idx, val_string, max_features,
                                    min_samples_leaf)
                                 +
                                 'bs%d_nep%d_nla%d_nno%d_'
                                 % (batch_size, n_epochs, n_layers, n_nodes)
                                 +
                                 'lr%f_nd%d_fold%d.pt'
                                 % (lr, num_durations, fold_idx))
                time_elapsed_filename = model_filename[:-3] + '_time.txt'
                fine_tune_time = float(np.loadtxt(time_elapsed_filename))
                elapsed = rsf_neural_approx_time + fine_tune_time

                fitting_times.append(elapsed)
                if num_durations not in fitting_times_by_num_durations:
                    fitting_times_by_num_durations[num_durations] \
                        = [elapsed]
                else:
                    fitting_times_by_num_durations[num_durations].append(
                        elapsed)

                fine_tuning_times.append(fine_tune_time)
                if num_durations not in fine_tuning_times_by_num_durations:
                    fine_tuning_times_by_num_durations[num_durations] \
                        = [fine_tune_time]
                else:
                    fine_tuning_times_by_num_durations[num_durations].append(
                        fine_tune_time)

        num_durations_range \
            = list(sorted(fitting_times_by_num_durations.keys()))
        if num_durations_range[0] == 0 and len(num_durations_range) > 2:
            num_durations_range = num_durations_range[1:] + [0]

        # note: the RSF neural approximation does *not* need to be re-fit for
        # every hyperparameter choice, so there will be duplicate values
        # (from just loading a previous fitting time); here we just get the
        # unique times' mean/std
        unique_rsf_neural_approx_times = np.unique(rsf_neural_approx_times)
        print('RSF neural approx times: %f +/- %f (std dev)'
              % (np.mean(unique_rsf_neural_approx_times),
                 np.std(unique_rsf_neural_approx_times)))
        print()
        print('Fine tuning times during hyperparameter sweep: '
              + '%f +/- %f (std dev)'
              % (np.mean(fine_tuning_times),
                 np.std(fine_tuning_times)))
        for num_durations in num_durations_range:
            print('Fine tuning (num durations %d): %f +/- %f (std dev)'
                  % (num_durations,
                     np.mean(
                         fine_tuning_times_by_num_durations[num_durations]),
                     np.std(
                         fine_tuning_times_by_num_durations[num_durations])))
        print()
        print('Fitting times during hyperparameter sweep: %f +/- %f (std dev)'
              % (np.mean(fitting_times),
                 np.std(fitting_times)))
        for num_durations in num_durations_range:
            print('Fitting times (num durations %d): %f +/- %f (std dev)'
                  % (num_durations,
                     np.mean(fitting_times_by_num_durations[num_durations]),
                     np.std(fitting_times_by_num_durations[num_durations])))
        print()

        output_timing_filename \
            = os.path.join(output_dir, 'timing',
                           '%s_%s_exp%d_%s_fitting_times.pkl'
                           % (survival_estimator_name, dataset, experiment_idx,
                              val_string))

        with open(output_timing_filename, 'wb') as pickle_file:
            pickle.dump((fitting_times,
                         fitting_times_by_num_durations),
                        pickle_file)


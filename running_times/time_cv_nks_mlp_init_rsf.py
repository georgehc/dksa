#!/usr/bin/env python
import ast
import configparser
import gc
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
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
method_header = 'method: %s' % survival_estimator_name
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'timing'), exist_ok=True)


hyperparams = \
    [(batch_size, n_epochs, n_layers, n_nodes, lr, num_durations)
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for n_epochs
     in ast.literal_eval(config[method_header]['n_epochs'])
     for n_layers
     in ast.literal_eval(config[method_header]['n_layers'])
     for n_nodes
     in ast.literal_eval(config[method_header]['n_nodes'])
     for lr
     in ast.literal_eval(config[method_header]['learning_rate'])
     for num_durations
     in ast.literal_eval(config[method_header]['num_durations'])]


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

        init_best_cv_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_cv%d_best_cv_hyperparams.pkl'
                           % (init_survival_estimator_name, dataset,
                              experiment_idx,
                              cross_val_n_folds))
        assert os.path.isfile(init_best_cv_hyperparam_filename)
        with open(init_best_cv_hyperparam_filename, 'rb') as pickle_file:
            best_cv_hyperparams = pickle.load(pickle_file)
        arg_max_cindex, max_cindex = best_cv_hyperparams['cindex_td']
        max_features, min_samples_leaf, use_km = arg_max_cindex

        model_filename = \
            os.path.join(output_dir, 'models',
                         '%s_%s_exp%d_mf%d_msl%d_km%d_test.pkl'
                         % (init_survival_estimator_name, dataset,
                            experiment_idx, max_features,
                            min_samples_leaf, use_km))
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
        print('Total pre-processing excluding RSF cross-val: %f'
              % (rsf_time + prox_matrix_time + mds_time))
        print()

        cv_fitting_times_by_num_durations = {}
        cv_fitting_times = []
        cv_fine_tuning_times_by_num_durations = {}
        cv_fine_tuning_times = []
        rsf_neural_approx_times = []
        for hyperparam in hyperparams:
            batch_size, n_epochs, n_layers, n_nodes, lr, num_durations \
                = hyperparam

            for cross_val_idx in range(cross_val_n_folds):
                emb_model_filename = \
                    os.path.join(output_dir, 'models',
                                 'rsf_full_train_neural_approx_'
                                 +
                                 '%s_exp%d_mf%d_msl%d_km%d_'
                                 % (dataset, experiment_idx,
                                    max_features, min_samples_leaf,
                                    use_km)
                                 +
                                 'bs%d_nep%d_nla%d_nno%d_'
                                 % (batch_size, 100, n_layers, n_nodes)
                                 +
                                 'lr%f_cv%d.pt'
                                 % (lr, cross_val_idx))
                time_elapsed_filename = emb_model_filename[:-3] + '_time.txt'
                rsf_neural_approx_time = float(np.loadtxt(time_elapsed_filename))

                rsf_neural_approx_times.append(rsf_neural_approx_time)

                model_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_mf%d_msl%d_km%d_'
                                 % (survival_estimator_name, dataset,
                                    experiment_idx, max_features,
                                    min_samples_leaf, use_km)
                                 +
                                 'bs%d_nep%d_nla%d_nno%d_'
                                 % (batch_size, n_epochs, n_layers, n_nodes)
                                 +
                                 'lr%f_nd%d_cv%d.pt'
                                 % (lr, num_durations, cross_val_idx))
                time_elapsed_filename = model_filename[:-3] + '_time.txt'
                fine_tune_time = float(np.loadtxt(time_elapsed_filename))
                elapsed = rsf_neural_approx_time + fine_tune_time

                cv_fitting_times.append(elapsed)
                if num_durations not in cv_fitting_times_by_num_durations:
                    cv_fitting_times_by_num_durations[num_durations] \
                        = [elapsed]
                else:
                    cv_fitting_times_by_num_durations[num_durations].append(
                        elapsed)

                cv_fine_tuning_times.append(fine_tune_time)
                if num_durations not in cv_fine_tuning_times_by_num_durations:
                    cv_fine_tuning_times_by_num_durations[num_durations] \
                        = [fine_tune_time]
                else:
                    cv_fine_tuning_times_by_num_durations[num_durations].append(
                        fine_tune_time)

        num_durations_range \
            = list(sorted(cv_fitting_times_by_num_durations.keys()))
        if num_durations_range[0] == 0 and len(num_durations_range) > 2:
            num_durations_range = num_durations_range[1:] + [0]
        print('RSF neural approx times: %f +/- %f (std dev)'
              % (np.mean(rsf_neural_approx_times),
                 np.std(rsf_neural_approx_times) / len(num_durations_range)))
        print()
        print('CV fine tuning times: %f +/- %f (std dev)'
              % (np.mean(cv_fine_tuning_times),
                 np.std(cv_fine_tuning_times)))
        for num_durations in num_durations_range:
            print('CV fine tuning (num durations %d): %f +/- %f (std dev)'
                  % (num_durations,
                     np.mean(
                         cv_fine_tuning_times_by_num_durations[num_durations]),
                     np.std(
                         cv_fine_tuning_times_by_num_durations[num_durations])))
        print()
        print('CV fitting times: %f +/- %f (std dev)'
              % (np.mean(cv_fitting_times),
                 np.std(cv_fitting_times)))
        for num_durations in num_durations_range:
            print('CV fitting times (num durations %d): %f +/- %f (std dev)'
                  % (num_durations,
                     np.mean(cv_fitting_times_by_num_durations[num_durations]),
                     np.std(cv_fitting_times_by_num_durations[num_durations])))
        print()

        output_timing_filename \
            = os.path.join(output_dir, 'timing',
                           '%s_%s_exp%d_cv%d_fitting_times.pkl'
                           % (survival_estimator_name, dataset, experiment_idx,
                              cross_val_n_folds))

        with open(output_timing_filename, 'wb') as pickle_file:
            pickle.dump((cv_fitting_times,
                         cv_fitting_times_by_num_durations),
                        pickle_file)


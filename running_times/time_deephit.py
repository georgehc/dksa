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


survival_estimator_name = 'deephit'

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
    [(batch_size, n_epochs, n_layers, n_nodes, lr, alpha, sigma, num_durations)
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
     for alpha
     in ast.literal_eval(config[method_header]['alpha'])
     for sigma
     in ast.literal_eval(config[method_header]['sigma'])
     for num_durations
     in ast.literal_eval(config[method_header]['num_durations'])]

if use_cross_val:
    val_string = 'cv%d' % cross_val_n_folds
    n_folds = cross_val_n_folds
else:
    val_string = 'vr%f' % val_ratio
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

        fitting_times_by_num_durations = {}
        fitting_times = []
        for hyperparam in hyperparams:
            batch_size, n_epochs, n_layers, n_nodes, lr, alpha, sigma, \
                num_durations = hyperparam

            for fold_idx in range(n_folds):
                model_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_%s_bs%d_nep%d_nla%d_nno%d_'
                                 % (survival_estimator_name, dataset,
                                    experiment_idx, val_string, batch_size,
                                    n_epochs, n_layers, n_nodes)
                                 +
                                 'lr%f_a%f_s%f_nd%d_fold%d.pt'
                                 % (lr, alpha, sigma, num_durations,
                                    fold_idx))
                time_elapsed_filename = model_filename[:-3] + '_time.txt'
                elapsed = float(np.loadtxt(time_elapsed_filename))

                fitting_times.append(elapsed)
                if num_durations not in fitting_times_by_num_durations:
                    fitting_times_by_num_durations[num_durations] = [elapsed]
                else:
                    fitting_times_by_num_durations[num_durations].append(
                        elapsed)

        print('Fitting times during hyperparameter sweep: %f +/- %f (std dev)'
              % (np.mean(fitting_times),
                 np.std(fitting_times)))
        num_durations_range \
            = list(sorted(fitting_times_by_num_durations.keys()))
        if num_durations_range[0] == 0 and len(num_durations_range) > 2:
            num_durations_range = num_durations_range[1:] + [0]
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


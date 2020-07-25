#!/usr/bin/env python
import ast
import configparser
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pickle
import sys

import numpy as np


survival_estimator_name = 'nks_diag'

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
os.makedirs(os.path.join(output_dir, 'timing'), exist_ok=True)

hyperparams = \
    [(batch_size, n_epochs, lr, num_durations)
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for n_epochs
     in ast.literal_eval(config[method_header]['n_epochs'])
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

        cv_fitting_times_by_num_durations = {}
        cv_fitting_times = []
        for hyperparam in hyperparams:
            batch_size, n_epochs, lr, num_durations = hyperparam

            for cross_val_idx in range(cross_val_n_folds):
                model_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_bs%d_nep%d_'
                                 % (survival_estimator_name, dataset,
                                    experiment_idx, batch_size, n_epochs)
                                 +
                                 'lr%f_nd%d_cv%d.pt'
                                 % (lr, num_durations, cross_val_idx))
                time_elapsed_filename = model_filename[:-3] + '_time.txt'
                elapsed = float(np.loadtxt(time_elapsed_filename))

                cv_fitting_times.append(elapsed)
                if num_durations not in cv_fitting_times_by_num_durations:
                    cv_fitting_times_by_num_durations[num_durations] = [elapsed]
                else:
                    cv_fitting_times_by_num_durations[num_durations].append(
                        elapsed)

        print('CV fitting times: %f +/- %f (std dev)'
              % (np.mean(cv_fitting_times),
                 np.std(cv_fitting_times)))
        num_durations_range \
            = list(sorted(cv_fitting_times_by_num_durations.keys()))
        if num_durations_range[0] == 0 and len(num_durations_range) > 2:
            num_durations_range = num_durations_range[1:] + [0]
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


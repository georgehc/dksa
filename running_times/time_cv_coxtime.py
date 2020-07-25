#!/usr/bin/env python
import ast
import configparser
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pickle
import sys

import numpy as np


survival_estimator_name = 'coxtime'

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
    [(batch_size, n_epochs, n_layers, n_nodes, lr)
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for n_epochs
     in ast.literal_eval(config[method_header]['n_epochs'])
     for n_layers
     in ast.literal_eval(config[method_header]['n_layers'])
     for n_nodes
     in ast.literal_eval(config[method_header]['n_nodes'])
     for lr
     in ast.literal_eval(config[method_header]['learning_rate'])]


for experiment_idx in range(n_experiment_repeats):
    for dataset in datasets:
        if dataset == 'rotterdam-gbsg2' and experiment_idx > 0:
            # for the Rotterdam/GBSG2 combo, we train on Rotterdam data and
            # test on GBSG2 data, i.e., there's only 1 train/test split
            continue

        print('[Dataset: %s, experiment: %d]' % (dataset, experiment_idx))
        print()

        cv_fitting_times = []
        for hyperparam in hyperparams:
            batch_size, n_epochs, n_layers, n_nodes, lr = hyperparam

            for cross_val_idx in range(cross_val_n_folds):
                model_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_bs%d_nep%d_nla%d_nno%d_'
                                 % (survival_estimator_name, dataset,
                                    experiment_idx, batch_size, n_epochs,
                                    n_layers, n_nodes)
                                 +
                                 'lr%f_cv%d.pt'
                                 % (lr, cross_val_idx))
                time_elapsed_filename = model_filename[:-3] + '_time.txt'
                elapsed = float(np.loadtxt(time_elapsed_filename))

                cv_fitting_times.append(elapsed)

        print('CV fitting times: %f +/- %f (std dev)'
              % (np.mean(cv_fitting_times),
                 np.std(cv_fitting_times)))
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


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
__package__ = 'prediction_intervals'

import numpy as np
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn

import torchtuples as tt
from pycox.models import LogisticHazard
from survival_datasets import load_dataset


survival_estimator_name = 'nnet_survival'
survival_estimator_display_name = 'Nnet-survival'

if not (len(sys.argv) == 4 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" ' % sys.argv[0]
          + '[config file] [experiment index] [dataset]')
    sys.exit()

experiment_idx = int(sys.argv[2])
dataset = sys.argv[3]

config = configparser.ConfigParser()
config.read(sys.argv[1])
use_cross_val = int(config['DEFAULT']['use_cross_val']) > 0
use_early_stopping = int(config['DEFAULT']['use_early_stopping']) > 0
val_ratio = float(config['DEFAULT']['simple_data_splitting_val_ratio'])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
output_dir = config['DEFAULT']['output_dir']
method_header = 'method: %s' % survival_estimator_name
method_random_seed = int(config[method_header]['random_seed'])
conformal_prediction_CI_coverage_range = \
    ast.literal_eval(config['DEFAULT']['conformal_prediction_CI_coverage'])
conformal_prediction_random_seed = \
    int(config['DEFAULT']['conformal_prediction_random_seed'])
conformal_prediction_n_samples = \
    int(config['DEFAULT']['conformal_prediction_n_samples'])
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'split_conformal_prediction'),
            exist_ok=True)
os.makedirs(os.path.join(output_dir, 'weighted_split_conformal_prediction'),
            exist_ok=True)

if use_cross_val:
    val_string = 'cv%d' % cross_val_n_folds
else:
    val_string = 'vr%f' % val_ratio
    if use_early_stopping:
        val_string += '_earlystop'

output_best_val_hyperparam_filename \
    = os.path.join(output_dir, 'train',
                   '%s_%s_exp%d_%s_best_val_hyperparams.pkl'
                   % (survival_estimator_name, dataset, experiment_idx,
                      val_string))
if not os.path.isfile(output_best_val_hyperparam_filename):
    raise Exception("File does not exist: %s\n\n"
                    % output_best_val_hyperparam_filename
                    + 'Running the benchmark demo first should resolve this.')

with open(output_best_val_hyperparam_filename, 'rb') as pickle_file:
    best_val_hyperparams = pickle.load(pickle_file)
arg_max_cindex, max_cindex = best_val_hyperparams['cindex_td']
batch_size, n_epochs, n_layers, n_nodes, lr, num_durations = arg_max_cindex
hyperparam = arg_max_cindex

if dataset == 'rotterdam-gbsg2' and experiment_idx > 0:
    # for the Rotterdam/GBSG2 combo, we train on Rotterdam data and
    # test on GBSG2 data, i.e., there's only 1 train/test split
    print('*** WARNING: Rotterdam/GSBG2 only has 1 experiment. Exiting. ***')
    sys.exit()

print('[Dataset: %s, experiment: %d]' % (dataset, experiment_idx))
print()

X_train, y_train, X_test, y_test, feature_names, \
        compute_features_and_transformer, transform_features = \
    load_dataset(dataset, experiment_idx)

print('Testing...', flush=True)
X_train_std, transformer = \
        compute_features_and_transformer(X_train)
X_test_std = transform_features(X_test, transformer)
X_train_std = X_train_std.astype('float32')
X_test_std = X_test_std.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
labtrans = LogisticHazard.label_transform(num_durations)
y_train_discrete = labtrans.fit_transform(*y_train.T)

torch.manual_seed(method_random_seed)
torch.cuda.manual_seed_all(method_random_seed)
np.random.seed(method_random_seed)

batch_norm = True
dropout = 0.0
output_bias = True

net = tt.practical.MLPVanilla(X_train_std.shape[1],
                              [n_nodes for layer_idx
                               in range(n_layers)],
                              labtrans.out_features,
                              batch_norm,
                              dropout,
                              output_bias=output_bias)

optimizer = tt.optim.Adam(lr=lr)

surv_model = LogisticHazard(net, optimizer,
                            duration_index=labtrans.cuts)

model_filename = \
    os.path.join(output_dir, 'models',
                 '%s_%s_exp%d_%s_bs%d_nep%d_nla%d_nno%d_lr%f_nd%d_test.pt'
                 % (survival_estimator_name, dataset, experiment_idx,
                    val_string, batch_size, n_epochs, n_layers, n_nodes, lr,
                    num_durations))
assert os.path.isfile(model_filename)
print('*** Loading ***', flush=True)
surv_model.load_net(model_filename)

surv_df = surv_model.interpolate(10).predict_surv_df(X_test_std)
surv = surv_df.to_numpy().T

print()
print('[Test data statistics]')
sorted_y_test_times = np.sort(y_test[:, 0])
print('Quartiles:')
print('- Min observed time:', np.min(y_test[:, 0]))
print('- Q1 observed time:',
      sorted_y_test_times[int(0.25 * len(sorted_y_test_times))])
print('- Median observed time:', np.median(y_test[:, 0]))
print('- Q3 observed time:',
      sorted_y_test_times[int(0.75 * len(sorted_y_test_times))])
print('- Max observed time:', np.max(y_test[:, 0]))
print('Mean observed time:', np.mean(y_test[:, 0]))
print('Fraction censored:', 1. - np.mean(y_test[:, 1]))
print()


time_points = surv_df.index.to_numpy()
from util import compute_median_survival_time, compute_mean_survival_time


def survival_time_nonconformity(predicted_survival_time,
        observed_time, event_ind):
    if event_ind == 1:
        return np.abs(observed_time - predicted_survival_time)
    else:
        return np.maximum(observed_time - predicted_survival_time, 0)


mean_predicted_survival_times = \
    np.array([compute_mean_survival_time(time_points, row) for row in surv])

median_predicted_survival_times = \
    np.array([compute_median_survival_time(time_points, row) for row in surv])


for survival_time_estimator in ['mean', 'median']:
    if survival_time_estimator == 'mean':
        predicted_survival_times = mean_predicted_survival_times.copy()
    else:
        predicted_survival_times = median_predicted_survival_times.copy()

    for calib_frac in np.linspace(0.1, 1, 10):
        print('[Split conformal prediction - %s, calib frac %f]'
              % (survival_time_estimator, calib_frac))

        rng = np.random.RandomState(conformal_prediction_random_seed)

        coverages = {conformal_prediction_CI_coverage: []
                     for conformal_prediction_CI_coverage
                     in conformal_prediction_CI_coverage_range}
        qhats = {conformal_prediction_CI_coverage: []
                 for conformal_prediction_CI_coverage
                 in conformal_prediction_CI_coverage_range}
        for repeat_idx in range(conformal_prediction_n_samples):
            n_calib = X_test.shape[0] // 2
            n_test_start_idx = n_calib
            n_calib = int(calib_frac * n_calib)

            random_shuffle = rng.permutation(X_test.shape[0])
            calibration_indices = random_shuffle[:n_calib]
            test_indices = random_shuffle[n_test_start_idx:]

            y_calib = y_test[calibration_indices]
            estimated_T_calib = predicted_survival_times[calibration_indices]
            nonconformity_scores = \
                (y_calib[:, 1] * np.abs(y_calib[:, 0] - estimated_T_calib)) \
                + ((1 - y_calib[:, 1])
                   * np.maximum(y_calib[:, 0] - estimated_T_calib, 0))
            nonconformity_scores = np.append(nonconformity_scores, np.inf)
            sorted_nonconformity_scores = np.sort(nonconformity_scores)
            qhat = {conformal_prediction_CI_coverage:
                        sorted_nonconformity_scores[
                            int(np.ceil(conformal_prediction_CI_coverage
                                        * (n_calib + 1)))]
                    for conformal_prediction_CI_coverage
                    in conformal_prediction_CI_coverage_range}

            included = {conformal_prediction_CI_coverage: 0
                        for conformal_prediction_CI_coverage
                        in conformal_prediction_CI_coverage_range}
            for test_point_idx in test_indices:
                for conformal_prediction_CI_coverage \
                        in conformal_prediction_CI_coverage_range:
                    CI_lower = predicted_survival_times[test_point_idx] \
                        - qhat[conformal_prediction_CI_coverage]
                    CI_upper = predicted_survival_times[test_point_idx] \
                        + qhat[conformal_prediction_CI_coverage]

                    if y_test[test_point_idx, 1] > 0:
                        if y_test[test_point_idx, 0] >= CI_lower \
                                and y_test[test_point_idx, 0] <= CI_upper:
                            included[conformal_prediction_CI_coverage] += 1
                    else:
                        if y_test[test_point_idx, 0] <= CI_upper:
                            included[conformal_prediction_CI_coverage] += 1
            for conformal_prediction_CI_coverage \
                    in conformal_prediction_CI_coverage_range:
                coverages[conformal_prediction_CI_coverage].append(
                    included[conformal_prediction_CI_coverage] / len(test_indices))
                qhats[conformal_prediction_CI_coverage].append(
                    qhat[conformal_prediction_CI_coverage])

        coverages_np = []
        qhats_np = []
        for conformal_prediction_CI_coverage \
                in conformal_prediction_CI_coverage_range:
            print('Coverage: mean %0.8f, std dev %0.8f'
                  % (np.mean(coverages[conformal_prediction_CI_coverage]),
                     np.std(coverages[conformal_prediction_CI_coverage])))
            print('q_hat: mean %0.8f, std dev %0.8f'
                  % (np.mean(qhats[conformal_prediction_CI_coverage]),
                     np.std(qhats[conformal_prediction_CI_coverage])))
            coverages_np.append(
                np.insert(coverages[conformal_prediction_CI_coverage],
                          0, conformal_prediction_CI_coverage))
            qhats_np.append(np.insert(qhats[conformal_prediction_CI_coverage],
                                      0, conformal_prediction_CI_coverage))
        coverages_np = np.array(coverages_np)
        qhats_np = np.array(qhats_np)
        output_filename_prefix = \
            os.path.join(output_dir, 'split_conformal_prediction',
                         '%s_%s_exp%d_%s_bs%d_nep%d_nla%d_nno%d_lr%f_'
                         % (survival_estimator_name, dataset, experiment_idx,
                            val_string, batch_size, n_epochs, n_layers, n_nodes,
                            lr)
                         +
                         'nd%d_%s_calib%f'
                         % (num_durations, survival_time_estimator, calib_frac))
        np.savetxt(output_filename_prefix + '_coverages.txt',
                   coverages_np)
        np.savetxt(output_filename_prefix + '_qhats.txt',
                   qhats_np)

        print()

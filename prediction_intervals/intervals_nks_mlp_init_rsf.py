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
__package__ = 'prediction_intervals'

import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

import torchtuples as tt
from pycox.evaluation import EvalSurv
from neural_kernel_survival import NKS, NKSDiscrete, ResidualBlock, Scaler
from npsurvival_models import RandomSurvivalForest
from survival_datasets import load_dataset
from torchtuples import Model


init_survival_estimator_name = 'rsf'
survival_estimator_name = 'nks_mlp_init_rsf_full_train'
survival_estimator_display_name = 'NKS-MLP (init: RSF)'

if not (len(sys.argv) == 4 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" ' % sys.argv[0]
          + '[config file] [experiment index] [dataset]')
    sys.exit()

experiment_idx = int(sys.argv[2])
dataset = sys.argv[3]

config = configparser.ConfigParser()
config.read(sys.argv[1])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
output_dir = config['DEFAULT']['output_dir']
init_random_seed = int(config['method: %s'
                              % init_survival_estimator_name]['random_seed'])
method_header = 'method: %s' % survival_estimator_name
fine_tune_random_seed = int(config[method_header]['random_seed'])
mds_n_init = int(config['DEFAULT']['mds_n_init'])
mds_random_seed = int(config['DEFAULT']['mds_random_seed'])
conformal_prediction_CI_coverage_range = \
    ast.literal_eval(config['DEFAULT']['conformal_prediction_CI_coverage'])
conformal_prediction_random_seed = \
    int(config['DEFAULT']['conformal_prediction_random_seed'])
conformal_prediction_n_samples = \
    int(config['DEFAULT']['conformal_prediction_n_samples'])
max_n_cores = int(config['DEFAULT']['max_n_cores'])
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'split_conformal_prediction'),
            exist_ok=True)
os.makedirs(os.path.join(output_dir, 'weighted_split_conformal_prediction'),
            exist_ok=True)

n_jobs = min(max_n_cores, os.cpu_count())

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

output_best_cv_hyperparam_filename \
    = os.path.join(output_dir, 'train',
                   '%s_%s_exp%d_cv%d_best_cv_hyperparams.pkl'
                   % (survival_estimator_name, dataset, experiment_idx,
                      cross_val_n_folds))
if not os.path.isfile(output_best_cv_hyperparam_filename):
    raise Exception("File does not exist: %s\n\n"
                    % output_best_cv_hyperparam_filename
                    + 'Running the benchmark demo first should resolve this.')

with open(output_best_cv_hyperparam_filename, 'rb') as pickle_file:
    best_cv_hyperparams = pickle.load(pickle_file)
arg_max_cindex, max_cindex = best_cv_hyperparams['cindex_td']
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

print('Pre-training...')
model_filename = \
    os.path.join(output_dir, 'models',
                 '%s_%s_exp%d_mf%d_msl%d_km%d_test.pkl'
                 % (init_survival_estimator_name, dataset,
                    experiment_idx, max_features,
                    min_samples_leaf, use_km))
if not os.path.isfile(model_filename):
    surv_model = \
        RandomSurvivalForest(n_estimators=100,
                             max_features=max_features,
                             max_depth=None,
                             oob_score=False,
                             feature_importance=False,
                             min_samples_leaf=min_samples_leaf,
                             random_state=init_random_seed,
                             n_jobs=n_jobs)
    surv_model.fit(X_train, y_train)
    surv_model.save(model_filename)
else:
    surv_model = RandomSurvivalForest.load(model_filename)

print('*** Extracting proximity matrix...')
prox_filename = model_filename[:-4] + '_prox_matrix.txt'
if not os.path.isfile(prox_filename):
    leaf_ids = surv_model.predict_leaf_ids(X_train)
    n = len(X_train)
    prox_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            prox_matrix[i, j] = \
                np.mean(leaf_ids[i] == leaf_ids[j])
            prox_matrix[j, i] = prox_matrix[i, j]
    np.savetxt(prox_filename, prox_matrix)
else:
    prox_matrix = np.loadtxt(prox_filename)

del surv_model
gc.collect()

print('*** Computing MDS embedding...')
kernel_matrix = np.clip(prox_matrix + 1e-7, 0., 1.)
rsf_dists = np.sqrt(-np.log(kernel_matrix))
mds_size = min(len(X_train), X_train.shape[1])
mds_filename = model_filename[:-4] + '_mds%d.txt' % mds_size
if not os.path.isfile(mds_filename):
    mds = MDS(n_components=mds_size,
              metric=True,
              n_init=mds_n_init,
              n_jobs=n_jobs,
              random_state=mds_random_seed,
              dissimilarity='precomputed')
    mds_embedding = mds.fit_transform(rsf_dists)
    np.savetxt(mds_filename, mds_embedding)
else:
    mds_embedding = np.loadtxt(mds_filename)
mds_embedding = mds_embedding.astype('float32')

print()


print('*** Fitting neural net to MDS transformation...')
torch.manual_seed(fine_tune_random_seed)
np.random.seed(fine_tune_random_seed)
batch_norm = True
dropout = 0.
output_bias = True
optimizer = tt.optim.Adam(lr=lr)
net = tt.practical.MLPVanilla(X_train_std.shape[1],
                              [n_nodes for layer_idx
                               in range(n_layers)],
                              mds_size,
                              batch_norm,
                              dropout,
                              output_bias=output_bias)
loss = nn.MSELoss()
emb_model = Model(net, loss, optimizer)

emb_model_filename = \
    os.path.join(output_dir, 'models',
                 'rsf_full_train_neural_approx_'
                 +
                 '%s_exp%d_mf%d_msl%d_km%d_'
                 % (dataset, experiment_idx,
                    max_features, min_samples_leaf, use_km)
                 +
                 'bs%d_nep%d_nla%d_nno%d_'
                 % (batch_size, 100, n_layers, n_nodes)
                 +
                 'lr%f_test.pt' % lr)
if not os.path.isfile(emb_model_filename):
    emb_model.fit(X_train_std, mds_embedding,
                  batch_size=batch_size, epochs=100,
                  verbose=False)
    emb_model.save_net(emb_model_filename)
else:
    emb_model.load_net(emb_model_filename)
emb_model.net.train()

print('*** Fine-tuning with DKSA...')
torch.manual_seed(fine_tune_random_seed + 1)
np.random.seed(fine_tune_random_seed + 1)
optimizer = tt.optim.Adam(lr=lr)
if num_durations > 0:
    labtrans = NKSDiscrete.label_transform(num_durations)
    y_train_discrete = labtrans.fit_transform(*y_train.T)
    surv_model = NKSDiscrete(emb_model.net, optimizer,
                             duration_index=labtrans.cuts)
else:
    surv_model = NKS(emb_model.net, optimizer)

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
                 'lr%f_nd%d_test.pt'
                 % (lr, num_durations))
if not os.path.isfile(model_filename):
    print('*** Fitting with hyperparam:', hyperparam, flush=True)
    if num_durations > 0:
        surv_model.fit(X_train_std, y_train_discrete,
                       batch_size, n_epochs, verbose=False)
    else:
        surv_model.fit(X_train_std,
                       (y_train[:, 0],
                        y_train[:, 1]), batch_size,
                       n_epochs, verbose=False)
    surv_model.save_net(model_filename)
else:
    print('*** Loading ***', flush=True)
    surv_model.load_net(model_filename)


if num_durations > 0:
    surv_df = surv_model.interpolate(10).predict_surv_df(X_test_std)
else:
    surv_df = surv_model.predict_surv_df(X_test_std)
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
                         '%s_%s_exp%d_bs%d_nep%d_nla%d_nno%d_lr%f_'
                         % (survival_estimator_name, dataset, experiment_idx,
                            batch_size, n_epochs, n_layers, n_nodes, lr)
                         +
                         'nd%d_%s_calib%f'
                         % (num_durations, survival_time_estimator, calib_frac))
        np.savetxt(output_filename_prefix + '_coverages.txt',
                   coverages_np)
        np.savetxt(output_filename_prefix + '_qhats.txt',
                   qhats_np)

        print()


embeddings = surv_model.compute_embeddings(X_test_std)
from scipy.spatial.distance import pdist, squareform
sq_dists = squareform(pdist(embeddings, 'sqeuclidean'))
kernel_weights = np.exp(-sq_dists)

max_conformal_prediction_CI_coverage = \
    np.max(conformal_prediction_CI_coverage_range)
sorted_conformal_prediction_CI_coverage_range = \
    np.sort(conformal_prediction_CI_coverage_range)

for survival_time_estimator in ['mean', 'median']:
    if survival_time_estimator == 'mean':
        predicted_survival_times = mean_predicted_survival_times.copy()
    else:
        predicted_survival_times = median_predicted_survival_times.copy()

    for calib_frac in np.linspace(0.1, 1, 10):
        print('[Weighted split conformal prediction - %s, calib frac %f]'
              % (survival_time_estimator, calib_frac))

        rng = np.random.RandomState(conformal_prediction_random_seed)

        coverages = {conformal_prediction_CI_coverage: []
                     for conformal_prediction_CI_coverage
                     in conformal_prediction_CI_coverage_range}
        qhats = {conformal_prediction_CI_coverage: []
                 for conformal_prediction_CI_coverage
                 in conformal_prediction_CI_coverage_range}

        for repeat_idx in range(conformal_prediction_n_samples):
            print(repeat_idx)
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
            sort_indices = np.argsort(nonconformity_scores)
            sorted_nonconformity_scores = \
                np.append(nonconformity_scores[sort_indices], np.inf)

            for test_point_idx in test_indices[:100]:
                weights = kernel_weights[test_point_idx][calibration_indices]
                sampling_probs = kernel_weights[test_point_idx][test_indices]
                sampling_probs /= sampling_probs.sum()

                included = {conformal_prediction_CI_coverage: 0
                            for conformal_prediction_CI_coverage
                            in conformal_prediction_CI_coverage_range}
                for test_point_idx2 in rng.choice(test_indices, size=100,
                                                  p=sampling_probs):
                    sorted_weights = np.append(weights[sort_indices],
                                               kernel_weights[test_point_idx2,
                                                              test_point_idx])
                    sorted_weights /= sorted_weights.sum()
                    cum_prob = 0.
                    cur_CI_coverage_idx = 0
                    cur_CI_coverage = \
                        sorted_conformal_prediction_CI_coverage_range[
                            cur_CI_coverage_idx]
                    qhat = {}
                    for jhat in range(len(sorted_nonconformity_scores)):
                        cum_prob += sorted_weights[jhat]
                        if cum_prob >= cur_CI_coverage:
                            qhat[cur_CI_coverage] = \
                                sorted_nonconformity_scores[jhat]
                            if cur_CI_coverage == \
                                    max_conformal_prediction_CI_coverage:
                                break
                            cur_CI_coverage_idx += 1
                            cur_CI_coverage = \
                                sorted_conformal_prediction_CI_coverage_range[
                                    cur_CI_coverage_idx]

                    for conformal_prediction_CI_coverage \
                            in conformal_prediction_CI_coverage_range:
                        if conformal_prediction_CI_coverage in qhat:
                            CI_lower = \
                                predicted_survival_times[test_point_idx2] \
                                - qhat[conformal_prediction_CI_coverage]
                            CI_upper = \
                                predicted_survival_times[test_point_idx2] \
                                + qhat[conformal_prediction_CI_coverage]
                            qhats[conformal_prediction_CI_coverage].append(
                                qhat[conformal_prediction_CI_coverage])
                        else:
                            CI_lower = -np.inf
                            CI_upper = np.inf
                            qhats[conformal_prediction_CI_coverage].append(
                                np.inf)

                        if y_test[test_point_idx2, 1] > 0:
                            if y_test[test_point_idx2, 0] >= CI_lower \
                                    and y_test[test_point_idx2, 0] <= CI_upper:
                                included[conformal_prediction_CI_coverage] += 1
                        else:
                            if y_test[test_point_idx2, 0] <= CI_upper:
                                included[conformal_prediction_CI_coverage] += 1

                for conformal_prediction_CI_coverage \
                        in conformal_prediction_CI_coverage_range:
                    coverages[conformal_prediction_CI_coverage].append(
                        included[conformal_prediction_CI_coverage] / 100)

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
            os.path.join(output_dir, 'weighted_split_conformal_prediction',
                         '%s_%s_exp%d_bs%d_nep%d_nla%d_nno%d_lr%f_'
                         % (survival_estimator_name, dataset, experiment_idx,
                            batch_size, n_epochs, n_layers, n_nodes, lr)
                         +
                         'nd%d_%s_calib%f'
                         % (num_durations, survival_time_estimator, calib_frac))
        np.savetxt(output_filename_prefix + '_coverages.txt',
                   coverages_np)
        np.savetxt(output_filename_prefix + '_qhats.txt',
                   qhats_np)

        print()

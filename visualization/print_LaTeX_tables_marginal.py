import ast
import configparser
import os
import sys
from glob import glob

import numpy as np
import scipy
import scipy.stats

if not (len(sys.argv) == 3 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file] [survival estimators file]'
          % sys.argv[0])
    sys.exit()

experiment_idx = 0

config = configparser.ConfigParser()
config.read(sys.argv[1])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
# coverage_range = \
#     ast.literal_eval(config['DEFAULT']['conformal_prediction_CI_coverage'])
coverage_range = [0.6, 0.7, 0.8, 0.9, 0.95]
output_dir = config['DEFAULT']['output_dir']

survival_estimator_names = []
estimator_display_names = []
with open(sys.argv[2], 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if not line.startswith('#'):
            pieces = line.split(' : ')
            if len(pieces) == 2:
                survival_estimator_names.append(pieces[0])
                estimator_display_names.append(pieces[1])


estimator_display_name_to_LaTeX = {
    'Cox': "\\textsc{cox}",
    'RSF': "\\textsc{rsf}",
    'DeepSurv': "\\textsc{deepsurv}",
    'DeepHit': "\\textsc{deephit}",
    'MTLR': "\\textsc{mtlr}",
    'Nnet-survival': "\\textsc{nnet-survival}",
    'Cox-CC': "\\textsc{cox-cc}",
    'Cox-Time': "\\textsc{cox-time}",
    'PC-Hazard': "\\textsc{pc-hazard}",
    'NKS-Basic': "\\textsc{nks-basic}",
    'NKS-Diag': "\\textsc{nks-diag}",
    'NKS-Res-Basic': "\\textsc{nks-res-basic}",
    'NKS-Res-Diag': "\\textsc{nks-res-diag}",
    'NKS-MLP': "\\textsc{nks-mlp}",
    'NKS-MLP (init: RSF)': "\\textsc{nks-mlp} (init: \\textsc{rsf})",
    'NKS-MLP (init: DeepHit)': "\\textsc{nks-mlp} (init: \\textsc{deephit})",
}

# method for translating survival curve into single survival time estimate
survival_time_method = 'median'

# use all calibration data
calib_frac = 1.0

n_estimators = len(survival_estimator_names)
n_datasets = len(datasets)
for what in ['qhats', 'coverages']:
    for target_coverage in coverage_range:
        centers = np.zeros((n_estimators, n_datasets))
        spreads = np.zeros((n_estimators, n_datasets))

        for dataset_idx, dataset in enumerate(datasets):
            for estimator_idx, survival_estimator_name \
                    in enumerate(survival_estimator_names):
                matches = list(glob(
                    os.path.join(
                        output_dir,
                        'split_conformal_prediction',
                        '%s_%s_exp%d_*%s_calib%f_%s.txt'
                        % (survival_estimator_name, dataset, experiment_idx,
                           survival_time_method, calib_frac, what))))
                if len(matches) != 1:
                    print("\n".join(matches))
                assert len(matches) == 1
                match = matches[0]
                coverages = np.loadtxt(match)
                for row in coverages:
                    if row[0] == target_coverage:
                        if what == 'qhats':
                            if dataset == 'support2_onehot':
                                centers[estimator_idx, dataset_idx] \
                                    = np.mean(row[1:]) / 30.42 * 2
                                spreads[estimator_idx, dataset_idx] \
                                    = np.std(row[1:]) / 30.42 * 2
                            else:
                                centers[estimator_idx, dataset_idx] \
                                    = np.mean(row[1:]) * 2
                                spreads[estimator_idx, dataset_idx] \
                                    = np.std(row[1:]) * 2
                        else:
                            centers[estimator_idx, dataset_idx] \
                                = np.mean(row[1:])
                            spreads[estimator_idx, dataset_idx] \
                                = np.std(row[1:])

        if what == 'coverages':
            print('[Empirical coverages for target coverage level %f]'
                  % target_coverage)
            print()
            row_strings = []
            for estimator_idx in range(n_estimators):
                row_string = \
                    estimator_display_name_to_LaTeX[
                        estimator_display_names[estimator_idx]]
                for dataset_idx in range(n_datasets):
                    row_string += \
                        " & $%0.3f\\pm%0.3f$" \
                        % (centers[estimator_idx, dataset_idx],
                           spreads[estimator_idx, dataset_idx])
                row_string += " \\tabularnewline"
                row_strings.append(row_string)
            print("\n\\hline\n".join(row_strings))

        elif what == 'qhats':

            print('[Prediction interval widths for target coverage level %f]'
                  % target_coverage)
            print()

            # figure out what estimator achieves the minimum q_hat per dataset;
            # we will be bolding these numbers
            arg_mins = [np.argmin(centers[:, dataset_idx])
                        for dataset_idx in range(n_datasets)]

            # note: we multiply the numbers all by 2 in the table to get widths
            # rather than radii

            row_strings = []
            for estimator_idx in range(n_estimators):
                row_string = \
                    estimator_display_name_to_LaTeX[
                        estimator_display_names[estimator_idx]]
                for dataset_idx in range(n_datasets):
                    if estimator_idx == arg_mins[dataset_idx]:
                        row_string += \
                            " & $\\mathbf{%0.3f}\\pm%0.3f$" \
                            % (centers[estimator_idx, dataset_idx],
                               spreads[estimator_idx, dataset_idx])
                    else:
                        row_string += \
                            " & $%0.3f\\pm%0.3f$" \
                            % (centers[estimator_idx, dataset_idx],
                               spreads[estimator_idx, dataset_idx])
                row_string += " \\tabularnewline"
                row_strings.append(row_string)
            print("\n\\hline\n".join(row_strings))
            print()

        print()
        print()


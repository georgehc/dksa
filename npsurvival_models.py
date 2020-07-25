"""
Nonparametric survival estimators

Author: George H. Chen (georgechen [at symbol] cmu.edu)

This file contains the following classes and various helper functions (all of
these implement both Kaplan-Meier and Nelson-Aalen versions):
- BasicSurvival : basic Kaplan-Meier and Nelson-Aalen estimators that do not
  account for feature vectors
- KNNSurvival : k-NN survival estimation
- KNNWeightedSurvival: weighted k-NN survival estimation
- KernelSurvival : kernel survival estimation
- RandomSurvivalForest : a heavily modified version of Wrymm's random survival
  forest code (the version last updated Feb 28, 2017)
  [https://github.com/Wrymm/Random-Survival-Forests]; changes are discussed
  below
- RandomSurvivalForestANN : kernel survival estimation where the kernel is
  learned using a random survival forest (ANN stands for "adaptive nearest
  neighbors"; one can interpret this is either an adaptive kernel method or an
  adaptive nearest neighbors method where the neighbors are weighted)
- CDFRegressionKNNWeightedSurvival : implements the "cdf-reg" two-step method
  mentioned in the ICML paper:

      George H. Chen. Nearest Neighbor and Kernel Survival Analysis:
      Nonasymptotic Error Bounds and Strong Consistency Rates. ICML 2019.

Random survival forests are by Hemant Ishwaran, Udaya B. Kogalur, Eugene H.
Blackstone, and Michael S. Lauer: "Random survival forests" (Annals of Applied
Stats 2008); see also Ishwaran and Kogalur's "Random survival forests for R"
article in Rnews (2007) and their R package "randomForestSRC".

Setup
-----
Be sure to compile the cython code by running:

    python setup_random_survival_forest_cython.py build_ext --inplace

* * * * *

Main changes to Wrymm's code (the version last updated Feb 28, 2017):
- the log-rank splitting score denominator calculation appeared to be missing a
  Y_i factor (prior to taking the square root); this has been fixed
- the log-rank splitting score code is implemented in cython
- Wrymm's code only splits on medians of feature values rather than optimizing
  for the best split; I have added both an exhaustive split option (tries
  every split threshold among the observed feature values) and a random split
  option (Ishwaran et al suggest in their Annals of Applied Stats paper that
  this randomized strategy actually works quite well)
- Wrymm's code has `min_samples_split` refer to what scikit-learn calls
  `min_samples_leaf`; I switched the variable name to match that of
  scikit-learn and also introduced what scikit-learn calls `min_samples_split`
  as a parameter
- many survival probabilities are computed at once for a given feature vector
  (i.e., rather than computing the probability of a subject surviving beyond
  one choice of time, compute the probabilities of a subject surviving beyond a
  collection of different times)
- added code to predict subject-specific cumulative hazard functions
- randomization can now be made deterministic by providing either an integer
  random seed or a numpy RandomState instance
- pandas has been removed to speed up the code
- parallelism is now supported both in fitting and prediction
"""
from collections import Counter
import functools
import pickle

import numpy as np
from joblib import Parallel, delayed
from lifelines.utils import concordance_index
from sklearn.neighbors import NearestNeighbors

from random_survival_forest_cython import logrank


class RandomSurvivalForest():
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, split='logrank',
                 split_threshold_mode='exhaustive', random_state=None,
                 n_jobs=None, oob_score=False, feature_importance=False):
        """
        A random survival forest survival probability estimator. This is very
        similar to the usual random forest that is used for regression and
        classification. However, in a random survival forest, the prediction
        task is to estimate the survival probability function for a test
        feature vector. Training data can have right-censoring. For details,
        see any introductory text on survival analysis.

        Parameters
        ----------
        n_estimators : int, optional (default=100)
            Number of trees.

        max_features : int, string, optional (default='sqrt')
            Number of features chosen per tree. Allowable string choices are
            'sqrt' (max_features=ceil(sqrt(n_features))) and 'log2'
            (max_features=ceil(log2(n_features))).

        max_depth : int, optional (default=None)
            Maximum depth of each tree. If None, then each tree is grown
            until other termination criteria are met (see `min_samples_split`
            and `min_samples_leaf` parameters).

        min_samples_split : int, optional (default=2)
            A node must have at least this many samples to be split.

        min_samples_leaf : int, float, optional (default=1)
            Both sides of a split must have at least this many samples
            (or in the case of a fraction, at least a fraction of samples)
            for the split to happen. Otherwise, the node is turned into a
            leaf node.

        split : string, optional (default='logrank')
            Currently only the log-rank splitting criterion is supported.

        split_threshold_mode : string, optional (default='exhaustive')
            If 'exhaustive', then we compute the split score for every observed
            feature value as a possible threshold (this can be very expensive).
            If 'median', then for any feature, we always split on the median
            value observed for that feature (this is the only supported option
            in Wrymm's original random survival analysis code).
            If 'random', then for any feature, we randomly choose a split
            threshold among the observed feature values (this is recommended by
            the random survival forest authors if fast computation is desired).

        random_state : int, numpy RandomState instance, None, optional
            (default=None)
            If an integer, then a new numpy RandomState is created with the
            integer as the random seed. If a numpy RandomState instance is
            provided, then it is used as the pseudorandom number generator. If
            None is specified, then a new numpy RandomState is created without
            providing a seed.

        n_jobs : int, None, optional (default=None)
            Number of cores to use with joblib's Parallel. This is the same
            `n_jobs` parameter as for Parallel. Setting `n_jobs` to -1 uses all
            the cores.

        oob_score : boolean, optional (default=False)
            Whether to compute an out-of-bag (OOB) accuracy estimate (as with
            the original random survival forest paper, this is done using
            c-index with cumulative hazard estimates). The OOB estimate is
            computed during model fitting (via fit()), and the resulting
            c-index estimate is stored in the attribute `oob_score_`.

        feature_importance : boolean, optional (default=False)
            Whether to compute feature importances (requires `oob_score` to
            be set to True). Feature importances are computed during the
            model fitting (via fit()), and the resulting feature importances is
            stored in the attribute `feature_importances_`.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.split_threshold_mode = split_threshold_mode
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.feature_importance = feature_importance
        self.column_names = None
        self.oob_score_ = None
        self.feature_importances_ = None

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif type(random_state) == int:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        if split == 'logrank':
            self.split_score_function = logrank
        else:
            raise NotImplementedError('Unsupported split criterion '
                                      + '"{0}"'.format(split))

    def save(self, filename):
        data = {'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'split_threshold_mode': self.split_threshold_mode,
                'n_jobs': self.n_jobs,
                'oob_score': self.oob_score,
                'feature_importance': self.feature_importance,
                'column_names': list(self.column_names),
                'oob_score_': self.oob_score_}

        if self.feature_importances_ is not None:
            data['feature_importances_'] = self.feature_importances_.tolist()
        else:
            data['feature_importances_'] = None

        data['trees'] = \
            [_convert_to_not_use_numpy(tree) for tree in self.trees]

        data['tree_bootstrap_indices'] = \
            [indices.tolist() for indices in self.tree_bootstrap_indices]

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        rsf = \
            RandomSurvivalForest(n_estimators=data['n_estimators'],
                                 max_features=data['max_features'],
                                 max_depth=data['max_depth'],
                                 min_samples_split=data['min_samples_split'],
                                 min_samples_leaf=data['min_samples_leaf'],
                                 split='logrank',
                                 split_threshold_mode='exhaustive',
                                 random_state=None,
                                 n_jobs=data['n_jobs'],
                                 oob_score=data['oob_score'],
                                 feature_importance=data['feature_importance'])

        rsf.column_names = data['column_names']
        rsf.oob_score_ = data['oob_score_']

        if data['feature_importances_'] is None:
            rsf.feature_importances_ = None
        else:
            rsf.feature_importances_ = np.array(data['feature_importances_'])

        rsf.trees = [_convert_to_use_numpy(tree) for tree in data['trees']]
        rsf.tree_bootstrap_indices = \
            np.array([indices for indices in data['tree_bootstrap_indices']])

        for tree in rsf.trees:
            _label_leaves(tree)

        return rsf

    def fit(self, X, y, column_names=None):
        """
        Fits the random survival forest to training data.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        column_names : list, None, optional (default=None)
            Names for features can be specified. This is only for display
            purposes when using the `draw` method. If set to None, then
            `column_names` is just set to be a range of integers indexing the
            columns from 0.

        Returns
        -------
        None
        """
        if column_names is None:
            self.column_names = list(range(X.shape[1]))
        else:
            self.column_names = column_names
            assert len(column_names) == X.shape[1]

        if type(self.max_features) == str:
            if self.max_features == 'sqrt':
                max_features = int(np.ceil(np.sqrt(X.shape[1])))
            elif self.max_features == 'log2':
                max_features = int(np.ceil(np.log2(X.shape[1])))
            else:
                raise NotImplementedError('Unsupported max features choice '
                                          + '"{0}"'.format(self.max_features))
        else:
            max_features = self.max_features

        self.tree_bootstrap_indices = []
        sort_indices = np.argsort(y[:, 0])
        X = X[sort_indices].astype(np.float)
        y = y[sort_indices].astype(np.float)
        random_state = self.random_state
        for tree_idx in range(self.n_estimators):
            bootstrap_indices = np.sort(random_state.choice(X.shape[0],
                                                            X.shape[0],
                                                            replace=True))
            self.tree_bootstrap_indices.append(bootstrap_indices)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            self.trees = \
                parallel(
                  delayed(_build_tree)(
                      X[self.tree_bootstrap_indices[tree_idx]],
                      y[self.tree_bootstrap_indices[tree_idx]],
                      0, self.max_depth, max_features,
                      self.split_score_function, self.min_samples_split,
                      self.min_samples_leaf, self.split_threshold_mode,
                      np.random.RandomState(random_state.randint(4294967296)))
                  for tree_idx in range(self.n_estimators))

            if self.oob_score:
                parallel_args = []
                oob_masks = []
                for tree_idx, bootstrap_indices \
                        in enumerate(self.tree_bootstrap_indices):
                    oob_mask = np.ones(X.shape[0], dtype=np.bool)
                    for idx in bootstrap_indices:
                        oob_mask[idx] = 0
                    if oob_mask.sum() > 0:
                        X_oob = X[oob_mask]
                        if len(X_oob.shape) == 1:
                            X_oob = X_oob.reshape(1, -1)
                        parallel_args.append((tree_idx, X_oob))
                        oob_masks.append(
                            (oob_mask,
                             {original_idx: new_idx
                              for new_idx, original_idx
                              in enumerate(np.where(oob_mask)[0])}))

                sorted_unique_times = np.unique(y[:, 0])
                results = parallel(
                    delayed(_predict_tree)(
                        self.trees[tree_idx], 'cum_haz', X_oob,
                        sorted_unique_times, True)
                    for (tree_idx, X_oob) in parallel_args)

                num_unique_times = len(sorted_unique_times)
                cum_hazard_scores = []
                oob_y = []
                for idx in range(X.shape[0]):
                    num = 0.
                    den = 0.
                    for tree_idx2, (oob_mask, forward_map) \
                            in enumerate(oob_masks):
                        if oob_mask[idx]:
                            num += results[tree_idx2][forward_map[idx]].sum()
                            den += 1
                    if den > 0:
                        cum_hazard_scores.append(num / den)
                        oob_y.append(y[idx])

                cum_hazard_scores = np.array(cum_hazard_scores)
                oob_y = np.array(oob_y)

                self.oob_score_ = concordance_index(oob_y[:, 0],
                                                    -cum_hazard_scores,
                                                    oob_y[:, 1])

                if self.feature_importance:
                    self.feature_importances_ = []
                    for col_idx in range(X.shape[1]):
                        vimp_results = \
                            parallel(
                                delayed(_predict_tree_vimp)(
                                    self.trees[tree_idx], 'cum_haz',
                                    X_oob, sorted_unique_times, True,
                                    col_idx,
                                    np.random.RandomState(
                                        random_state.randint(4294967296)))
                                for (tree_idx, X_oob)
                                in parallel_args)

                        cum_hazard_scores = []
                        oob_y = []
                        for idx in range(X.shape[0]):
                            num = 0.
                            den = 0.
                            for tree_idx2, (oob_mask, forward_map) \
                                    in enumerate(oob_masks):
                                if oob_mask[idx]:
                                    num += vimp_results[tree_idx2][
                                        forward_map[idx]].sum()
                                    den += 1
                            if den > 0:
                                cum_hazard_scores.append(num / den)
                                oob_y.append(y[idx])

                        if len(cum_hazard_scores) > 0:
                            cum_hazard_scores = np.array(cum_hazard_scores)
                            oob_y = np.array(oob_y)

                            vimp = self.oob_score_ - \
                                concordance_index(oob_y[:, 0],
                                                  -cum_hazard_scores,
                                                  oob_y[:, 1])
                        else:
                            vimp = np.nan
                        self.feature_importances_.append(vimp)
                    self.feature_importances_ \
                        = np.array(self.feature_importances_)

        for tree in self.trees:
            _label_leaves(tree)

    def predict_leaf_ids(self, X):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_tree_leaf_id)(self.trees[tree_idx], X)
            for tree_idx in range(self.n_estimators))
        return np.array(results).T

    def predict_surv(self, X, times, presorted_times=False,
                     use_kaplan_meier=True):
        """
        Computes the forest's survival probability function estimate for each
        feature vector evaluated at user-specified times.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        use_kaplan_meier : boolean, optional (default=True)
            In the original random survival forests paper, only the cumulative
            hazard function H(t|x) is predicted from the leafs rather than the
            survival function S(t|x). One can back out the survival function
            from the cumulative hazard function since S(t|x)=exp(-H(t|x)).
            If this flag is set to True, then we have the forest predict S(t|x)
            using Kaplan-Meier estimates at the leaves (instead of the
            default of predicting H(t|x) with Nelson-Aalen estimates at the
            leaves), and average the trees' S(t|x) estimates.

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if use_kaplan_meier:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_predict_tree)(self.trees[tree_idx], 'surv', X, times,
                                       presorted_times)
                for tree_idx in range(self.n_estimators))
            return functools.reduce(lambda x, y: x + y, results) \
                / self.n_estimators
        else:
            return np.exp(-self.predict_cum_haz(X, times, presorted_times))

    def predict_cum_haz(self, X, times, presorted_times=False,
                        use_kaplan_meier=False, surv_eps=1e-12):
        """
        Computes the forest's cumulative hazard function estimate for each
        feature vector evaluated at user-specified times.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        use_kaplan_meier : boolean, optional (default=False)
            In the original random survival forests paper, only the cumulative
            hazard function H(t|x) is predicted from the leafs rather than the
            survival function S(t|x). One can back out the cumulative hazard
            function from the survival function since H(t|x)=-log(S(t|x)).
            If this flag is set to True, then we have the forest predict S(t|x)
            first using Kaplan-Meier estimates at the leaves (instead of the
            default of predicting H(t|x) with Nelson-Aalen estimates at the
            leaves), and then we back out an estimate for H(t|x).

        surv_eps : float, optional (default=1e-12)
            If `use_kaplan_meier` is set to True, then we clip the estimated
            survival function so that any value less than `surv_eps` is set to
            `surv_eps`. This makes it so that when we take the negative log of
            the survival function, we don't take logs of 0.

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if use_kaplan_meier:
            surv = self.predict_surv(X, times, presorted_times, True)
            return -np.log(np.clip(surv, surv_eps, 1.))
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_predict_tree)(self.trees[tree_idx], 'cum_haz', X, times,
                                       presorted_times)
                for tree_idx in range(self.n_estimators))
            return functools.reduce(lambda x, y: x + y, results) \
                / self.n_estimators


    def _print_with_depth(self, string, depth):
        """
        Auxiliary function to print a string with indentation dependent on
        depth.
        """
        print("{0}{1}".format("    " * depth, string))

    def _print_tree(self, tree, current_depth=0):
        """
        Auxiliary function to print a survival tree.
        """
        if 'surv' in tree:
            self._print_with_depth(tree['times'], current_depth)
            return
        self._print_with_depth(
            "{0} > {1}".format(self.column_names[tree['feature']],
                               tree['threshold']),
            current_depth)
        self._print_tree(tree['left'], current_depth + 1)
        self._print_tree(tree['right'], current_depth + 1)

    def draw(self):
        """
        Prints out each tree of the random survival forest.
        """
        for tree_idx, tree in enumerate(self.trees):
            print("==========================================\nTree",
                  tree_idx)
            self._print_tree(tree)


class BasicSurvival():
    def __init__(self):
        self.tree = None

    def fit(self, y):
        self.tree = _fit_leaf(y)

    def predict_surv(self, times, presorted_times=False,
                     limit_from_left=False):
        """
        Computes the Kaplan-Meier survival probability function estimate at
        user-specified times.

        Parameters
        ----------
        times : 1D numpy array (default=None)
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        Returns
        -------
        output : 1D numpy array
            Survival probability function evaluated at each of the times
            specified in `times`.
        """
        return _predict_leaf(self.tree, 'surv', times, presorted_times,
                             limit_from_left)

    def predict_cum_haz(self, times, presorted_times=False,
                        limit_from_left=False):
        """
        Computes the Nelson-Aalen cumulative hazard function estimate at
        user-specified times.

        Parameters
        ----------
        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        Returns
        -------
        output : 1D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times`.
        """
        return _predict_leaf(self.tree, 'cum_haz', times, presorted_times,
                             limit_from_left)


class KNNSurvival():
    def __init__(self, *args, **kwargs):
        """
        Arguments are the same as for `sklearn.neighbors.NearestNeighbors`.
        The simplest usage of this class is to use a single argument, which is
        `n_neighbors` for the number of nearest neighbors (Euclidean distance
        is assumed in this case). If you want to parallelize across different
        search queries, use the `n_jobs` keyword parameter (-1 to use all
        cores). To use other distances and for other details, please refer to
        the documentation for sklearn's `NearestNeighbors` class.

        *Important:* The prediction methods for this class use unweighted
        k-nearest neighbors, where "k" is set equal to the `n_neighbors`
        parameter.
        """
        self.NN_index_args = args
        self.NN_index_kwargs = kwargs
        self.NN_index = None

    def fit(self, X, y):
        """
        Constructs a nearest-neighbor index given training data (so that for
        a future data point, we can use the nearest-neighbor index to quickly
        find what the closest training data are to the future point).

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        Returns
        -------
        None
        """
        self.train_y = y
        self.NN_index = NearestNeighbors(*self.NN_index_args,
                                         **self.NN_index_kwargs)
        self.NN_index.fit(X)

    def predict_surv(self, X, times, presorted_times=False,
                     limit_from_left=False, n_neighbors=None):
        """
        Computes the k-NN Kaplan-Meier survival probability function estimate
        at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                           return_distance=False)
        train_y = self.train_y
        return np.array([_predict_leaf(_fit_leaf(train_y[idx]), 'surv', times,
                                       presorted_times, limit_from_left)
                         for idx in indices])

    def predict_cum_haz(self, X, times, presorted_times=False,
                        limit_from_left=False, n_neighbors=None):
        """
        Computes the k-NN Nelson-Aalen cumulative hazard function estimate at
        user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                           return_distance=False)
        train_y = self.train_y
        return np.array([_predict_leaf(_fit_leaf(train_y[idx]), 'cum_haz',
                                       times, presorted_times, limit_from_left)
                         for idx in indices])


class KNNWeightedSurvival():
    def __init__(self, *args, **kwargs):
        """
        Arguments are the same as for `sklearn.neighbors.NearestNeighbors`.
        The simplest usage of this class is to use a single argument, which is
        `n_neighbors` for the number of nearest neighbors (Euclidean distance
        is assumed in this case). If you want to parallelize across different
        search queries, use the `n_jobs` keyword parameter (-1 to use all
        cores). To use other distances and for other details, please refer to
        the documentation for sklearn's `NearestNeighbors` class.

        *Important:* The prediction methods for this class use weighted
        k-nearest neighbors, where "k" is set equal to the `n_neighbors`
        parameter. The weights are specified through a kernel function K. In
        particular, the i-th nearest neighbor X_i for a test point x is given a
        weight of:
            K( (distance between x and X_i) / (distance between x and X_k) ).
        """
        self.NN_index_args = args
        self.NN_index_kwargs = kwargs
        self.NN_index = None

    def fit(self, X, y):
        """
        Constructs a nearest-neighbor index given training data (so that for
        a future data point, we can use the nearest-neighbor index to quickly
        find what the closest training data are to the future point).

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        Returns
        -------
        None
        """
        self.train_y = y
        self.NN_index = NearestNeighbors(*self.NN_index_args,
                                         **self.NN_index_kwargs)
        self.NN_index.fit(X)

    def predict_surv(self, X, times, presorted_times=False,
                     limit_from_left=False, n_neighbors=None,
                     kernel_function=None):
        """
        Computes the weighted k-NN Kaplan-Meier survival probability function
        estimate at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to unweighted k-NN
            survival analysis. If a function is specified, then the weighting
            function used is of the form
            "kernel(distance / distance to k-th nearest neighbor)".

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if kernel_function is None:
            kernel_function = lambda s: 1
        dists, indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                                  return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            max_dist = np.max(dist)
            weights = np.array([kernel_function(d / max_dist) for d in dist])
            zero_weight = (weights == 0)
            if zero_weight.sum() > 0:
                weights_subset = weights[~zero_weight]
                if weights_subset.size > 0:
                    output.append(
                        _predict_leaf(
                            _fit_leaf_weighted(train_y[idx[~zero_weight]],
                                               weights_subset),
                            'surv', times, presorted_times, limit_from_left))
                else:
                    output.append(np.ones(n_times))
            else:
                output.append(
                    _predict_leaf(
                        _fit_leaf_weighted(train_y[idx],
                                           weights),
                        'surv', times, presorted_times, limit_from_left))
        return np.array(output)

    def predict_cum_haz(self, X, times, presorted_times=False,
                        limit_from_left=False, n_neighbors=None,
                        kernel_function=None):
        """
        Computes the weighted k-NN Nelson-Aalen cumulative hazard function
        estimate at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to unweighted k-NN
            survival analysis. If a function is specified, then the weighting
            function used is of the form
            "kernel(distance / distance to k-th nearest neighbor)".

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if kernel_function is None:
            kernel_function = lambda s: 1
        dists, indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                                  return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            max_dist = np.max(dist)
            weights = np.array([kernel_function(d / max_dist) for d in dist])
            zero_weight = (weights == 0)
            if zero_weight.sum() > 0:
                weights_subset = weights[~zero_weight]
                if weights_subset.size > 0:
                    output.append(
                        _predict_leaf(
                            _fit_leaf_weighted(train_y[idx[~zero_weight]],
                                               weights_subset),
                            'cum_haz', times, presorted_times, limit_from_left))
                else:
                    output.append(np.zeros(n_times))
            else:
                output.append(
                    _predict_leaf(
                        _fit_leaf_weighted(train_y[idx],
                                           weights),
                        'cum_haz', times, presorted_times, limit_from_left))
        return np.array(output)


class KernelSurvival():
    def __init__(self, *args, **kwargs):
        """
        Arguments are the same as for `sklearn.neighbors.NearestNeighbors`.
        The simplest usage of this class is to use a single argument, which is
        `radius` for fixed-radius near-neighbor search (Euclidean distance is
        assumed in this case). Put another way, any training data point farther
        than `radius` away from a test point is assumed to contribute 0 weight
        toward prediction for the test point. If you want to parallelize across
        different search queries, use the `n_jobs` keyword parameter (-1 to use
        all cores). To use other distances and for other details, please refer
        to the documentation for sklearn's `NearestNeighbors` class.
        """
        self.NN_index_args = args
        self.NN_index_kwargs = kwargs
        self.NN_index = None

    def fit(self, X, y):
        """
        Constructs a nearest-neighbor index given training data (so that for
        a future data point, we can use the nearest-neighbor index to quickly
        find what the closest training data are to the future point).

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        Returns
        -------
        None
        """
        self.train_y = y
        self.NN_index = NearestNeighbors(*self.NN_index_args,
                                         **self.NN_index_kwargs)
        self.NN_index.fit(X)

    def predict_surv(self, X, times, presorted_times=False,
                     limit_from_left=False, radius=None,
                     kernel_function=None):
        """
        Computes the kernel Kaplan-Meier survival probability function estimate
        at user-specified times.

        *Important:* The default radius to use is whatever was specified in
        `args` or `kwargs` when creating an instance of this class!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        radius : float, None, optional (default=None)
            Neighbors farther than this distance from a test point have kernel
            weight 0.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to fixed-radius near
            neighbors kernel survival analysis (i.e., a box kernel that
            becomes 0 after `radius` distance away). If a function is
            specified, then the weighting function used is of the form
            "kernel(distance / radius)".

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if radius is None:
            radius = self.NN_index.radius
        if kernel_function is None:
            kernel_function = lambda s: 1  # box kernel (i.e., uniform weights)

        dists, indices = self.NN_index.radius_neighbors(X, radius=radius,
                                                        return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            if dist.size > 0:
                weights = np.array([kernel_function(d / radius) for d in dist])
                zero_weight = (weights == 0)
                if zero_weight.sum() > 0:
                    weights_subset = weights[~zero_weight]
                    if weights_subset.size > 0:
                        output.append(
                            _predict_leaf(
                                _fit_leaf_weighted(train_y[idx[~zero_weight]],
                                                   weights_subset),
                                'surv', times, presorted_times,
                                limit_from_left))
                    else:
                        output.append(np.ones(n_times))
                else:
                    output.append(
                        _predict_leaf(
                            _fit_leaf_weighted(train_y[idx],
                                               weights),
                            'surv', times, presorted_times, limit_from_left))
            else:
                output.append(np.ones(n_times))
        return np.array(output)

    def predict_cum_haz(self, X, times, presorted_times=False,
                        limit_from_left=False, radius=None,
                        kernel_function=None):
        """
        Computes the kernel Nelson-Aalen cumulative hazard function estimate at
        user-specified times.

        *Important:* The default radius to use is whatever was specified in
        `args` or `kwargs` when creating an instance of this class!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        radius : float, None, optional (default=None)
            Neighbors farther than this distance from a test point have kernel
            weight 0.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to fixed-radius near
            neighbors kernel survival analysis (i.e., a box kernel that
            becomes 0 after `radius` distance away). If a function is
            specified, then the weighting function used is of the form
            "kernel(distance / radius)".

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times specified
            in `times` for each feature vector. The i-th row corresponds to the
            i-th feature vector.
        """
        if radius is None:
            radius = self.NN_index.radius
        if kernel_function is None:
            kernel_function = lambda s: 1  # box kernel (i.e., uniform weights)

        dists, indices = self.NN_index.radius_neighbors(X, radius=radius,
                                                        return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            if dist.size > 0:
                weights = np.array([kernel_function(d / radius) for d in dist])
                zero_weight = (weights == 0)
                if zero_weight.sum() > 0:
                    weights_subset = weights[~zero_weight]
                    if weights_subset.size > 0:
                        output.append(
                            _predict_leaf(
                                _fit_leaf_weighted(train_y[idx[~zero_weight]],
                                                   weights_subset),
                                'cum_haz', times, presorted_times,
                                limit_from_left))
                    else:
                        output.append(np.zeros(n_times))
                else:
                    output.append(
                        _predict_leaf(
                            _fit_leaf_weighted(train_y[idx],
                                               weights),
                            'cum_haz', times, presorted_times,
                            limit_from_left))
            else:
                output.append(np.zeros(n_times))
        return np.array(output)


class RandomSurvivalForestANN():
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, split='logrank',
                 split_threshold_mode='exhaustive', random_state=None,
                 n_jobs=None):
        """
        A modified version of the random survival forest survival probability
        estimator. From a theoretical standpoint, tree construction works the
        same way so each survival tree is associated with the same training
        subjects as regular random survival forests. However, what needs to be
        stored at each leaf is different in that instead of computing survival
        probability or cumulative hazard function estimates per tree, we
        instead use the learned tree only for identifying the adaptive nearest
        neighbors (which have weights!). These weighted nearest neighbors
        found per test point are then used to make a survival probability or
        cumulative hazard function estimate using kernel variants of the
        Kaplan-Meier and Nelson-Aalen estimators.

        Parameters
        ----------
        n_estimators : int, optional (default=100)
            Number of trees.

        max_features : int, string, optional (default='sqrt')
            Number of features chosen per tree. Allowable string choices are
            'sqrt' (max_features=ceil(sqrt(n_features))) and 'log2'
            (max_features=ceil(log2(n_features))).

        max_depth : int, optional (default=None)
            Maximum depth of each tree. If None, then each tree is grown
            until other termination criteria are met (see `min_samples_split`
            and `min_samples_leaf` parameters).

        min_samples_split : int, optional (default=2)
            A node must have at least this many samples to be split.

        min_samples_leaf : int, float, optional (default=1)
            Both sides of a split must have at least this many samples
            (or in the case of a fraction, at least a fraction of samples)
            for the split to happen. Otherwise, the node is turned into a
            leaf node.

        split : string, optional (default='logrank')
            Currently only the log-rank splitting criterion is supported.

        split_threshold_mode : string, optional (default='exhaustive')
            If 'exhaustive', then we compute the split score for every observed
            feature value as a possible threshold (this can be very expensive).
            If 'median', then for any feature, we always split on the median
            value observed for that feature (this is the only supported option
            in Wrymm's original random survival analysis code).
            If 'random', then for any feature, we randomly choose a split
            threshold among the observed feature values (this is recommended by
            the random survival forest authors if fast computation is desired).

        random_state : int, numpy RandomState instance, None, optional
            (default=None)
            If an integer, then a new numpy RandomState is created with the
            integer as the random seed. If a numpy RandomState instance is
            provided, then it is used as the pseudorandom number generator. If
            None is specified, then a new numpy RandomState is created without
            providing a seed.

        n_jobs : int, None, optional (default=None)
            Number of cores to use with joblib's Parallel. This is the same
            `n_jobs` parameter as for Parallel. Setting `n_jobs` to -1 uses all
            the cores.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.split_threshold_mode = split_threshold_mode
        self.n_jobs = n_jobs
        self.column_names = None

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif type(random_state) == int:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        if split == 'logrank':
            self.split_score_function = logrank
        else:
            raise NotImplementedError('Unsupported split criterion '
                                      + '"{0}"'.format(split))

    def save(self, filename):
        data = {'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'split_threshold_mode': self.split_threshold_mode,
                'n_jobs': self.n_jobs,
                'oob_score': self.oob_score,
                'feature_importance': self.feature_importance,
                'column_names': list(self.column_names),
                'oob_score_': self.oob_score_}

        if self.feature_importances_ is not None:
            data['feature_importances_'] = self.feature_importances_.tolist()
        else:
            data['feature_importances_'] = None

        data['trees'] = \
            [_convert_to_not_use_numpy(tree) for tree in self.trees]

        data['tree_bootstrap_indices'] = \
            [indices.tolist() for indices in self.tree_bootstrap_indices]

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        rsf = \
            RandomSurvivalForest(n_estimators=data['n_estimators'],
                                 max_features=data['max_features'],
                                 max_depth=data['max_depth'],
                                 min_samples_split=data['min_samples_split'],
                                 min_samples_leaf=data['min_samples_leaf'],
                                 split='logrank',
                                 split_threshold_mode='exhaustive',
                                 random_state=None,
                                 n_jobs=data['n_jobs'],
                                 oob_score=data['oob_score'],
                                 feature_importance=data['feature_importance'])

        rsf.column_names = data['column_names']
        rsf.oob_score_ = data['oob_score_']

        if data['feature_importances_'] is None:
            rsf.feature_importances_ = None
        else:
            rsf.feature_importances_ = np.array(data['feature_importances_'])

        rsf.trees = [_convert_to_use_numpy(tree) for tree in data['trees']]
        rsf.tree_bootstrap_indices = \
            np.array([indices for indices in data['tree_bootstrap_indices']])

        return rsf

    def fit(self, X, y, column_names=None):
        """
        Fits the random survival forest to training data.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        column_names : list, None, optional (default=None)
            Names for features can be specified. This is only for display
            purposes when using the `draw` method. If set to None, then
            `column_names` is just set to be a range of integers indexing the
            columns from 0.

        Returns
        -------
        None
        """
        if column_names is None:
            self.column_names = list(range(X.shape[1]))
        else:
            self.column_names = column_names
            assert len(column_names) == X.shape[1]

        if type(self.max_features) == str:
            if self.max_features == 'sqrt':
                max_features = int(np.ceil(np.sqrt(X.shape[1])))
            elif self.max_features == 'log2':
                max_features = int(np.ceil(np.log2(X.shape[1])))
            else:
                raise NotImplementedError('Unsupported max features choice '
                                          + '"{0}"'.format(self.max_features))
        else:
            max_features = self.max_features

        self.tree_bootstrap_indices = []
        sort_indices = np.argsort(y[:, 0])
        X = X[sort_indices].astype(np.float)
        y = y[sort_indices].astype(np.float)
        self.train_y = y
        random_state = self.random_state
        for tree_idx in range(self.n_estimators):
            bootstrap_indices = np.sort(random_state.choice(X.shape[0],
                                                            X.shape[0],
                                                            replace=True))
            self.tree_bootstrap_indices.append(bootstrap_indices)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            self.trees = \
                parallel(
                  delayed(_build_tree_ANN)(
                      X[self.tree_bootstrap_indices[tree_idx]],
                      y[self.tree_bootstrap_indices[tree_idx]],
                      self.tree_bootstrap_indices[tree_idx],
                      0, self.max_depth, max_features,
                      self.split_score_function, self.min_samples_split,
                      self.min_samples_leaf, self.split_threshold_mode,
                      np.random.RandomState(random_state.randint(4294967296)))
                  for tree_idx in range(self.n_estimators))

    def predict_surv(self, X, times, presorted_times=False,
                     use_kaplan_meier=True):
        """
        Computes the forest's survival probability function estimate for each
        feature vector evaluated at user-specified times.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        use_kaplan_meier : boolean, optional (default=False)
            If this flag is set to True, then we have the forest predict S(t|x)
            using a conditional Kaplan-Meier estimator. Otherwise, we have the
            forest predict H(t|x) using a conditional Nelson-Aalen estimator
            and then back out an estimate of S(t|x) via S(t|x)=exp(-H(t|x)).

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if use_kaplan_meier:
            # step 1: find adaptive nearest neighbors
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_compute_tree_ANN)(self.trees[tree_idx], X)
                for tree_idx in range(self.n_estimators))

            # step 2: aggregate adaptive nearest neighbors
            output = []
            y = self.train_y
            for i in range(len(X)):
                histogram = Counter()
                total = 0
                for t in range(self.n_estimators):
                    for j in results[t][i]:
                        histogram[j] += 1
                    total += len(results[t][i])
                nearest_neighbors = sorted(histogram.keys())
                weights = [histogram[j] / total for j in nearest_neighbors]
                output.append(
                    _predict_leaf(
                        _fit_leaf_weighted(y[np.array(nearest_neighbors,
                                                      dtype=np.int)],
                                           np.array(weights)),
                        'surv', times, presorted_times))
            return np.array(output)
        else:
            return np.exp(-self.predict_cum_haz(X, times, presorted_times))

    def predict_cum_haz(self, X, times, presorted_times=False,
                        use_kaplan_meier=False, surv_eps=1e-12):
        """
        Computes the forest's cumulative hazard function estimate for each
        feature vector evaluated at user-specified times.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        use_kaplan_meier : boolean, optional (default=False)
            If this flag is set to True, then we have the forest predict S(t|x)
            first using a conditional Kaplan-Meier estimate and then back out
            an estimate of H(t|x) via H(t|x)=-log(S(t|x)).

        surv_eps : float, optional (default=1e-12)
            If `use_kaplan_meier` is set to True, then we clip the estimated
            survival function so that any value less than `surv_eps` is set to
            `surv_eps`. This makes it so that when we take the negative log of
            the survival function, we don't take logs of 0.

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if use_kaplan_meier:
            surv = self.predict_surv(X, times, presorted_times, True)
            return -np.log(np.clip(surv, surv_eps, 1.))
        else:
            # step 1: find adaptive nearest neighbors
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_compute_tree_ANN)(self.trees[tree_idx], X)
                for tree_idx in range(self.n_estimators))

            # step 2: aggregate adaptive nearest neighbors
            output = []
            y = self.train_y
            for i in range(len(X)):
                histogram = Counter()
                total = 0
                for t in range(self.n_estimators):
                    for j in results[t][i]:
                        histogram[j] += 1
                    total += len(results[t][i])
                nearest_neighbors = sorted(histogram.keys())
                weights = [histogram[j] / total for j in nearest_neighbors]
                output.append(
                    _predict_leaf(
                        _fit_leaf_weighted(y[np.array(nearest_neighbors,
                                                      dtype=np.int)],
                                           np.array(weights)),
                        'cum_haz', times, presorted_times))
            return np.array(output)


    def _print_with_depth(self, string, depth):
        """
        Auxiliary function to print a string with indentation dependent on
        depth.
        """
        print("{0}{1}".format("    " * depth, string))

    def _print_tree(self, tree, current_depth=0):
        """
        Auxiliary function to print a survival tree.
        """
        if 'surv' in tree:
            self._print_with_depth(tree['times'], current_depth)
            return
        self._print_with_depth(
            "{0} > {1}".format(self.column_names[tree['feature']],
                               tree['threshold']),
            current_depth)
        self._print_tree(tree['left'], current_depth + 1)
        self._print_tree(tree['right'], current_depth + 1)

    def draw(self):
        """
        Prints out each tree of the random survival forest.
        """
        for tree_idx, tree in enumerate(self.trees):
            print("==========================================\nTree",
                  tree_idx)
            self._print_tree(tree)


def _find_best_feature_split(X, y, max_features, split_score_function,
                             min_samples_split, min_samples_leaf,
                             split_threshold_mode, random_state):
    """
    Finds the best single feature to split on and the split threshold to use.

    Parameters
    ----------
    X : 2D numpy array, shape = [n_samples, n_features]
        Feature vectors.

    y : 2D numpy array, shape = [n_samples, 2]
        Survival labels (first column is for observed times, second column is
        for event indicators). The i-th row corresponds to the i-th row in `X`.

    max_features : int
        Number of randomly chosen features that we find a split for.

    split_score_function : function
        Function that computes a split score. Look at `logrank` for an example.

    min_samples_split : int
        See documentation for RandomSurvivalForest's `__init__` function.

    min_samples_leaf : int, float
        See documentation for RandomSurvivalForest's `__init__` function.

    split_threshold_mode : string
        See documentation for RandomSurvivalForest's `__init__` function.

    random_state : numpy RandomState instance
        Pseudorandom number generator.
        *Warning*: for this function, `random_state` actually does have to be a
        numpy RandomState instance. This is for computational efficiency
        reasons as to not keep having to sanity check the input.

    Returns
    -------
    None, or (feature column index as integer, split threshold as float, mask
    for which data go into the left branch)
    """
    num_features = X.shape[1]
    if max_features >= num_features:
        candidate_features = list(range(num_features))
    else:
        candidate_features = list(random_state.choice(num_features,
                                                      max_features,
                                                      replace=False))

    num_candidate_features = len(candidate_features)
    X_slice = X[:, candidate_features].copy()
    drop_features = []
    keep_feature_mask = np.ones(num_candidate_features, dtype=np.bool)
    for idx in range(num_candidate_features):
        nan_mask = np.isnan(X_slice[:, idx])
        num_nans = nan_mask.sum()
        if num_nans > 0:
            not_nan_mask = ~nan_mask
            if np.any(not_nan_mask):
                # impute
                X_slice[nan_mask, idx] = \
                    random_state.choice(X_slice[not_nan_mask, idx],
                                        num_nans)
            else:
                drop_features.append(idx)
                keep_feature_mask[idx] = 0
    num_drop_features = len(drop_features)
    num_candidate_features -= num_drop_features
    if num_candidate_features == 0:
        return None
    if num_drop_features > 0:
        X_slice = X_slice[:, keep_feature_mask]
        for idx in drop_features[::-1]:
            del candidate_features[idx]

    if split_threshold_mode == 'exhaustive':
        score_arg_pairs \
            = [(split_score_function(X_slice[:, idx], y, split_threshold,
                                     min_samples_split, min_samples_leaf),
                (col_idx,
                 split_threshold,
                 X_slice[:, idx] <= split_threshold))
               for idx, col_idx in enumerate(candidate_features)
               for split_threshold in np.sort(np.unique(X_slice[:, idx]))]
        argmax = np.argmax([score for score, arg in score_arg_pairs])
        best_score, best_arg = score_arg_pairs[argmax]
        if best_score == 0:
            return None
        else:
            return best_arg
    elif split_threshold_mode == 'median':
        max_score = -np.inf
        best_arg = None
        for idx, col_idx in enumerate(candidate_features):
            split_threshold = np.median(X_slice[:, idx])
            score = split_score_function(X_slice[:, idx], y, split_threshold,
                                         min_samples_split, min_samples_leaf)
            if score > max_score:
                max_score = score
                best_arg = (col_idx, split_threshold,
                            X_slice[:, idx] <= split_threshold)
        if max_score == 0:
            return None
        else:
            return best_arg
    elif split_threshold_mode == 'random':
        max_score = -np.inf
        best_arg = None
        for idx, col_idx in enumerate(candidate_features):
            split_threshold = random_state.choice(X_slice[:, idx])
            score = split_score_function(X_slice[:, idx], y, split_threshold,
                                         min_samples_split, min_samples_leaf)
            if score > max_score:
                max_score = score
                best_arg = (col_idx, split_threshold,
                            X_slice[:, idx] <= split_threshold)
        if max_score == 0:
            return None
        else:
            return best_arg
    else:
        raise NotImplementedError('Unsupported split threshold strategy '
                                  + '"{0}"'.format(split_threshold_mode))


def _fit_leaf(y):
    """
    Computes leaf node information given survival labels (observed times and
    event indicators).

    Parameters
    ----------
    y : 2D numpy array, shape=[n_samples, 2]
        The two columns correspond to observed times and event indicators.

    Returns
    -------
    tree : dictionary
        The leaf node information stored as a dictionary. Specifically, the
        key-value pairs of this dictionary are as follows:
        - 'times': stores the sorted unique observed times
        - 'event_counts': in the same order as `times`, the number of events
            at each unique observed time
        - 'at_risk_counts': in the same order as `times`, the number of
            subjects at risk at each unique observed time
        - 'surv': in the same order as `times`, the Kaplan-Meier survival
            probability estimate at each unique observed time
        - 'cum_haz': in the same order as `times`, the Nelson-Aalen cumulative
            hazard estimate at each unique observed time
    """
    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    sorted_unique_observed_times = np.unique(y[:, 0])
    num_unique_observed_times = len(sorted_unique_observed_times)
    time_to_idx = {time: idx
                   for idx, time in enumerate(sorted_unique_observed_times)}
    event_counts = np.zeros(num_unique_observed_times)
    dropout_counts = np.zeros(num_unique_observed_times)
    at_risk_counts = np.zeros(num_unique_observed_times)
    at_risk_counts[0] = len(y)

    for observed_time, event_ind in y:
        idx = time_to_idx[observed_time]
        if event_ind:
            event_counts[idx] += 1
        dropout_counts[idx] += 1

    for idx in range(num_unique_observed_times - 1):
        at_risk_counts[idx + 1] = at_risk_counts[idx] - dropout_counts[idx]

    event_mask = (event_counts > 0)
    if event_mask.sum() > 0:
        sorted_unique_observed_times = sorted_unique_observed_times[event_mask]
        event_counts = event_counts[event_mask]
        at_risk_counts = at_risk_counts[event_mask]
    else:
        sorted_unique_observed_times = np.zeros((1,))
        event_counts = np.zeros((1,))
        at_risk_counts = np.array([len(y)])

    hazard_func = event_counts / np.clip(at_risk_counts, 1e-12, None)
    surv_func = np.exp(np.cumsum(np.log(1. - hazard_func + 1e-12)))
    cum_haz_func = np.cumsum(hazard_func)

    return {'times': sorted_unique_observed_times,
            'event_counts': event_counts,
            'at_risk_counts': at_risk_counts,
            'surv': surv_func,
            'cum_haz': cum_haz_func}


def _fit_leaf_weighted(y, weights):
    """
    Computes leaf node information given survival labels (observed times and
    event indicators) that have weights. This is for computing kernel variants
    of the Kaplan-Meier and Nelson-Aalen estimators.

    Parameters
    ----------
    y : 2D numpy array, shape=[n_samples, 2]
        The two columns correspond to observed times and event indicators.

    weights : 1D numpy array, shape=[n_samples]
        Nonnegative weights; i-th weight corresponds to the i-th row in `y`.

    Returns
    -------
    tree : dictionary
        The leaf node information stored as a dictionary. Specifically, the
        key-value pairs of this dictionary are as follows:
        - 'times': stores the sorted unique observed times
        - 'event_counts': in the same order as `times`, the number of events
            at each unique observed time
        - 'at_risk_counts': in the same order as `times`, the number of
            subjects at risk at each unique observed time
        - 'surv': in the same order as `times`, the Kaplan-Meier survival
            probability estimate at each unique observed time
        - 'cum_haz': in the same order as `times`, the Nelson-Aalen cumulative
            hazard estimate at each unique observed time
    """
    if y.size == 0:
        return {'times': sorted_unique_observed_times,
                'event_counts': event_counts,
                'at_risk_counts': at_risk_counts,
                'surv': surv_func,
                'cum_haz': cum_haz_func}

    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    sorted_unique_observed_times = np.sort(np.unique(y[:, 0]))
    num_unique_observed_times = len(sorted_unique_observed_times)
    time_to_idx = {time: idx
                   for idx, time in enumerate(sorted_unique_observed_times)}
    event_counts = np.zeros(num_unique_observed_times)
    dropout_counts = np.zeros(num_unique_observed_times)
    at_risk_counts = np.zeros(num_unique_observed_times)
    at_risk_counts[0] = np.sum(weights)

    for (observed_time, event_ind), weight in zip(y, weights):
        idx = time_to_idx[observed_time]
        if event_ind:
            event_counts[idx] += weight
        dropout_counts[idx] += weight

    for idx in range(num_unique_observed_times - 1):
        at_risk_counts[idx + 1] = at_risk_counts[idx] - dropout_counts[idx]

    event_mask = (event_counts > 0)
    if event_mask.sum() > 0:
        sorted_unique_observed_times = sorted_unique_observed_times[event_mask]
        event_counts = event_counts[event_mask]
        at_risk_counts = at_risk_counts[event_mask]
    else:
        sorted_unique_observed_times = np.zeros((1,))
        event_counts = np.zeros((1,))
        at_risk_counts = np.array([len(y)])

    hazard_func = event_counts / np.clip(at_risk_counts, 1e-12, None)
    surv_func = np.exp(np.cumsum(np.log(1. - hazard_func + 1e-12)))
    cum_haz_func = np.cumsum(hazard_func)

    return {'times': sorted_unique_observed_times,
            'event_counts': event_counts,
            'at_risk_counts': at_risk_counts,
            'surv': surv_func,
            'cum_haz': cum_haz_func}


def _build_tree(X, y, current_depth, max_depth, max_features,
                split_score_function, min_samples_split, min_samples_leaf,
                split_threshold_mode, random_state):
    """
    Builds a survival tree.

    Parameters
    ----------
    X : 2D numpy array, shape = [n_samples, n_features]
        Feature vectors.

    y : 2D numpy array, shape = [n_samples, 2]
        Survival labels (first column is for observed times, second column is
        for event indicators). The i-th row corresponds to the i-th row in `X`.

    current_depth : int
        Current depth of the tree building progress (starts at 0).

    max_depth : int
        Maximum depth of tree building progress. If `current_depth` is equal to
        `max_depth`, then we do not split any further and create a leaf node at
        `tree`.

    max_features : int
        Number of randomly chosen features that we find a split for.

    split_score_function : function
        Function that computes a split score. Look at `logrank` for an example.

    min_samples_split : int
        See documentation for RandomSurvivalForest's `__init__` function.

    min_samples_leaf : int, float
        See documentation for RandomSurvivalForest's `__init__` function.

    split_threshold_mode : string
        See documentation for RandomSurvivalForest's `__init__` function.

    random_state : numpy RandomState instance
        Pseudorandom number generator.
        *Warning*: for this function, `random_state` actually does have to be a
        numpy RandomState instance. This is for computational efficiency
        reasons as to not keep having to sanity check the input.

    Returns
    -------
    tree : dictionary
        A tree built using the given data. If the tree is a leaf node, then its
        key-value pairs are explained in the documentation for _fit_leaf (see
        the return value). Otherwise, the key-value pairs are as follows:
        - 'feature': which feature index to split on at the current node
        - 'threshold': which feature threshold value to split on at the current
            node; the splits are <= threshold (left branch), and > threshold
            (right branch)
        - 'left': the tree for the left branch, stored as a dictionary
        - 'right': the tree for the right branch, stored as a dictionary
    """
    if len(np.unique(y[:, 0])) == 1 or current_depth == max_depth:
        return _fit_leaf(y)

    best_arg = _find_best_feature_split(X, y, max_features,
                                        split_score_function,
                                        min_samples_split, min_samples_leaf,
                                        split_threshold_mode, random_state)

    if best_arg == None:
        return _fit_leaf(y)

    best_feature_idx, split_threshold, left_mask = best_arg

    tree = {'feature': best_feature_idx,
            'threshold': split_threshold}

    tree['left'] = _build_tree(X[left_mask], y[left_mask], current_depth + 1,
                               max_depth, max_features, split_score_function,
                               min_samples_split, min_samples_leaf,
                               split_threshold_mode, random_state)

    right_mask = ~left_mask
    tree['right'] = _build_tree(X[right_mask], y[right_mask],
                                current_depth + 1, max_depth, max_features,
                                split_score_function, min_samples_split,
                                min_samples_leaf, split_threshold_mode,
                                random_state)
    return tree


def _build_tree_ANN(X, y, train_indices, current_depth, max_depth,
                    max_features, split_score_function, min_samples_split,
                    min_samples_leaf, split_threshold_mode, random_state):
    """
    Similar to `_build_tree()` but for the adaptive nearest neighbors variant
    of random survival forests.

    Parameters
    ----------
    X : 2D numpy array, shape = [n_samples, n_features]
        Feature vectors.

    y : 2D numpy array, shape = [n_samples, 2]
        Survival labels (first column is for observed times, second column is
        for event indicators). The i-th row corresponds to the i-th row in `X`.

    train_indices : 1D numpy array, shape = [n_samples]
        Specifies which training subject index each row of `X` corresponds to.

    current_depth : int
        Current depth of the tree building progress (starts at 0).

    max_depth : int
        Maximum depth of tree building progress. If `current_depth` is equal to
        `max_depth`, then we do not split any further and create a leaf node at
        `tree`.

    max_features : int
        Number of randomly chosen features that we find a split for.

    split_score_function : function
        Function that computes a split score. Look at `logrank` for an example.

    min_samples_split : int
        See documentation for RandomSurvivalForest's `__init__` function.

    min_samples_leaf : int, float
        See documentation for RandomSurvivalForest's `__init__` function.

    split_threshold_mode : string
        See documentation for RandomSurvivalForest's `__init__` function.

    random_state : numpy RandomState instance
        Pseudorandom number generator.
        *Warning*: for this function, `random_state` actually does have to be a
        numpy RandomState instance. This is for computational efficiency
        reasons as to not keep having to sanity check the input.

    Returns
    -------
    tree : dictionary
        A tree built using the given data. If the tree is a leaf node, then its
        key-value pairs are explained in the documentation for _fit_leaf (see
        the return value). Otherwise, the key-value pairs are as follows:
        - 'feature': which feature index to split on at the current node
        - 'threshold': which feature threshold value to split on at the current
            node; the splits are <= threshold (left branch), and > threshold
            (right branch)
        - 'left': the tree for the left branch, stored as a dictionary
        - 'right': the tree for the right branch, stored as a dictionary
    """
    if len(np.unique(y[:, 0])) == 1 or current_depth == max_depth:
        return {'train_indices': train_indices}

    best_arg = _find_best_feature_split(X, y, max_features,
                                        split_score_function,
                                        min_samples_split, min_samples_leaf,
                                        split_threshold_mode, random_state)

    if best_arg == None:
        return {'train_indices': train_indices}

    best_feature_idx, split_threshold, left_mask = best_arg

    tree = {'feature': best_feature_idx,
            'threshold': split_threshold}

    tree['left'] = _build_tree_ANN(X[left_mask], y[left_mask],
                                   train_indices[left_mask], current_depth + 1,
                                   max_depth, max_features, split_score_function,
                                   min_samples_split, min_samples_leaf,
                                   split_threshold_mode, random_state)

    right_mask = ~left_mask
    tree['right'] = _build_tree_ANN(X[right_mask], y[right_mask],
                                    train_indices[right_mask], current_depth + 1,
                                    max_depth, max_features, split_score_function,
                                    min_samples_split, min_samples_leaf,
                                    split_threshold_mode, random_state)
    return tree


def _predict_leaf(tree, mode, times, presorted_times, limit_from_left=False):
    """
    Computes either the Kaplan-Meier survival function estimate or the
    Nelson-Aalen cumulative hazard function estimate at user-specified times
    using survival label data in a leaf node.

    Parameters
    ----------
    tree : dictionary
        Leaf node of a decision tree where we pull survival label information
        from.

    mode : string
        Either 'surv' for survival probabilities or 'cum_haz' for cumulative
        hazard function.

    times : 1D numpy array
        Times to compute the survival probability or cumulative hazard function
        at.

    presorted_times : boolean
        Flag for whether `times` is already sorted.

    limit_from_left : boolean, optional (default=False)
        Flag for whether to output the function evaluated at a time just to the
        left, i.e., instead of outputting f(t) where f is either the survival
        probability or cumulative hazard function estimate, output:
            f(t-) := limit as t' approaches t from the left of f(t').

    Returns
    -------
    output : 1D numpy array
        Survival probability or cumulative hazard function evaluated at each of
        the times specified in `times`.
    """
    unique_observed_times = tree['times']
    surv_func = tree[mode]

    if times is None:
        return surv_func

    if presorted_times:
        sort_indices = range(len(times))
    else:
        sort_indices = np.argsort(times)

    num_leaf_times = len(unique_observed_times)
    leaf_time_idx = 0
    last_seen_surv_prob = 1.
    output = np.zeros(len(times))
    if limit_from_left:
        for sort_idx in sort_indices:
            time = times[sort_idx]
            while leaf_time_idx < num_leaf_times:
                if unique_observed_times[leaf_time_idx] <= time:
                    last_seen_surv_prob = surv_func[leaf_time_idx]
                    leaf_time_idx += 1
                else:
                    break
            output[sort_idx] = last_seen_surv_prob
        # return np.interp(times, unique_observed_times[1:], surv_func[:-1])
    else:
        for sort_idx in sort_indices:
            time = times[sort_idx]
            while leaf_time_idx < num_leaf_times:
                if unique_observed_times[leaf_time_idx] < time:
                    last_seen_surv_prob = surv_func[leaf_time_idx]
                    leaf_time_idx += 1
                else:
                    break
            output[sort_idx] = last_seen_surv_prob
        # return np.interp(times, unique_observed_times, surv_func)
    return output


def _predict_row(tree, mode, x, times, presorted_times):
    """
    For a given survival tree and a feature vector, compute the tree's
    survival probability or cumulative hazard function estimate for the feature
    vector evaluated at user-specified times.

    Parameters
    ----------
    tree : dictionary
        Tree node of a decision tree. We traverse down the tree taking branches
        that depend on the given feature vector's values.

    mode : string
        Either 'surv' for survival probabilities or 'cum_haz' for cumulative
        hazard function.

    x : 1D numpy array, shape = [n_features]
        Feature vector.

    times : 1D numpy array
        Times to compute the survival probability or cumulative hazard function
        at.

    presorted_times : boolean
        Flag for whether `times` is already sorted.

    Returns
    -------
    output : 1D numpy array
        Survival probability or cumulative hazard function evaluated at each of
        the times specified in `times`.
    """
    if 'surv' in tree:
        return _predict_leaf(tree, mode, times, presorted_times)

    if x[tree['feature']] <= tree['threshold']:
        return _predict_row(tree['left'], mode, x, times, presorted_times)
    else:
        return _predict_row(tree['right'], mode, x, times, presorted_times)


def _predict_tree(tree, mode, X, times, presorted_times):
    """
    For a given survival tree and many feature vectors, compute the tree's
    survival probability function estimate for each feature vector evaluated at
    user-specified times.

    Parameters
    ----------
    tree : dictionary
        Tree node of a decision tree. We traverse down the tree taking branches
        that depend on the given feature vector's values.

    mode : string
        Either 'surv' for survival probabilities or 'cum_haz' for cumulative
        hazard function.

    X : 2D numpy array, shape = [n_samples, n_features]
        Feature vectors.

    times : 1D numpy array
        Times to compute the survival probability or cumulative hazard function
        at.

    presorted_times : boolean
        Flag for whether `times` is already sorted.

    Returns
    -------
    output : 2D numpy array
        Survival probability or cumulative hazard function evaluated at each of
        the times specified in `times` for each feature vector. The i-th row
        corresponds to the i-th feature vector.
    """
    return np.array([_predict_row(tree, mode, x, times, presorted_times)
                     for x in X])


def _predict_row_vimp(tree, mode, x, times, presorted_times, randomize_idx,
                      random_state):
    """
    The same as _predict_row except for handling variable importance, i.e., a
    single variable's branching decisions get randomized.

    Parameters
    ----------
    tree : dictionary
        Tree node of a decision tree. We traverse down the tree taking branches
        that depend on the given feature vector's values.

    mode : string
        Either 'surv' for survival probabilities or 'cum_haz' for cumulative
        hazard function.

    x : 1D numpy array, shape = [n_features]
        Feature vector.

    times : 1D numpy array
        Times to compute the survival probability or cumulative hazard function
        at.

    presorted_times : boolean
        Flag for whether `times` is already sorted.

    randomize_idx : int
        When this feature index is encountered, randomize the branching
        decision.

    random_state : numpy RandomState instance
        Pseudorandom number generator.
        *Warning*: for this function, `random_state` actually does have to be a
        numpy RandomState instance. This is for computational efficiency
        reasons as to not keep having to sanity check the input.

    Returns
    -------
    output : 1D numpy array
        Survival probability or cumulative hazard function evaluated at each of
        the times specified in `times`.
    """
    if 'surv' in tree:
        return _predict_leaf(tree, mode, times, presorted_times)

    if tree['feature'] == randomize_idx:
        if random_state.randint(2) == 0:
            return _predict_row_vimp(tree['left'], mode, x, times,
                                     presorted_times, randomize_idx,
                                     random_state)
        else:
            return _predict_row_vimp(tree['right'], mode, x, times,
                                     presorted_times, randomize_idx,
                                     random_state)
    elif x[tree['feature']] <= tree['threshold']:
        return _predict_row_vimp(tree['left'], mode, x, times,
                                 presorted_times, randomize_idx, random_state)
    else:
        return _predict_row_vimp(tree['right'], mode, x, times,
                                 presorted_times, randomize_idx, random_state)


def _predict_tree_vimp(tree, mode, X, times, presorted_times, randomize_idx,
                       random_state):
    """
    The same as _predict_tree except for handling variable importance, i.e., a
    single variable's branching decisions get randomized.

    Parameters
    ----------
    tree : dictionary
        Tree node of a decision tree. We traverse down the tree taking branches
        that depend on the given feature vector's values.

    mode : string
        Either 'surv' for survival probabilities or 'cum_haz' for cumulative
        hazard function.

    X : 2D numpy array, shape = [n_samples, n_features]
        Feature vectors.

    times : 1D numpy array
        Times to compute the survival probability or cumulative hazard function
        at.

    presorted_times : boolean
        Flag for whether `times` is already sorted.

    randomize_idx : int
        When this feature index is encountered, randomize the branching
        decision.

    random_state : numpy RandomState instance
        Pseudorandom number generator.
        *Warning*: for this function, `random_state` actually does have to be a
        numpy RandomState instance. This is for computational efficiency
        reasons as to not keep having to sanity check the input.

    Returns
    -------
    output : 2D numpy array
        Survival probability or cumulative hazard function evaluated at each of
        the times specified in `times` for each feature vector. The i-th row
        corresponds to the i-th feature vector.
    """
    return np.array([_predict_row_vimp(tree, mode, x, times,
                                       presorted_times, randomize_idx,
                                       random_state)
                    for x in X])


def _compute_tree_ANN(tree, X):
    """
    Finds the adaptive nearest neighbors for a collection of feature vectors.

    Parameters
    ----------
    tree : dictionary
        Tree node of a decision tree. We traverse down the tree taking branches
        that depend on the given feature vector's values.

    X : 2D numpy array, shape = [n_samples, n_features]
        Feature vectors.

    Returns
    -------
    output : 2D numpy array
        Survival probability or cumulative hazard function evaluated at each of
        the times specified in `times` for each feature vector. The i-th row
        corresponds to the i-th feature vector.
    """
    return [_compute_ANN_row(tree, x) for x in X]


def _compute_ANN_row(tree, x):
    """
    For a given survival tree and a feature vector, traverse down the tree to
    find the feature vector's adaptive nearest neighbors.

    Parameters
    ----------
    tree : dictionary
        Tree node of a decision tree. We traverse down the tree taking branches
        that depend on the given feature vector's values.

    x : 1D numpy array, shape = [n_features]
        Feature vector.

    Returns
    -------
    output : 1D numpy array
        Training subject indices that are the adaptive nearest neighbors of the
        input feature vector.
    """
    if 'train_indices' in tree:
        return tree['train_indices']

    if x[tree['feature']] <= tree['threshold']:
        return _compute_ANN_row(tree['left'], x)
    else:
        return _compute_ANN_row(tree['right'], x)


class CDFRegressionKNNWeightedSurvival():
    def __init__(self, *args, **kwargs):
        """
        Arguments are the same as for `sklearn.neighbors.NearestNeighbors`.
        The simplest usage of this class is to use a single argument, which is
        `n_neighbors` for the number of nearest neighbors (Euclidean distance
        is assumed in this case). If you want to parallelize across different
        search queries, use the `n_jobs` keyword parameter (-1 to use all
        cores). To use other distances and for other details, please refer to
        the documentation for sklearn's `NearestNeighbors` class.

        *Important:* The prediction methods for this class use weighted
        k-nearest neighbors, where "k" is set equal to the `n_neighbors`
        parameter. The weights are specified through a kernel function K. In
        particular, the i-th nearest neighbor X_i for a test point x is given a
        weight of:
            K( (distance between x and X_i) / (distance between x and X_k) ).
        """
        self.NN_index_args = args
        self.NN_index_kwargs = kwargs
        self.NN_index = None

    def fit(self, X, y):
        """
        Constructs a nearest-neighbor index given training data (so that for
        a future data point, we can use the nearest-neighbor index to quickly
        find what the closest training data are to the future point).

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        Returns
        -------
        None
        """
        self.train_y = y
        self.NN_index = NearestNeighbors(*self.NN_index_args,
                                         **self.NN_index_kwargs)
        self.NN_index.fit(X)

    def predict_surv(self, X, times, presorted_times=False,
                     limit_from_left=False, n_neighbors=None,
                     kernel_function=None):
        """
        Computes the weighted k-NN CDF estimation followed by k-NN regression
        survival probability function estimate at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to unweighted k-NN
            survival analysis. If a function is specified, then the weighting
            function used is of the form
            "kernel(distance / distance to k-th nearest neighbor)".

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if kernel_function is None:
            kernel_function = lambda s: 1
        dists, indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                                  return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            max_dist = np.max(dist)
            weights = np.array([kernel_function(d / max_dist) for d in dist])
            zero_weight = (weights == 0)
            if zero_weight.sum() > 0:
                weights_subset = weights[~zero_weight]
                if weights_subset.size > 0:
                    labels_subset = train_y[idx[~zero_weight]]
                else:
                    output.append(np.ones(n_times))
                    continue
            else:
                labels_subset = train_y[idx]
                weights_subset = weights

            # step 1
            weighted_edf_times, weighted_edf = \
                compute_weighted_edf(labels_subset[:, 0], weights_subset)
            one_minus_weighted_edf = 1 - weighted_edf
            if weighted_edf[0] < 1 and weighted_edf_times[0] > 0:
                weighted_edf_times = \
                    np.concatenate(([0.], weighted_edf_times))
                weighted_edf = \
                    np.concatenate(([1.], weighted_edf))

            # step 2
            denoms = np.interp(labels_subset[:, 0], weighted_edf_times,
                               weighted_edf)
            neg_log_S_est = np.zeros(len(times))
            for time_idx, t in enumerate(times):
                neg_log_S_est[time_idx] = \
                    np.inner(labels_subset[:, 1]
                             * (labels_subset[:, 0] <= t) / denoms,
                             weights_subset)

            output.append(np.exp(-neg_log_S_est))
        return np.array(output)

    def predict_cum_haz(self, X, times, presorted_times=False,
                        limit_from_left=False, n_neighbors=None,
                        kernel_function=None):
        """
        Computes the weighted k-NN CDF estimation followed by k-NN regression
        cumulative hazard function estimate at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to unweighted k-NN
            survival analysis. If a function is specified, then the weighting
            function used is of the form
            "kernel(distance / distance to k-th nearest neighbor)".

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if kernel_function is None:
            kernel_function = lambda s: 1
        dists, indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                                  return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            max_dist = np.max(dist)
            weights = np.array([kernel_function(d / max_dist) for d in dist])
            zero_weight = (weights == 0)
            if zero_weight.sum() > 0:
                weights_subset = weights[~zero_weight]
                if weights_subset.size > 0:
                    labels_subset = train_y[idx[~zero_weight]]
                else:
                    output.append(np.ones(n_times))
                    continue
            else:
                labels_subset = train_y[idx]
                weights_subset = weights

            # step 1
            weighted_edf_times, weighted_edf = \
                compute_weighted_edf(labels_subset[:, 0], weights_subset)
            one_minus_weighted_edf = 1 - weighted_edf
            if weighted_edf[0] < 1 and weighted_edf_times[0] > 0:
                weighted_edf_times = \
                    np.concatenate(([0.], weighted_edf_times))
                weighted_edf = \
                    np.concatenate(([1.], weighted_edf))

            # step 2
            denoms = np.interp(labels_subset[:, 0], weighted_edf_times,
                               weighted_edf)
            neg_log_S_est = np.zeros(len(times))
            for time_idx, t in enumerate(times):
                neg_log_S_est[time_idx] = \
                    np.inner(labels_subset[:, 1]
                             * (labels_subset[:, 0] <= t) / denoms,
                             weights_subset)

            output.append(neg_log_S_est)
        return np.array(output)


def compute_weighted_edf(obs, weights=None):
    """
    Computes a weighted empirical distribution function.

    Parameters
    ----------
    obs : 1D numpy array
        Observations to construct the weighted empirical distribution from.

    weights : 1D numpy array, None, optional (default=None)
        Nonnegative weights for the observations. The i-th weight corresponds
        to the i-th value in `obs`. None refers to using uniform weights,
        i.e., each point has weight 1/len(obs).

    Returns
    -------
    sorted_unique_obs : 1D numpy array
        Sorted unique observations in ascending order.

    weighted_edf : 1D numpy array
        The weighted empirical distribution function evaluated at each of the
        values in `sorted_unique_obs`, in the same order.
    """
    if weights is None:
        weights = np.ones(len(obs))
        weights /= weights.shape[0]

    sorted_unique_obs = np.sort(np.unique(obs))
    obs_to_idx = {obs: idx for idx, obs in enumerate(sorted_unique_obs)}
    weighted_edf = np.zeros(len(sorted_unique_obs))
    for x, w in zip(obs, weights):
        weighted_edf[obs_to_idx[x]] += w

    weighted_edf = np.cumsum(weighted_edf)
    return sorted_unique_obs, weighted_edf


def _convert_to_not_use_numpy(tree):
    if 'surv' in tree:
        new_leaf = {}
        for key in tree:
            if type(tree[key]) == np.ndarray:
                new_leaf[key] = tree[key].tolist()
            else:
                new_leaf[key] = tree[key]
        return new_leaf

    new_inner_node = {}
    for key in tree:
        if key == 'left':
            new_inner_node['left'] = _convert_to_not_use_numpy(tree['left'])
        elif key == 'right':
            new_inner_node['right'] = _convert_to_not_use_numpy(tree['right'])
        elif type(tree[key]) == np.ndarray:
            new_inner_node[key] = tree[key].tolist()
        else:
            new_inner_node[key] = tree[key]
    return new_inner_node


def _convert_to_use_numpy(tree):
    if 'surv' in tree:
        new_leaf = {}
        for key in tree:
            if type(tree[key]) == list:
                new_leaf[key] = np.array(tree[key])
            else:
                new_leaf[key] = tree[key]
        return new_leaf

    new_inner_node = {}
    for key in tree:
        if key == 'left':
            new_inner_node['left'] = _convert_to_use_numpy(tree['left'])
        elif key == 'right':
            new_inner_node['right'] = _convert_to_use_numpy(tree['right'])
        elif type(tree[key]) == list:
            new_inner_node[key] = np.array(tree[key])
        else:
            new_inner_node[key] = tree[key]
    return new_inner_node


def _label_leaves(tree, cur_leaf_id=0):
    if 'surv' in tree:
        tree['leaf_id'] = cur_leaf_id
        return cur_leaf_id + 1

    cur_leaf_id = _label_leaves(tree['left'], cur_leaf_id)
    cur_leaf_id = _label_leaves(tree['right'], cur_leaf_id)
    return cur_leaf_id

def _predict_tree_leaf_id(tree, X):
    return np.array([_predict_row_leaf_id(tree, x) for x in X])

def _predict_row_leaf_id(tree, x):
    if 'surv' in tree:
        return tree['leaf_id']

    if x[tree['feature']] <= tree['threshold']:
        return _predict_row_leaf_id(tree['left'], x)
    else:
        return _predict_row_leaf_id(tree['right'], x)

#!/usr/bin/env python
import numpy as np

from npsurvival_models import BasicSurvival

# some kernel functions (note that the assumption is that these are only ever
# evaluated at inputs in the range [0,1])
kernel_box = lambda s: 1
kernel_triangle = lambda s: 1 - np.abs(s)
kernel_epanechnikov = lambda s: 0.75 * (1 - s*s)
kernel_trunc_gauss1 = lambda s: np.exp(-(s*s)/2.)
kernel_trunc_gauss2 = lambda s: np.exp(-(s*s)/8.)
kernel_trunc_gauss3 = lambda s: np.exp(-(s*s)/18.)


def compute_mean_survival_time(times, surv_function):
    """
    TODO: Documentation

    Parameters
    ----------
    times : 1D numpy array
        Sorted list of unique times (in ascending order).

    surv_function : 1D numpy array
        A survival function evaluated at each of time in `times`, in the same
        order.

    Returns
    -------
    output : float
        Mean survival time estimate.
    """
    return np.trapz(surv_function, times)


def compute_median_survival_time(times, surv_function):
    """
    Computes a median survival time estimate by looking for where the survival
    function crosses 1/2.

    Parameters
    ----------
    times : 1D numpy array
        Sorted list of unique times (in ascending order).

    surv_function : 1D numpy array
        A survival function evaluated at each of time in `times`, in the same
        order.

    Returns
    -------
    output : float
        Median survival time estimate.
    """
    t_left = times[0]
    t_right = times[-1]

    if surv_function[-1] > 1/2:
        # survival function never crosses 1/2; just output this last time point
        return t_right

    for t, s in zip(times, surv_function):
        if s >= 0.5:
            t_left = t

    for t, s in zip(reversed(times), reversed(surv_function)):
        if s <= 0.5:
            t_right = t
    return (t_left + t_right) / 2.


def compute_IPEC_scores(y_train, y_test, times, surv_functions, IPEC_horizons,
                        eps=1e-6):
    """
    Computes a median survival time estimate by looking for where the survival
    function crosses 1/2.

    Parameters
    ----------
    y_train : 2D numpy array
        Training set survival labels (first column is for observed times,
        second column is for event indicators).

    y_test : 2D numpy array
        Test set survival labels (first column is for observed times, second
        column is for event indicators).

    times : 1D numpy array
        Sorted list of unique times (in ascending order).

    surv_functions : 1D numpy array
        Estimated survival functions, one for each test subject; these need to
        be evaluated each at the times specified by `times`.

    IPEC_horizons : 1D numpy array
        IPEC score horizons to use.

    eps : float
        If one of the tail probability estimates for censoring time becomes
        too small (i.e., less than `eps`), we change the probability value to
        `eps`. This is for numerical reasons.

    Returns
    -------
    test_IPEC_scores : dictionary
        Each key is one of the IPEC horizons. Each value is its IPEC score.
    """
    y_train_indicator_swap = y_train.copy()
    y_train_indicator_swap[:, 1] = 1 - y_train_indicator_swap[:, 1]

    bs = BasicSurvival()
    bs.fit(y_train_indicator_swap)

    G = bs.predict_surv(times, presorted_times=True, limit_from_left=False)
    G_left = bs.predict_surv(times, presorted_times=True, limit_from_left=True)

    G_small = (G < eps)
    if G_small.sum() > 0:
        G[G_small] = eps

    G_left_small = (G_left < eps)
    if G_left_small.sum() > 0:
        G_left[G_left_small] = eps

    num_test_subjects = y_test.shape[0]
    IPEC_function = np.zeros(len(times))
    for j in range(num_test_subjects):
        IPEC_function += \
            (y_test[j, 1]
             * (y_test[j, 0] <= times) / G_left
             + (y_test[j, 0] > times) / G) \
            * ((y_test[j, 0] > times) - surv_functions[j])**2
    IPEC_function /= num_test_subjects

    test_IPEC_scores = {}
    for IPEC_horizon in IPEC_horizons:
        horizon_mask = (times <= IPEC_horizon)
        if horizon_mask.sum() > 0:
            IPEC_score = \
                piecewise_integrate(times[horizon_mask],
                                    IPEC_function[horizon_mask], 0.,
                                    IPEC_horizon)
        else:
            IPEC_score = 0.
        test_IPEC_scores[IPEC_horizon] = IPEC_score
    return test_IPEC_scores


def piecewise_integrate(x, y, a, b):
    """
    Integrate a piecewise constant function.

    Parameters
    ----------
    x : 1D numpy array
        Sorted distinct values. The function we integrate changes at these x
        values.

    y : 1D numpy array
        The function we are integrating specified for each value in x.
        Specifically, the value from [x[0], x[1]) is y[0], etc.

    a : float
        This is where we start integrating from. The code here actually
        requires a to be equal to x[0].

    b : float
        This is where we stop integrating.

    Returns
    -------
    output : float
        The computed integral.
    """
    assert x[0] == a
    assert x[-1] <= b
    output = 0.
    num_x = len(x)
    if x[-1] == b:
        for idx in range(num_x - 1):
            output += y[idx] * (x[idx+1] - x[idx])
    else:
        for idx in range(num_x):
            if idx < num_x - 1:
                output += y[idx] * (x[idx+1] - x[idx])
            else:
                output += y[idx] * (b - x[idx])
    return output

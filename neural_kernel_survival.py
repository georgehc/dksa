"""
Contains building blocks for neural kernel survival analysis estimators

Author: George H. Chen (georgechen [at symbol] cmu.edu)

To understand what the code is doing, please see the paper:

    George H. Chen. Deep Kernel Survival Analysis and Subject-Specific
    Survival Time Prediction Intervals. MLHC 2020.

"""
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from joblib import Parallel, delayed
from pycox import models
from pycox.models.utils import pad_col, make_subgrid
from pycox.preprocessing import label_transforms
from pycox.models.interpolation import InterpolateLogisticHazard
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


class Scaler(nn.Module):
    """
    PyTorch neural net module that just scales the input by a single scalar
    """
    def __init__(self, num_features, scale=1.):
        super(Scaler, self).__init__()
        self.scaler = torch.nn.Parameter(torch.tensor([scale]))

    def forward(self, input):
        return self.scaler * input


class DiagonalScaler(nn.Module):
    """
    PyTorch neural net module that scales each entry of the input by a
    different scalar
    """
    def __init__(self, num_features, scale=1.):
        super(DiagonalScaler, self).__init__()
        self.scaler = torch.nn.Parameter(scale * torch.ones(num_features))

    def forward(self, input):
        return input * self.scaler.unsqueeze(0)


class ResidualBlock(nn.Module):
    """
    PyTorch residual block that takes an input x and outputs x + alpha*phi(x)
    where phi is a neural net
    """
    def __init__(self, num_features, net, alpha=0.1):
        super(ResidualBlock, self).__init__()
        self.net = net
        self.alpha = alpha

    def forward(self, input):
        return self.alpha * self.net(input) + input


# symmetric squared Euclidean distance calculation from:
# https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/7
def symmetric_squared_pairwise_distances(x):
    r = torch.mm(x, x.t())
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    return diag + diag.t() - 2*r


# efficient pairwise distances from:
# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
def squared_pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
            x[i,:] and y[j,:] if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


class NLLKernelHazardLoss(models.loss._Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor,
                events: Tensor) -> Tensor:
        return nll_kernel_hazard(phi, idx_durations, events, self.reduction)


class NLLKernelHazardNoDiscretizationLoss(torch.nn.Module):
    def forward(self, phi: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return nll_kernel_hazard_no_discretization(phi, durations, events)


def nll_kernel_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                      reduction: str = 'mean') -> Tensor:
    """
    Computes the kernel hazard function loss in the paper:

        George H. Chen. Deep Kernel Survival Analysis and Subject-Specific
        Survival Time Prediction Intervals. MLHC 2020.

    The inputs are the same as for pycox's nll_logistic_hazard function, where
    phi is the output of the base neural net. Time is assumed to have already
    been discretized.
    """
    if events.dtype is not torch.float:
        events = events.float()

    batch_size = phi.size(0)
    num_durations = idx_durations.max().item() + 1
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1, 1)
    y_bce = torch.zeros((batch_size, num_durations), dtype=torch.float,
                        device=phi.device).scatter(1, idx_durations, events)

    # compute kernel matrix
    weights = (-symmetric_squared_pairwise_distances(phi)).exp() \
        - torch.eye(batch_size, device=phi.device)

    # bin weights in the same time index together (only for columns)
    weights_discretized = \
        torch.matmul(weights,
                     torch.zeros((batch_size, num_durations),
                                 dtype=torch.float,
                                 device=phi.device).scatter(1, idx_durations,
                                                            1))

    # kernel hazard function calculation
    num_at_risk = ((weights_discretized.flip(1)).cumsum(1)).flip(1) + 1e-12
    num_deaths = weights_discretized * y_bce
    hazards = torch.clamp(num_deaths / num_at_risk, 1e-12, 1. - 1e-12)

    bce = F.binary_cross_entropy(hazards, y_bce, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return models.loss._reduction(loss, reduction)


def nll_kernel_hazard_no_discretization(phi: Tensor, durations: Tensor,
                                        events: Tensor,
                                        reduction: str = 'mean') -> Tensor:
    """
    Computes the kernel hazard function loss in the paper:

        George H. Chen. Deep Kernel Survival Analysis and Subject-Specific
        Survival Time Prediction Intervals. MLHC 2020.

    This variant uses all times seen in the training data and ignores ties in
    times (breaking ties in an arbitrary fashion).
    """
    sort_indices = durations.sort()[1]
    events = events[sort_indices]
    phi = phi[sort_indices]
    batch_size = phi.size(0)
    idx_durations = torch.arange(batch_size, device=phi.device).view(-1, 1)
    y_bce = torch.diag(events).float().to(phi.device)

    # compute kernel matrix
    weights = (-symmetric_squared_pairwise_distances(phi)).exp() \
        - torch.eye(batch_size, device=phi.device)

    # kernel hazard function calculation
    num_at_risk = ((weights.flip(1)).cumsum(1)).flip(1) + 1e-12
    num_deaths = weights * y_bce
    hazards = torch.clamp(num_deaths / num_at_risk, 1e-12, 1. - 1e-12)

    bce = F.binary_cross_entropy(hazards, y_bce, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return models.loss._reduction(loss, reduction)


class NKSDiscrete(models.base.SurvBase):
    """
    Discretized neural kernel survival analysis estimator
    """
    label_transform = label_transforms.LabTransDiscreteTime

    def __init__(self, net, optimizer=None, device=None, duration_index=None,
                 loss=None, truncate=np.inf, split_loss=False,
                 surv_method='km'):
        self.duration_index = duration_index
        self.truncate = truncate
        self.surv_method = surv_method
        if loss is None:
            loss = NLLKernelHazardLoss()
        super().__init__(net, loss, optimizer, device)

    @property
    def duration_index(self):
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def predict_surv_df(self, input, batch_size=8224, eval_=True,
                        num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True,
                                 num_workers)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0, epsilon=1e-7):
        hazard = self.predict_hazard(input, batch_size, False, eval_, to_cpu,
                                     num_workers)
        if self.surv_method == 'km':
            surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        else:
            # Nelson-Aalen
            surv = (-hazard.cumsum(1)).exp()
        return tt.utils.array_or_tensor(surv, numpy, input)

    def compute_embeddings(self, input, batch_size=8224, numpy=None, eval_=True,
                           to_cpu=False, num_workers=0):
        test_embeddings = self.predict(input, batch_size, False, eval_, False,
                                       to_cpu, num_workers)
        return test_embeddings.cpu().numpy()

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True,
                       to_cpu=False, num_workers=0):
        test_embeddings = self.predict(input, batch_size, False, eval_, False,
                                       to_cpu, num_workers)
        with torch.no_grad():
            test_embeddings = test_embeddings.to(self.device)
            train_embeddings = self.train_embeddings
            idx_durations, events = self.training_data[1]
            num_durations = idx_durations.max().item() + 1
            idx_durations = idx_durations.view(-1, 1)
            events = events.view(-1, 1)
            y_bce = torch.zeros((train_embeddings.size(0), num_durations),
                                dtype=torch.float,
                                device=self.device).scatter(1, idx_durations,
                                                            events)

            # compute kernel matrix
            weights = (-squared_pairwise_distances(test_embeddings,
                                                   train_embeddings)).exp()

            # bin weights in the same time index together (only for columns)
            weights_discretized = \
                torch.matmul(
                        weights,
                        torch.zeros((train_embeddings.shape[0],
                                     num_durations),
                                    dtype=torch.float,
                                    device=self.device).scatter(1,
                                                                idx_durations,
                                                                1))

            # kernel hazard function calculation
            num_at_risk = \
                ((weights_discretized.flip(1)).cumsum(1)).flip(1) + 1e-12
            num_deaths = torch.matmul(weights, y_bce)
            hazards = torch.clamp(num_deaths / num_at_risk, 1e-12, 1. - 1e-12)

        return tt.utils.array_or_tensor(hazards, numpy, input)

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolateLogisticHazard(self, scheme, duration_index, sub)

    def fit(self, input, target=None, batch_size=256, epochs=1, callbacks=None,
            verbose=True, num_workers=0, shuffle=True, metrics=None,
            val_data=None, val_batch_size=8224, **kwargs):
        n_trainable_params = sum(p.numel() for p in self.net.parameters()
                                 if p.requires_grad)
        if n_trainable_params > 0:
            super().fit(input, target, batch_size, epochs, callbacks, verbose,
                        num_workers, shuffle, metrics, val_data, val_batch_size,
                        **kwargs)

        sort_indices = np.argsort(target[0])
        if type(input) != tt.tupletree.TupleTree:
            sorted_input = input[sort_indices]
        else:
            sorted_input = tt.tuplefy(input[0][sort_indices],
                                      input[1][sort_indices])

        sorted_target = (torch.tensor(target[0][sort_indices],
                                      dtype=torch.int64,
                                      device=self.device),
                         torch.tensor(target[1][sort_indices],
                                      dtype=torch.float,
                                      device=self.device))
        self.training_data = (sorted_input, sorted_target)
        self.train_embeddings = self.predict(sorted_input, batch_size, False,
                                             True, False, False, 0)
        self.train_embeddings.to(self.device)

    def save_net(self, path, **kwargs):
        path, extension = os.path.splitext(path)
        assert extension == '.pt'
        super().save_model_weights(path + extension, **kwargs)

        sorted_input, sorted_target = self.training_data
        sorted_observed_times, sorted_events = sorted_target
        if type(sorted_input) != tt.tupletree.TupleTree:
            sorted_features = sorted_input
            sorted_categorical_features = None
        else:
            sorted_features = sorted_input[0]
            sorted_categorical_features = sorted_input[1]
        np.savetxt(path + '_train_features.txt', sorted_features)
        if sorted_categorical_features is not None:
            np.savetxt(path + '_train_categorical_features.txt',
                       sorted_categorical_features)
        np.savetxt(path + '_train_observed_times.txt',
                   sorted_observed_times.cpu().numpy())
        np.savetxt(path + '_train_events.txt',
                   sorted_events.cpu().numpy())
        np.savetxt(path + '_train_embeddings.txt',
                   self.train_embeddings.cpu().numpy())

    def load_net(self, path, **kwargs):
        path, extension = os.path.splitext(path)
        assert extension == '.pt'
        super().load_model_weights(path + extension, **kwargs)

        sorted_features = np.loadtxt(path + '_train_features.txt')
        sorted_features = sorted_features.astype('float32')
        if os.path.isfile(path + '_train_categorical_features.txt'):
            sorted_categorical_features = \
                np.loadtxt(path + '_train_categorical_features.txt')
            if len(sorted_categorical_features.shape) == 1:
                sorted_categorical_features = \
                    sorted_categorical_features.reshape(-1, 1)
            sorted_categorical_features = \
                sorted_categorical_features.astype('int64')
            sorted_input = tt.tuplefy(sorted_features,
                                      sorted_categorical_features)
        else:
            sorted_input = sorted_features

        sorted_observed_times = np.loadtxt(path + '_train_observed_times.txt')
        sorted_events = np.loadtxt(path + '_train_events.txt')
        sorted_observed_times = sorted_observed_times.astype('int64')
        sorted_events = sorted_events.astype('float32')

        sorted_target = (torch.tensor(sorted_observed_times,
                                      dtype=torch.int64,
                                      device=self.device),
                         torch.tensor(sorted_events,
                                      dtype=torch.float,
                                      device=self.device))

        self.training_data = (sorted_input, sorted_target)

        train_embeddings = \
                np.loadtxt(path + '_train_embeddings.txt').astype('float32')
        self.train_embeddings = torch.tensor(train_embeddings,
                                             device=self.device)


class NKS(models.base.SurvBase):
    """
    Neural kernel survival analysis estimator without discretizing time
    (and instead just uses all times seen in the training data)

    During training, for simplicity, the loss basically assumes that there are
    no ties, so if there are ties in times, the ties are broken arbitrarily.
    During testing, the unique training times are used.
    """
    def __init__(self, net, optimizer=None, device=None, loss=None,
                 surv_method='km'):
        self.surv_method = surv_method
        if loss is None:
            loss = NLLKernelHazardNoDiscretizationLoss()
        super().__init__(net, loss, optimizer, device)

    def predict_surv_df(self, input, batch_size=8224, eval_=True,
                        num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True,
                                 num_workers)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0, epsilon=1e-7):
        hazard = self.predict_hazard(input, batch_size, False, eval_, to_cpu,
                                     num_workers)
        if self.surv_method == 'km':
            surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        else:
            # Nelson-Aalen
            surv = (-hazard.cumsum(1)).exp()
        return tt.utils.array_or_tensor(surv, numpy, input)

    def compute_embeddings(self, input, batch_size=8224, numpy=None, eval_=True,
                           to_cpu=False, num_workers=0):
        test_embeddings = self.predict(input, batch_size, False, eval_, False,
                                       to_cpu, num_workers)
        return test_embeddings.cpu().numpy()

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True,
                       to_cpu=False, num_workers=0):
        test_embeddings = self.predict(input, batch_size, False, eval_, False,
                                       to_cpu, num_workers)
        sq_dist_to_weights = lambda x: torch.exp(-x)
        with torch.no_grad():
            test_embeddings = test_embeddings.to(self.device)
            train_embeddings = self.train_embeddings

            durations, events = self.training_data[1]
            duration_to_idx = {duration: idx for idx, duration
                               in enumerate(self.duration_index)}
            idx_durations = torch.tensor([duration_to_idx[duration.item()]
                                          for duration in durations.cpu()],
                                         dtype=torch.int64, device=self.device)
            num_durations = idx_durations.max().item() + 1
            idx_durations = idx_durations.view(-1, 1)
            events = events.view(-1, 1)
            y_bce = torch.zeros((train_embeddings.size(0), num_durations),
                                dtype=torch.float,
                                device=self.device).scatter(1, idx_durations,
                                                            events)

            # compute kernel matrix
            weights = (-squared_pairwise_distances(test_embeddings,
                                                   train_embeddings)).exp()

            # bin weights in the same time index together (only for columns)
            weights_discretized = \
                torch.matmul(
                        weights,
                        torch.zeros((train_embeddings.shape[0],
                                     num_durations),
                                    dtype=torch.float,
                                    device=self.device).scatter(1,
                                                                idx_durations,
                                                                1))

            # kernel hazard function calculation
            num_at_risk = \
                ((weights_discretized.flip(1)).cumsum(1)).flip(1) + 1e-12
            num_deaths = torch.matmul(weights, y_bce)
            hazards = torch.clamp(num_deaths / num_at_risk, 1e-12, 1. - 1e-12)

        return tt.utils.array_or_tensor(hazards, numpy, input)

    def fit(self, input, target=None, batch_size=256, epochs=1, callbacks=None,
            verbose=True, num_workers=0, shuffle=True, metrics=None,
            val_data=None, val_batch_size=8224, **kwargs):
        n_trainable_params = sum(p.numel() for p in self.net.parameters()
                                 if p.requires_grad)
        if n_trainable_params > 0:
            super().fit(input, target, batch_size, epochs, callbacks, verbose,
                        num_workers, shuffle, metrics, val_data, val_batch_size,
                        **kwargs)

        sort_indices = np.argsort(target[0])
        if type(input) != tt.tupletree.TupleTree:
            sorted_input = input[sort_indices]
        else:
            sorted_input = tt.tuplefy(input[0][sort_indices],
                                      input[1][sort_indices])
        sorted_target = (torch.tensor(target[0][sort_indices],
                                      dtype=torch.float,
                                      device=self.device),
                         torch.tensor(target[1][sort_indices],
                                      dtype=torch.float,
                                      device=self.device))
        self.training_data = (sorted_input, sorted_target)
        self.train_embeddings = self.predict(sorted_input, batch_size, False,
                                             True, False, False, 0)
        self.duration_index = np.unique(sorted_target[0].cpu().numpy())

    def save_net(self, path, **kwargs):
        path, extension = os.path.splitext(path)
        assert extension == '.pt'
        super().save_model_weights(path + extension, **kwargs)

        sorted_input, sorted_target = self.training_data
        sorted_observed_times, sorted_events = sorted_target
        if type(sorted_input) != tt.tupletree.TupleTree:
            sorted_features = sorted_input
            sorted_categorical_features = None
        else:
            sorted_features = sorted_input[0]
            sorted_categorical_features = sorted_input[1]
        np.savetxt(path + '_train_features.txt', sorted_features)
        if sorted_categorical_features is not None:
            np.savetxt(path + '_train_categorical_features.txt',
                       sorted_categorical_features)
        np.savetxt(path + '_train_observed_times.txt',
                   sorted_observed_times.cpu().numpy())
        np.savetxt(path + '_train_events.txt',
                   sorted_events.cpu().numpy())
        np.savetxt(path + '_train_embeddings.txt',
                   self.train_embeddings.cpu().numpy())

    def load_net(self, path, **kwargs):
        path, extension = os.path.splitext(path)
        assert extension == '.pt'
        super().load_model_weights(path + extension, **kwargs)

        sorted_features = np.loadtxt(path + '_train_features.txt')
        sorted_features = sorted_features.astype('float32')
        if os.path.isfile(path + '_train_categorical_features.txt'):
            sorted_categorical_features = \
                np.loadtxt(path + '_train_categorical_features.txt')
            if len(sorted_categorical_features.shape) == 1:
                sorted_categorical_features = \
                    sorted_categorical_features.reshape(-1, 1)
            sorted_categorical_features = \
                sorted_categorical_features.astype('int64')
            sorted_input = tt.tuplefy(sorted_features,
                                      sorted_categorical_features)
        else:
            sorted_input = sorted_features

        sorted_observed_times = np.loadtxt(path + '_train_observed_times.txt')
        sorted_events = np.loadtxt(path + '_train_events.txt')
        sorted_observed_times = sorted_observed_times.astype('float32')
        sorted_events = sorted_events.astype('float32')

        sorted_target = (torch.tensor(sorted_observed_times,
                                      dtype=torch.float,
                                      device=self.device),
                         torch.tensor(sorted_events,
                                      dtype=torch.float,
                                      device=self.device))

        self.training_data = (sorted_input, sorted_target)

        train_embeddings = \
                np.loadtxt(path + '_train_embeddings.txt').astype('float32')
        self.train_embeddings = torch.tensor(train_embeddings,
                                             dtype=torch.float,
                                             device=self.device)
        self.duration_index = np.unique(sorted_target[0].cpu().numpy())

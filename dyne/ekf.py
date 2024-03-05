"""Extended Kalman Filter."""
import numpy as np
from .linear import _kalman_update
from .util import Bunch
from ._common import check_measurements


def run_ekf(X0, P0, f, Q, n_epochs, measurements=None):
    """Run Extended Kalman Filter.

    Parameters
    ----------
    X0 : array_like, shape (n_states,)
        Initial state estimate.
    P0 : array_like, shape (n_states, n_states)
        Initial error covariance.
    f : callable
        Process function, must follow `dyne.util.process_callable` interface.
    Q : array_like, shape (n_epochs - 1, n_noises, n_noises) or (n_noises, n_noises)
        Process noise covariance matrix. Either constant or specified for each
        transition.
    n_epochs : int
        Number of epochs for estimation.
    measurements : list or None, optional
        Each element defines a single independent type of measurement as a tuple
        ``(epochs, Z, h, R)``, where

            - epochs : array_like, shape (n,)
                Epoch indices at which the measurement is available.
            - Z : array_like, shape (n, m)
                Measurement vectors.
            - h : callable
                The measurement function which must follow
                `dyne.util.measurement_callable` interface.
            - R : array_like, shape (n, m, m) or (m, m)
                Measurement noise covariance matrix specified for each epoch or a
                single matrix, constant for each epoch.

        None (default) corresponds to an empty list.

    Returns
    -------
    Bunch with the following fields:

        X : ndarray, shape (n_epochs, n_states)
            State estimates.
        P : ndarray, shape (n_epochs, n_states, n_states)
            Error covariance estimates.
    """
    n_states = len(X0)

    Q = np.asarray(Q)
    if Q.ndim == 2:
        Q = np.resize(Q, (n_epochs - 1, *Q.shape))
    n_noises = Q.shape[-1]

    X0 = np.asarray(X0)
    P0 = np.asarray(P0)
    if (X0.shape != (n_states,) or P0.shape != (n_states, n_states) or
            Q.shape != (n_epochs - 1, n_noises, n_noises)):
        raise ValueError("Inconsistent input shapes")
    measurements = check_measurements(measurements)

    X = np.empty((n_epochs, n_states))
    P = np.empty((n_epochs, n_states, n_states))

    X[0] = X0
    P[0] = P0

    for k in range(n_epochs):
        for epochs, Z, h, R in measurements:
            index = np.searchsorted(epochs, k)
            if index < len(epochs) and epochs[index] == k:
                Z_pred, H = h(k, X[k])
                x, P[k], *_ = _kalman_update(np.zeros(n_states), P[k],
                                             Z_pred - Z[index], H, R[index])
                X[k] -= x

        if k + 1 < n_epochs:
            X[k + 1], F, G = f(k, X[k])
            P[k + 1] = F @ P[k] @ F.T + G @ Q[k] @ G.T

    return Bunch(X=X, P=P)

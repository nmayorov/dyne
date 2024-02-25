"""Extended Kalman Filter."""
import numpy as np
from .linear import kf_update
from .util import Bunch


def run_ekf(X0, P0, f, Q, measurements):
    """Run Extended Kalman Filter.

    Parameters
    ----------
    X0 : array_like, shape (n_states,)
        Initial state estimate.
    P0 : array_like, shape (n_states, n_states)
        Initial error covariance.
    f : callable
        Process function, must follow `util.process_callable` interface.
    Q : array_like, shape (n_epochs - 1, n_noises, n_noises) or (n_noises, n_noises)
        Process noise covariance matrix. Either constant or specified for each
        transition.
    measurements : list of n_epoch lists
        Each list contains triples (Z, h, R) with measurement vector, measurement
        function and measurement noise covariance. The measurement function must
        follow `util.measurement_callable` interface.

    Returns
    -------
    Bunch with the following fields:

        X : ndarray, shape (n_epochs, n_states)
            State estimates.
        P : ndarray, shape (n_epochs, n_states, n_states)
            Error covariance estimates.
    """
    n_states = len(X0)
    n_epochs = len(measurements)

    Q = np.asarray(Q)
    if Q.ndim == 2:
        Q = np.resize(Q, (n_epochs - 1, *Q.shape))
    n_noises = Q.shape[-1]

    if (X0.shape != (n_states,) or P0.shape != (n_states, n_states) or
            Q.shape != (n_epochs - 1, n_noises, n_noises)):
        raise ValueError("Inconsistent input shapes")

    X = np.empty((n_epochs, n_states))
    P = np.empty((n_epochs, n_states, n_states))

    X[0] = X0
    P[0] = P0

    for k in range(n_epochs):
        for Z, h, R in measurements[k]:
            Z_pred, H = h(k, X[k])
            x, P[k], *_ = kf_update(np.zeros(n_states), P[k], Z_pred - Z, H, R)
            X[k] -= x

        if k + 1 < n_epochs:
            X[k + 1], F, G = f(k, X[k])
            P[k + 1] = F @ P[k] @ F.T + G @ Q[k] @ G.T

    return Bunch(X=X, P=P)

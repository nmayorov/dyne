"""Unscented Kalman Filter."""
import numpy as np
from scipy import linalg
from .util import Bunch


def run_ukf(X0, P0, f, Q, measurements, alpha=1.0):
    """Run Unscented Kalman Filter.

    Refer to [1]_, sec. 5.6.

    The "unscented transform" is implemented with 2 * n points computed as
    ``m +- alpha * n**0.5 * s``, where

        - ``m`` - current mean estimate
        - ``alpha`` - scaling parameter assumed to be (0, 1]
        - ``n`` - number of states
        - ``s`` - column of the root-covariance matrix

    With ``alpha = 1`` the filter is also known as cubature Kalman filter [1]_,
    sec. 6.6.

    Parameters
    ----------
    X0 : array_like, shape (n_states,)
        Initial state estimate.
    P0 : array_like, shape (n_states, n_states)
        Initial error covariance.
    f : callable
        Transition function callable.
    Q : array_like, shape (n_epochs - 1, n_noises, n_noises) or (n_noises, n_noises)
        Process noise covariance matrix. Either constant or specified for each
        transition.
    measurements : list of n_epoch lists
        Each list contains triples (Z, h, R) with measurement vector, measurement
        function and measurement noise covariance.

    Returns
    -------
    Bunch with the following fields:

        X : ndarray, shape (n_epochs, n_states)
            State estimates.
        P : ndarray, shape (n_epochs, n_states, n_states)
            Error covariance estimates.

    References
    ----------
    .. [1] S. Särkkä "Baysian Filtering and Smoothing"
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
            Sigma = alpha * np.sqrt(n_states) * linalg.cholesky(P[k])
            Z_probe = []
            X_probe = []
            # Default scipy cholesky gives U such that P = U^T U,
            # we need to add columns of U^T or rows of U
            for sigma in Sigma:
                for sign in [-1, 1]:
                    X_probe.append(X[k] + sign * sigma)
                    Z_probe.append(h(k, X_probe[-1], with_jacobian=False))
            Z_probe = np.asarray(Z_probe)
            Z_pred = np.mean(Z_probe, axis=0)
            Z_probe -= Z_pred

            X_probe = np.asarray(X_probe)
            X_probe -= X[k]

            P_ee = Z_probe.T @ Z_probe / (alpha**2 * len(Z_probe)) + R
            P_zx = Z_probe.T @ X_probe / (alpha**2 * len(Z_probe))
            K = linalg.cho_solve(linalg.cho_factor(P_ee), P_zx).T

            X[k] += K @ (Z - Z_pred)
            P[k] -= K @ P_ee @ K.T

        if k + 1 < n_epochs:
            Sigma = alpha * np.sqrt(n_states + n_noises) * linalg.block_diag(
                linalg.cholesky(P[k]), linalg.cholesky(Q[k]))
            X_probe = []
            for sigma in Sigma:
                for sign in [-1, 1]:
                    X_probe.append(f(k, X[k] + sign * sigma[:n_states],
                                     sign * sigma[n_states:], with_jacobian=False))
            X_probe = np.asarray(X_probe)
            X[k + 1] = np.mean(X_probe, axis=0)
            X_probe -= X[k + 1]
            P[k + 1] = X_probe.T @ X_probe / (alpha**2 * len(X_probe))

    return Bunch(X=X, P=P)

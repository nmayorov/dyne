"""Unscented Kalman Filter."""
import numpy as np
from scipy import linalg
from .util import Bunch
from ._common import check_input_arrays, check_measurements


def generate_sigma_points(P_blocks, alpha):
    """Generate sigma-points which represent a given covariance matrix.

    The covariance matrix is assumed to be block-diagonal, with blocks given in
    `P_blocks`.

    The sigma-points are computed from the eigen decomposition. The negative
    eigenvalues are replaced by zeros, which makes the algorithm robust for
    marginally indefinite (due to rounding errors) matrices.

    Parameters
    ----------
    P_blocks : list of ndarray
        Diagonal blocks of the covariance matrix.
    alpha : float
        Scaling factor for the eigenvectors.

    Returns
    ------
    sigma_points : ndarray, shape (2 * n, n)
        Generated sigma-points. Each of 2 * n rows represents a single point with n
        components, where n is the total number of rows or columns of the covariance
        matrix.
    w_mean : float
        Weights (equal for each point) to compute the sample mean.
    w_cov : float
        Weights (equal for each point) to compute the sample covariance.
    """
    root_blocks = []
    for P in P_blocks:
        if len(P) > 0:
            s, V = linalg.eigh(P)
            s[s < 0] = 0
            root_blocks.append(V * s ** 0.5)
    n_states = sum(len(P) for P in P_blocks)
    P_root = linalg.block_diag(*root_blocks)
    return (alpha * n_states ** 0.5 * np.hstack((P_root, -P_root)).T,
            1 / (2 * n_states), 1 / (2 * alpha ** 2 * n_states))


def run_ukf(X0, P0, f, Q, measurements, n_epochs, alpha=1.0):
    """Run Unscented Kalman Filter.

    The "unscented transform" is implemented with ``2 * n points`` computed as
    ``m +- alpha * n**0.5 * s``, where

        - ``m`` - current estimate
        - ``alpha`` - scaling parameter assumed to be (0, 1]
        - ``n`` - number of states
        - ``s`` - column of the root-covariance matrix

    With ``alpha = 1`` the filter is also known as cubature Kalman Filter.

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
    measurements : list or None
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
    n_epochs : int
        Number of epochs for estimation.
    alpha : float, optional
        Scaling factor for the unscented transform. Default is 1.

    Returns
    -------
    Bunch with the following fields:

        X : ndarray, shape (n_epochs, n_states)
            State estimates.
        P : ndarray, shape (n_epochs, n_states, n_states)
            Error covariance estimates.
    """
    X0, P0, Q, n_states, n_noises = check_input_arrays(X0, P0, Q, n_epochs)
    measurements = check_measurements(measurements)

    X = np.empty((n_epochs, n_states))
    P = np.empty((n_epochs, n_states, n_states))
    X[0] = X0
    P[0] = P0

    for k in range(n_epochs):
        for epochs, Z, h, R in measurements:
            index = np.searchsorted(epochs, k)
            if index < len(epochs) and epochs[index] == k:
                sigma_points, w_mean, w_cov = generate_sigma_points([P[k]], alpha)
                Z_probe = []
                X_probe = []
                for point in sigma_points:
                    X_probe.append(X[k] + point)
                    Z_probe.append(h(k, X_probe[-1], with_jacobian=False))
                Z_probe = np.asarray(Z_probe)
                Z_pred = w_mean * np.sum(Z_probe, axis=0)
                Z_probe -= Z_pred

                X_probe = np.asarray(X_probe)
                X_probe -= X[k]

                P_ee = w_cov * Z_probe.T @ Z_probe + R[index]
                P_zx = Z_probe.T @ X_probe / (alpha**2 * len(Z_probe))
                K = linalg.cho_solve(linalg.cho_factor(P_ee), P_zx).T

                X[k] += K @ (Z[index] - Z_pred)
                P[k] -= K @ P_ee @ K.T

        if k + 1 < n_epochs:
            sigma_points, w_mean, w_cov = generate_sigma_points([P[k], Q[k]], alpha)
            X_probe = []
            for point in sigma_points:
                X_probe.append(f(k, X[k] + point[:n_states], point[n_states:],
                                 with_jacobian=False))
            X_probe = np.asarray(X_probe)
            X[k + 1] = w_mean * np.sum(X_probe, axis=0)
            X_probe -= X[k + 1]
            P[k + 1] = w_cov * X_probe.T @ X_probe

    return Bunch(X=X, P=P)

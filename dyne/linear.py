"""Linear Kalman filter and smoother."""
import numpy as np
from scipy import linalg
from .util import Bunch


def _run_smoother(x0, P0, F, G, Q, measurements, u, w):
    n_epochs = len(measurements)
    n_states = len(x0)
    n_noises = Q.shape[-1]

    xf = np.empty((n_epochs, n_states))
    Pf = np.empty((n_epochs, n_states, n_states))
    xf[0] = x0
    Pf[0] = P0
    smoother_data = []

    for epoch in range(n_epochs):
        smoother_data.append([])
        for z, H, R in measurements[epoch]:
            xf[epoch], Pf[epoch], U, r, M = _kalman_update(
                xf[epoch], Pf[epoch], z, H, R)
            smoother_data[-1].append((U, r, M))

        if epoch + 1 < n_epochs:
            Fk = F[epoch]
            Gk = G[epoch]
            xf[epoch + 1] = Fk @ xf[epoch] + Gk @ w[epoch] + u[epoch]
            Pf[epoch + 1] = Fk @ Pf[epoch] @ Fk.T + Gk @ Q[epoch] @ Gk.T

    lamb = np.zeros(n_states)
    Lamb = np.zeros((n_states, n_states))

    xs = np.empty((n_epochs, n_states))
    Ps = np.empty((n_epochs, n_states, n_states))
    ws = np.empty((n_epochs - 1, n_noises))
    Qs = np.empty((n_epochs - 1, n_noises, n_noises))

    for epoch in reversed(range(n_epochs)):
        P = Pf[epoch]
        xs[epoch] = xf[epoch] + P @ lamb
        Ps[epoch] = P - P @ Lamb @ P

        for U, r, M in reversed(smoother_data[epoch]):
            lamb = U.T @ lamb + r
            Lamb = U.T @ Lamb @ U + M

        if epoch > 0:
            Gk = G[epoch - 1]
            Qk = Q[epoch - 1]
            ws[epoch - 1] = w[epoch - 1] + Qk @ Gk.T @ lamb
            Qs[epoch - 1] = Qk - Qk @ Gk.T @ Lamb @ Gk @ Qk

            Fk = F[epoch - 1]
            lamb = Fk.T @ lamb
            Lamb = Fk.T @ Lamb @ Fk

    return Bunch(x=xs, P=Ps, w=ws, Q=Qs, xf=xf, Pf=Pf)


def run_kalman_smoother(x0, P0, F, G, Q, measurements, n_epochs):
    """Run linear Kalman smoother.

    The algorithm with explicit co-state recursion is implemented
    ("Bryson-Frazier smoother"). It is more universal and doesn't rely on covariance
    matrices from Kalman filter being positive definite. See [1]_ for the discussion
    of different approaches to the linear smoothing.

    The Kalman filter result comes as a byproduct.

    Parameters
    ----------
    x0 : array_like, shape (n_states,)
        Initial state mean.
    P0 : array_like, shape (n_states, n_states)
        Initial state covariance.
    F : array_like, shape (n_epochs - 1, n_states, n_states) or (n_states, n_states)
        Transition matrices.
    G : array_like, shape (n_epochs - 1, n_states, n_noises) or (n_states, n_noises)
        Process noise input matrices.
    Q : array_like, shape (n_epochs - 1, n_noises, n_noises) or (n_noises, n_noises)
        Process noise covariance matrices.
    measurements : list
        Each element defines a single independent type of measurement as a tuple
        ``(epochs, z, H, R)``, where

            - epochs : array_like, shape (n,)
                Epoch indices at which the measurement is available.
            - z : array_like, shape (n, m)
                Measurement vectors.
            - H : array_like, shape (n, n_states, m) or (n_states, m)
                Measurement matrices specified for each epoch or a single matrix,
                constant for each epoch.
            - R : array_like, shape (n, m, m) or (m, m)
                Measurement noise covariance matrix specified for each epoch or a
                single matrix, constant for each epoch.

        None (default) corresponds to an empty list.
    n_epochs : int
        Number of epochs for estimation.

    Returns
    -------
    Bunch object with the following fields:

        - x : ndarray, shape (n_epochs, n_states)
            Smoother state estimates.
        - P : ndarray, shape (n_epochs, n_states, n_states)
            Smoother error covariance matrices.
        - w : ndarray, shape (n_epoch, n_noises)
            Smoother noise vector estimates.
        - Q : ndarray, shape (n_epoch, n_noises, n_noises)
            Smoother noise vector error covariance matrices.
        - xf : ndarray, shape (n_epochs, n_states, n_states)
            Filter state estimates.
        - Pf : ndarray, shape (n_epoch, n_states, n_states)
            Filter error covariances.

    References
    ----------
    .. [1] S. R. McReynolds "Fixed interval smoothing - Revisited",
       Journal of Guidance, Control, and Dynamics 1990, Vol. 23, No. 5
    """
    x0 = np.asarray(x0)
    P0 = np.asarray(P0)

    if measurements is None:
        measurements = []

    F = np.asarray(F)
    if F.ndim == 2:
        F = np.resize(F, (n_epochs - 1, *F.shape))
    G = np.asarray(G)
    if G.ndim == 2:
        G = np.resize(G, (n_epochs - 1, *G.shape))
    Q = np.asarray(Q)
    if Q.ndim == 2:
        Q = np.resize(Q, (n_epochs - 1, *Q.shape))

    n_states = len(x0)
    n_noises = G.shape[-1]

    meas = []
    for epochs, z, H, R in measurements:
        z = np.asarray(z)
        H = np.asarray(H)
        if H.ndim == 2:
            H = np.resize(H, (len(epochs), *H.shape))
        R = np.asarray(R)
        if R.ndim == 2:
            R = np.resize(R, (len(epochs), *R.shape))

        n = len(epochs)
        m = z.shape[-1]
        if z.shape != (n, m) or H.shape != (n, m, n_states) or R.shape != (n, m, m):
            raise ValueError("Inconsistent shapes in measurements")

        meas.append([epochs, z, H, R])

    if (x0.shape != (n_states,) or
        P0.shape != (n_states, n_states) or
        F.shape != (n_epochs - 1, n_states, n_states) or
        G.shape != (n_epochs - 1, n_states, n_noises) or
        Q.shape != (n_epochs - 1, n_noises, n_noises)
    ):
        raise ValueError("Inconsistent sizes of inputs")

    meas_each_epoch = []
    for epoch in range(n_epochs):
        meas_epoch = []
        for epochs, z, H, R in meas:
            index = np.searchsorted(epochs, epoch)
            if index < len(epochs) and epochs[index] == epoch:
                meas_epoch.append((z[index], H[index], R[index]))
        meas_each_epoch.append(meas_epoch)

    return _run_smoother(x0, P0, F, G, Q, meas_each_epoch,
                         np.zeros((n_epochs - 1, n_states)),
                         np.zeros((n_epochs - 1, n_noises)))


def _kalman_update(x, P, z, H, R):
    S = H @ P @ H.T + R
    J = linalg.cho_solve(linalg.cho_factor(S), H).T
    K = P @ J
    e = z - H @ x
    U = np.eye(len(x)) - K @ H
    return x + K @ e, U @ P @ U.T + K @ R @ K.T, U, J @ e, J @ H

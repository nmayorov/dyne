import numpy as np
from scipy import linalg
from ._common import Bunch


def run_kalman_smoother(x0, P0, F, G, Q, measurements, u=None, w=None):
    """Run linear Kalman smoother.

    The algorithm with explicit co-state recursion is implemented
    ("Bryson-Frazier smoother"). It is more universal and doesn't rely on covariance
    matrices from Kalman filter being positive definite. See [1]_ for the discussion
    of different approaches to the linear smoothing.

    Kalman filter result comes as a byproduct.

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
    measurements : list of n_epochs lists
        Each list contains tuples (z, H, R) with measurement vector, measurement matrix
        and noise covariance matrix.
    u : array_like, shape (n_epochs - 1, n_states) or (n_states,) or None, optional
        Input control vectors. If None (default) no control vectors are applied.
    w : array_like, shape (n_epochs - 1, n_noises) or (n_noises,) or None, optional
        Noise mean offset vectors, typically are not used in basic Kalman filtering, but
        may be required by other algorithms, which use the linear Kalman filter.
        If None (default), assumed to be zero.

    Returns
    -------
    Bunch object with the following fields:

        - xf : ndarray, shape (n_epochs, n_states)
            Filter state estimates.
        - Pf : ndarray, shape (n_epochs, n_states, n_states)
            Filter covariance matrices.
         - xo : ndarray, shape (n_epochs, n_states, n_states)
            Smoother (optimized) state estimates.
        - Po : ndarray, shape (n_epoch, n_states, n_states)
            Smoother covariance matrices.
        - wo : ndarray, shape (n_epoch, n_noises)
            Smoother noise vector estimates.
        - Qo : ndarray, shape (n_epoch, n_noises, n_noises)
            Smoother noise vector estimates.

    References
    ----------
    .. [1] S. R. McReynolds "Fixed interval smoothing - Revisited",
       Journal of Guidance, Control, and Dynamics 1990, Vol. 23, No. 5
    """
    x0 = np.asarray(x0)
    P0 = np.asarray(P0)

    n_epochs = len(measurements)
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

    u = np.zeros((n_epochs - 1, n_states)) if u is None else np.asarray(u)
    w = np.zeros((n_epochs - 1, n_noises)) if w is None else np.asarray(w)

    if (x0.shape != (n_states,) or
        P0.shape != (n_states, n_states) or
        F.shape != (n_epochs - 1, n_states, n_states) or
        G.shape != (n_epochs - 1, n_states, n_noises) or
        Q.shape != (n_epochs - 1, n_noises, n_noises) or
        u.shape != (n_epochs - 1, n_states) or
        w.shape != (n_epochs - 1, n_noises)
    ):
        raise ValueError("Inconsistent sizes of inputs")

    xf = np.empty((n_epochs, n_states))
    Pf = np.empty((n_epochs, n_states, n_states))
    xf[0] = x0
    Pf[0] = P0
    smoother_data = []

    for epoch in range(n_epochs):
        smoother_data.append([])
        for z, H, R in measurements[epoch]:
            xf[epoch], Pf[epoch], U, r, M = kf_update(xf[epoch], Pf[epoch], z, H, R)
            smoother_data[-1].append((U, r, M))
        if epoch + 1 < n_epochs:
            Fk = F[epoch]
            Gk = G[epoch]
            xf[epoch + 1] = Fk @ xf[epoch] + Gk @ w[epoch] + u[epoch]
            Pf[epoch + 1] = Fk @ Pf[epoch] @ Fk.T + Gk @ Q[epoch] @ Gk.T

    lamb = np.zeros(n_states)
    Lamb = np.zeros((n_states, n_states))

    xo = np.empty((n_epochs, n_states))
    Po = np.empty((n_epochs, n_states, n_states))
    wo = np.empty((n_epochs - 1, n_noises))
    Qo = np.empty((n_epochs - 1, n_noises, n_noises))

    for epoch in reversed(range(n_epochs)):
        P = Pf[epoch]
        xo[epoch] = xf[epoch] + P @ lamb
        Po[epoch] = P - P @ Lamb @ P

        if epoch > 0:
            Gk = G[epoch - 1]
            Qk = Q[epoch - 1]
            wo[epoch - 1] = w[epoch - 1] + Qk @ Gk.T @ lamb
            Qo[epoch - 1] = Qk - Qk @ Gk.T @ Lamb @ Gk @ Qk

        for U, r, M in reversed(smoother_data[epoch]):
            lamb = U.T @ lamb + r
            Lamb = U.T @ Lamb @ U + M

        if epoch > 0:
            Fk = F[epoch - 1]
            lamb = Fk.T @ lamb
            Lamb = Fk.T @ Lamb @ Fk

    return Bunch(xf=xf, Pf=Pf, xo=xo, Po=Po, wo=wo, Qo=Qo)


def kf_update(x, P, z, H, R):
    S = H @ P @ H.T + R
    J = linalg.cho_solve(linalg.cho_factor(S), H).T
    K = P @ J
    e = z - H @ x
    U = np.eye(len(x)) - K @ H
    return x + K @ e, U @ P @ U.T + K @ R @ K.T, U, J @ e, J @ H

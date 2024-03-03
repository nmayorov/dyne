"""Estimation algorithms based on optimization."""
import numpy as np
from scipy import linalg
from .linear import _run_smoother
from .ekf import run_ekf
from .util import Bunch
from ._common import check_measurements


def _eval_quadratic_step(x0, P0, Q, measurements_lin, wp, x, w):
    def _eval_quadratic_item(x, z, R, H=None):
        if R.size == 0:
            return 0.0, 0.0
        Hx = x if H is None else H @ x
        RiHx = linalg.cho_solve(linalg.cho_factor(R), Hx)
        return 0.5 * np.dot(Hx, RiHx) - np.dot(z, RiHx), -np.dot(z, RiHx)

    cost_change, grad_dot_step = _eval_quadratic_item(x[0], x0, P0)
    for xk, measurement_k in zip(x, measurements_lin):
        for z, H, R in measurement_k:
            c, d = _eval_quadratic_item(xk, z, R, H)
            cost_change += c
            grad_dot_step += d
    for wk, wpk, Qk in zip(w, wp, Q):
        c, d = _eval_quadratic_item(wk, wpk, Qk)
        cost_change += c
        grad_dot_step += d
    return cost_change, grad_dot_step


def _eval_cost(x0, P0, w, Q, measurements):
    def _eval_quadratic(x, P):
        return (0 if P.size == 0
                else 0.5 * np.dot(x, linalg.cho_solve(linalg.cho_factor(P), x)))

    result = _eval_quadratic(x0, P0)
    for wk, Qk in zip(w, Q):
        result += _eval_quadratic(wk, Qk)
    for measurements_k in measurements:
        for (z, _, R) in measurements_k:
            result += _eval_quadratic(z, R)
    return result


def _optimize(X0, P0, X, W, f, Q, measurements, start_epoch, stop_epoch,
              ftol, ctol, max_iter):
    RHO = 0.5
    MIN_ALPHA = 0.01
    TAU = 0.9
    ETA = 0.1
    GTOL = 1e-16

    n_states = X.shape[-1]
    n_noises = W.shape[-1]
    n_epochs = stop_epoch - start_epoch

    def _build_linear_problem(X, W):
        F = np.empty((n_epochs - 1, n_states, n_states))
        G = np.empty((n_epochs - 1, n_states, n_noises))
        u = np.empty((n_epochs - 1, n_states))
        measurements_lin = []

        for i in range(n_epochs):
            k = i + start_epoch
            meas_epoch = []
            for epochs, Z, h, R in measurements:
                index = np.searchsorted(epochs, k)
                if index < len(epochs) and epochs[index] == k:
                    Z_pred, H = h(k, X[i])
                    meas_epoch.append((Z[index] - Z_pred, H, R[index]))
            measurements_lin.append(meas_epoch)

            if i + 1 < n_epochs:
                X_pred, F[i], G[i] = f(k, X[i], W[i])
                u[i] = X_pred - X[i + 1]

        x0 = X0 - X[0]
        w = -W
        return (x0, F, G, measurements_lin, u, w,
                _eval_cost(x0, P0, w, Q, measurements_lin), np.sum(np.abs(u)))

    mu = 1.0
    for iteration in range(max_iter):
        x0, F, G, measurements_lin, u, w, cost, cv_l1 = _build_linear_problem(X, W)
        linear_result = _run_smoother(x0, P0, F, G, Q, measurements_lin, u, w)
        qp_cost_change, qp_grad_dot_step = _eval_quadratic_step(
            x0, P0, Q, measurements_lin, w, linear_result.x, linear_result.w)
        if qp_cost_change > 0:
            mu = max(mu, qp_cost_change / (1 - RHO) / cv_l1)
        D = qp_grad_dot_step - mu * cv_l1
        if abs(D) < GTOL:
            break
        merit = cost + mu * cv_l1

        alpha = 1.0
        while alpha > MIN_ALPHA:
            X_new = X + alpha * linear_result.x
            W_new = W + alpha * linear_result.w
            *_, u, _, cost_new, cv_l1 = _build_linear_problem(X_new, W_new)
            merit_new = cost_new + mu * cv_l1
            if merit_new < merit + ETA * alpha * D:
                break
            else:
                alpha *= TAU

        X = X_new
        W = W_new

        cost_check = abs(cost - cost_new) < ftol * cost
        cv_check = np.all(np.abs(u) < ctol * np.maximum(np.abs(X[1:]), 1.0))
        if cost_check and cv_check:
            break

    return X, W, linear_result.P, linear_result.Q


def run_optimization(X0, P0, f, Q, n_epochs, measurements=None,
                     ftol=1e-8, ctol=1e-8, max_iter=10):
    """Run batch optimization algorithm.

    The algorithm is conceptually known for some time and described relatively well in
    [1]_ for a sliding window optimization filter (see `run_mhf`).
    An improved approach to nonlinear iterations (implemented here) is described in
    [2]_.

    The iterations are terminated if both conditions controlled by `ftol` and `ctol`
    are satisfied or the number of iterations exceeds `max_iter`.

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
    ftol : float, optional
        Required tolerance for termination by the change of the cost function.
        The iterations can be terminated if the relative cost change on the last
        iteration is less than `ftol`. Default is 1e-8.
    ctol : float, optional
        Required tolerance of constraints satisfaction. The iterations can be terminated
        if the relative residual between predicted and actual `X` is less that
        `ctol`. Default is 1e-8.
    max_iter : int, optional
        Maximum allowed number of iterations. Default is 10.

    Returns
    -------
    Bunch objects with the following fields:

        - X : ndarray, shape (n_epochs, n_states)
            Optimized state estimates.
        - P : ndarray, shape (n_epochs, n_states, n_states)
            Error covariance matrices.
        - W : ndarray, shape (n_epochs, n_noises)
            Optimized noise vector estimates.
        - Q : ndarray, shape (n_epochs, n_noises, n_noises)
            Noise vector error covariance matrices.
        - Xf : ndarray, shape (n_epochs, n_states)
            State estimates computed by EKF.
        - Pf : ndarray, shape (n_epochs, n_states, n_states)
            Error covariance matrices for EKF estimates.

    References
    ----------
    .. [1] M. L. Psiaki, "Backward-Smoothing Extended Kalman Filter",
       Journal of Guidance, Control, and Dynamics 2005, Vol. 28, No. 5
    .. [2] N. Mayorov, "Nonlinear batch estimation",
       https://nmayorov.github.io/posts/nonlinear_batch_estimation/
    """
    n_states = len(X0)

    Q = np.asarray(Q)
    if Q.ndim == 2:
        Q = np.resize(Q, (n_epochs - 1, *Q.shape))
    n_noises = Q.shape[-1]

    if (X0.shape != (n_states,) or P0.shape != (n_states, n_states) or
            Q.shape != (n_epochs - 1, n_noises, n_noises)):
        raise ValueError("Inconsistent input shapes")
    measurements = check_measurements(measurements)
    ekf_result = run_ekf(X0, P0, f, Q, n_epochs, measurements)
    X, W, P, Q = _optimize(X0, P0, ekf_result.X, np.zeros((n_epochs - 1, n_noises)),
                           f, Q, measurements, 0, n_epochs, ftol, ctol, max_iter)
    return Bunch(X=X, P=P, W=W, Q=Q, Xf=ekf_result.X, Pf=ekf_result.P)


def run_mhf(X0, P0, f, Q, n_epochs, measurements=None, window=5,
            ftol=1e-8, ctol=1e-8, max_iter=10):
    """Run moving horizon filter.

    This filter performs nonlinear optimization within a window of a given size.
    The initial state prior and covariance for the optimization are computed by
    propagating the previous estimate using Extended Kalman Filter step.

    Note that it has higher computation cost (for moderately high `window`) and worse
    accuracy than `run_optimize`. But it is a *causal* filter (i.e. at epoch k only
    information available up to epoch k is used) and in principle can be executed
    in real time.

    See `run_optimize` for the references.

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
    window : int, optional
        Optimization window size. Value of 1 corresponds to iterated EKF algorithm.
        Default is 5.
    ftol : float, optional
        Required tolerance for termination by the change of the cost function.
        The iterations can be terminated if the relative cost change on the last
        iteration is less than `ftol`. Default is 1e-8.
    ctol : float, optional
        Required tolerance of constraints satisfaction. The iterations can be terminated
        if the relative residual between predicted and actual `X` is less that
        `ctol`. Default is 1e-8.
    max_iter : int, optional
        Maximum allowed number of iterations. Default is 10.

    Returns
    -------
    Bunch objects with the following fields:

        - X : ndarray, shape (n_epochs, n_states)
            State estimates.
        - P : ndarray, shape (n_epochs, n_states, n_states)
            Error covariance matrices.
    """
    n_states = len(X0)

    Q = np.asarray(Q)
    if Q.ndim == 2:
        Q = np.resize(Q, (n_epochs - 1, *Q.shape))
    n_noises = Q.shape[-1]

    if (X0.shape != (n_states,) or P0.shape != (n_states, n_states) or
            Q.shape != (n_epochs - 1, n_noises, n_noises)):
        raise ValueError("Inconsistent input shapes")
    measurements = check_measurements(measurements)

    Xo = np.empty((0, n_states))
    Wo = np.empty((0, n_noises))

    X = np.empty((n_epochs, n_states))
    P = np.empty((n_epochs, n_states, n_states))

    for k in range(n_epochs):
        if k == 0:
            Xo = np.vstack([Xo, X0])
        elif k < window:
            Xo = np.vstack([Xo, f(k - 1, X[k - 1], with_jacobian=False)])
            Wo = np.vstack([Wo, np.zeros(n_noises)])
        else:
            Xo = np.roll(Xo, -1, axis=0)
            Wo = np.roll(Wo, -1, axis=0)
            Xo[-1] = f(k - 1, X[k - 1], with_jacobian=False)
            if window > 1:
                Wo[-1] = np.zeros(n_noises)
            X0, F, G = f(k - window, X[k - window])
            P0 = F @ P[k - window] @ F.T + G @ Q[k - window] @ G.T

        start_epoch = max(0, k + 1 - window)
        stop_epoch = k + 1

        Xo, Wo, Po, _ = _optimize(X0, P0, Xo, Wo, f, Q[start_epoch : stop_epoch - 1],
                                  measurements, start_epoch, stop_epoch, ftol, ctol,
                                  max_iter)
        X[k] = Xo[-1]
        P[k] = Po[-1]

    return Bunch(X=X, P=P)

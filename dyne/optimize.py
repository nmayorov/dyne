from copy import deepcopy
import numpy as np
from scipy import linalg
from .linear import run_kalman_smoother
from .ekf import run_ekf
from .util import Bunch


def _build_linear_problem(X0, X, W, f, measurements):
    n_epochs = len(measurements)
    n_states = len(X0)
    n_noises = W.shape[-1]
    F = np.empty((n_epochs - 1, n_states, n_states))
    G = np.empty((n_epochs - 1, n_states, n_noises))
    u = np.empty((n_epochs - 1, n_states))
    measurements_lin = []

    for k in range(n_epochs):
        measurements_lin.append([])
        for Z, h, R in measurements[k]:
            Z_pred, H = h(k, X[k])
            measurements_lin[-1].append((Z - Z_pred, H, R))

        if k + 1 < n_epochs:
            X_pred, F[k], G[k] = f(k, X[k], W[k])
            u[k] = X_pred - X[k + 1]

    return X0 - X[0], F, G, measurements_lin, u, -W


def _eval_quadratic_step(x0, P0, Q, measurements_lin, wp, x, w):
    def _eval_quadratic_item(x, z, R, H=None):
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
        return 0.5 * np.dot(x, linalg.cho_solve(linalg.cho_factor(P), x))

    result = _eval_quadratic(x0, P0)
    for wk, Qk in zip(w, Q):
        result += _eval_quadratic(wk, Qk)
    for measurements_k in measurements:
        for (z, _, R) in measurements_k:
            result += _eval_quadratic(z, R)
    return result


def run_optimization(X0, P0, f, Q, measurements, ftol=1e-8, ctol=1e-8, max_iter=10):
    """Run batch optimization algorithm.

    The iterations are terminated if both conditions controlled by `ftol` and `ctol`
    are satisfied. Or the number of iterations exceeds `max_iter`.

    Parameters
    ----------
    X0 : array_like, shape (n_states,)
        Initial state estimate.
    P0 : array_like, shape (n_states, n_states)
        Initial error covariance.
    f : callable
        Transition function with a signature ``f(k, X, W=None) -> X_next, F, G``, where:

            - k - epoch index to transition from, i.e. from k to k + 1
            - X - state vector
            - W - optional noise vector. If None, then it must be treated as zeros with
              an appropriate size
            - X_next - predicted next state vector
            - F - Jacobian of the transition function w.r.t. to ``X``
            - G - Jacobian of the transition function w.r.t. to the noise vector ``W``.

    Q : array_like, shape (n_epochs - 1, n_noises, n_noises) or (n_noises, n_noises)
        Process noise covariance matrix. Either constant or specified for each
        transition.
    measurements : list of n_epoch lists
        Each list contains triples (Z, h, R) with measurement vector, measurement
        function and measurement noise covariance. The measurement functions must
        have a signature ``h(k, X) -> Z_pred, H``, where:

            - k - epoch index
            - X - state vector
            - Z_pred - predicted measurement vector
            - H - measurement function Jacobian w.r.t. ``X``
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
    """
    RHO = 0.5
    MIN_ALPHA = 0.01
    TAU = 0.9
    ETA = 0.1

    def _build_lp(X, W):
        x0, F, G, measurements_lin, u, w = _build_linear_problem(X0, X, W, f,
                                                                 measurements)
        return (x0, F, G, measurements_lin, u, w,
                _eval_cost(x0, P0, w, Q, measurements_lin), np.sum(np.abs(u)))

    n_states = len(X0)
    n_epochs = len(measurements)

    Q = np.asarray(Q)
    if Q.ndim == 2:
        Q = np.resize(Q, (n_epochs - 1, *Q.shape))
    n_noises = Q.shape[-1]

    if (X0.shape != (n_states,) or P0.shape != (n_states, n_states) or
            Q.shape != (n_epochs - 1, n_noises, n_noises)):
        raise ValueError("Inconsistent input shapes")

    ekf_result = run_ekf(X0, P0, f, Q, measurements)

    X = deepcopy(ekf_result.X)
    W = np.zeros((n_epochs - 1, n_noises))

    mu = 1.0
    for iteration in range(max_iter):
        x0, F, G, measurements_lin, u, w, cost, cv_l1 = _build_lp(X, W)
        linear_result = run_kalman_smoother(x0, P0, F, G, Q, measurements_lin, u, w)
        qp_cost_change, qp_grad_dot_step = _eval_quadratic_step(
            x0, P0, Q, measurements_lin, w, linear_result.x, linear_result.w)
        mu = max(mu, qp_cost_change / (1 - RHO) / cv_l1)
        D = qp_grad_dot_step - mu * cv_l1
        merit = cost + mu * cv_l1

        alpha = 1.0
        while alpha > MIN_ALPHA:
            X_new = X + alpha * linear_result.x
            W_new = W + alpha * linear_result.w
            *_, u, _, cost_new, cv_l1 = _build_lp(X_new, W_new)
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

    return Bunch(X=X, P=linear_result.P, W=W, Q=linear_result.Q, Xf=ekf_result.X,
                 Pf=ekf_result.P)

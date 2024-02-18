from copy import deepcopy
import numpy as np
from scipy import linalg
from .linear import run_kalman_smoother
from .ekf import run_ekf
from ._common import verify_array, verify_function
from itertools import chain


def _build_linear_problem(X0, Xo, Wo, fs, Zs, hs):
    x0 = X0 - Xo[0]
    zs = []
    Hs = []
    ws = []
    us = []
    Fs = []
    Gs = []
    for X, X_next, W, f, Z, h in zip(Xo, chain(Xo[1:], [None]), chain(Wo, [None]),
                                     chain(fs, [None]), Zs, hs):
        Z_hat, H = h(X)
        zs.append(Z - Z_hat)
        Hs.append(H)

        if X_next is not None:
            X_pred, F, G = f(X, W)
            us.append(X_pred - X_next)
            ws.append(-W)
            Fs.append(F)
            Gs.append(G)

    return x0, Fs, Gs, zs, Hs, us, ws


def _eval_quadratic_step(x0, P0, Qs, zs, Hs, Rs, ws, xo, wo):
    def _eval_quadratic_item(x, z, R, H=None):
        Hx = x if H is None else H @ x
        RiHx = linalg.cho_solve(linalg.cho_factor(R), Hx)
        return 0.5 * np.dot(Hx, RiHx) - np.dot(z, RiHx), -np.dot(z, RiHx)

    cost_change, grad_dot_step = _eval_quadratic_item(xo[0], x0, P0)
    for x, z, H, R in zip(xo, zs, Hs, Rs):
        c, d = _eval_quadratic_item(x, z, R, H)
        cost_change += c
        grad_dot_step += d
    for w, wp, Q in zip(wo, ws, Qs):
        c, d = _eval_quadratic_item(w, wp, Q)
        cost_change += c
        grad_dot_step += d
    return cost_change, grad_dot_step


def _eval_cost(x0, P0, zs, Rs, ws, Qs):
    def _eval_quadratic(x, P):
        return 0.5 * np.dot(x, linalg.cho_solve(linalg.cho_factor(P), x))
    result = _eval_quadratic(x0, P0)
    for z, R in zip(zs, Rs):
        result += _eval_quadratic(z, R)
    for w, Q in zip(ws, Qs):
        result += _eval_quadratic(w, Q)
    return result


def _eval_cv_norm(us):
    return sum(np.linalg.norm(u, ord=1) for u in us)


def run_optimization(X0, P0, fs, Qs, Zs, hs, Rs, n_epoch, ftol=1e-8, max_iter=10):
    RHO = 0.5
    MIN_ALPHA = 0.01
    TAU = 0.9
    ETA = 0.1

    def _build_lp(X, W):
        x0, Fs, Gs, zs, Hs, us, ws = _build_linear_problem(X0, X, W, fs, Zs, hs)
        return x0, Fs, Gs, zs, Hs, us, ws, _eval_cost(x0, P0, zs, Rs, ws, Qs)

    Xf, Pf = run_ekf(X0, P0, fs, Qs, Zs, hs, Rs, n_epoch)

    fs = verify_function(fs, n_epoch - 1, 'fs')
    Qs = verify_array(Qs, n_epoch - 1, 'Qs')
    hs = verify_function(hs, n_epoch, 'hs')
    Rs = verify_array(Rs, n_epoch, 'Rs')

    Po = Pf
    Xo = deepcopy(Xf)
    Wo = [np.zeros(len(Q)) for Q in Qs]
    Qo = Qs

    mu = 1.0
    for iteration in range(max_iter):
        x0, Fs, Gs, zs, Hs, us, ws, cost = _build_lp(Xo, Wo)
        linear_result = run_kalman_smoother(x0, P0, Fs, Gs, Qs, zs, Hs, Rs, us,
                                            ws)
        qp_cost_change, qp_grad_dot_step = _eval_quadratic_step(x0, P0, Qs, zs, Hs, Rs,
                                                                ws, linear_result.xo,
                                                                linear_result.wo)
        cv_l1 = _eval_cv_norm(us)
        mu = max(mu, qp_cost_change / (1 - RHO) / cv_l1)
        D = qp_grad_dot_step - mu * cv_l1
        merit = cost + mu * cv_l1

        alpha = 1.0
        while alpha > MIN_ALPHA:
            Xo_new = []
            Wo_new = []
            for X, x in zip(Xo, linear_result.xo):
                Xo_new.append(X + alpha * x)
            for W, w in zip(Wo, linear_result.wo):
                Wo_new.append(W + alpha * w)

            x0, Fs, Gs, zs, Hs, us, ws, cost_new = _build_lp(Xo_new, Wo_new)
            merit_new = cost_new + mu * _eval_cv_norm(us)
            if merit_new < merit + ETA * alpha * D:
                break
            else:
                alpha *= TAU

        Xo = Xo_new
        Wo = Wo_new

        if abs(cost - cost_new) < ftol * cost:
            break

    if all(len(X) == len(Xf[0]) for X in Xf):
        Xf = np.asarray(Xf)
        Pf = np.asarray(Pf)
        Xo = np.asarray(Xo)
        Po = np.asarray(Po)

    if all(len(W) == len(Wo[0]) for W in Wo):
        Wo = np.asarray(Wo)
        Qo = np.asarray(Qo)

    return Xf, Pf, Xo, Po, Wo, Qo

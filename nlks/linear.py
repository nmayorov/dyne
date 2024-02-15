import numpy as np
from scipy import linalg


def run_linear_kalman_smoother(x0, P0, Fs, Gs, Qs, zs, Hs, Rs, us=None, ws=None):
    n_epoch = len(zs)
    if (
        len(Fs) + 1 != n_epoch
        or (us is not None and len(us) + 1 != n_epoch)
        or len(Gs) + 1 != n_epoch
        or len(Qs) + 1 != n_epoch
        or (ws is not None and len(ws) + 1 != n_epoch)
        or len(zs) != n_epoch
        or len(Hs) != n_epoch
        or len(Rs) != n_epoch
    ):
        raise ValueError("Inconsistent sizes of inputs")

    if us is None:
        us = [np.zeros(len(F)) for F in Fs]
    if ws is None:
        ws = [np.zeros(len(Q)) for Q in Qs]

    x = x0
    P = P0

    xf = []
    Pf = []
    Us = []
    rs = []
    Ms = []

    def update_and_append(x, P, z, H, R):
        x, P, U, r, M = kf_update(x, P, z, H, R)
        xf.append(x)
        Pf.append(P)
        Us.append(U)
        rs.append(r)
        Ms.append(M)
        return x, P

    for F, u, G, Q, w, z, H, R in zip(Fs, us, Gs, Qs, ws, zs, Hs, Rs):
        x, P = update_and_append(x, P, z, H, R)
        x = F @ x + G @ w + u
        P = F @ P @ F.T + G @ Q @ G.T

    x, P = update_and_append(x, P, zs[-1], Hs[-1], Rs[-1])
    lamb = np.zeros(len(x))
    Lamb = np.zeros((len(x), len(x)))

    xo = []
    Po = []
    wo = []
    Qo = []
    for x, P, F, G, Q, H, U, r, M in zip(
        reversed(xf),
        reversed(Pf),
        reversed(Fs),
        reversed(Gs),
        reversed(Qs),
        reversed(Hs),
        reversed(Us),
        reversed(rs),
        reversed(Ms),
    ):
        xo.append(x + P @ lamb)
        Po.append(P - P @ Lamb @ P)
        wo.append(Q @ G.T @ lamb)
        Qo.append(Q - Q @ G.T @ Lamb @ G @ Q)

        lamb = U.T @ lamb + r
        Lamb = U.T @ Lamb @ U + M

        lamb = F.T @ lamb
        Lamb = F.T @ Lamb @ F

    xo.append(xf[0] + Pf[0] @ lamb)
    Po.append(Pf[0] - Pf[0] @ Lamb @ Pf[0])

    wo.reverse()
    Qo.reverse()
    xo.reverse()
    Po.reverse()

    if all(len(x) == len(xf[0]) for x in xf):
        xf = np.asarray(xf)
        Pf = np.asarray(Pf)
        xo = np.asarray(xo)
        Po = np.asarray(Po)

    if all(len(w) == len(wo[0]) for w in wo):
        wo = np.asarray(wo)
        Qo = np.asarray(Qo)

    return xf, Pf, xo, Po, wo, Qo


def kf_update(x, P, z, H, R):
    S = H @ P @ H.T + R
    J = linalg.cho_solve(linalg.cho_factor(S), H).T
    K = P @ J
    e = z - H @ x
    U = np.eye(len(x)) - K @ H
    return x + K @ e, U @ P @ U.T + K @ R @ K.T, U, J @ e, J @ H

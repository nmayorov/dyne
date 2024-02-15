"""Extended Kalman Filter."""
import numpy as np
from .linear import kf_update
from ._common import verify_array, verify_function
from itertools import chain


def run_ekf(X0, P0, fs, Qs, Zs, hs, Rs, n_epoch):
    fs = verify_function(fs, n_epoch - 1, 'fs')
    Qs = verify_array(Qs, n_epoch - 1, 'Qs')
    hs = verify_function(hs, n_epoch, 'hs')
    Rs = verify_array(Rs, n_epoch, 'Rs')

    Xf = []
    Pf = []

    X = X0
    P = P0

    for f, Q, Z, h, R in zip(chain(fs, [None]), chain(Qs, [None]), Zs, hs, Rs):
        Zf, H = h(X)
        x, P, *_ = kf_update(np.zeros(H.shape[1]), P, Zf - Z, H, R)
        X -= x

        Xf.append(X)
        Pf.append(P)

        if f is not None and Q is not None:
            X, F, G = f(X)
            P = F @ P @ F.T + G @ Q @ G.T

    if all(len(Xi) == len(X) for Xi in Xf):
        Xf = np.asarray(Xf)
        Pf = np.asarray(Pf)

    return Xf, Pf

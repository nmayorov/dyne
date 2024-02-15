"""Extended Kalman Filter."""
import numpy as np
from .linear import kf_update
from ._common import verify_array, verify_function
from itertools import chain


def run_ekf(X0, P0, fs, Qs, Zs, hs, Rs, n_epoch):
    def verify_function(function, size, name):
        try:
            if len(function) != size:
                raise ValueError(f"Incorrect size of {name}")
        except TypeError:
            function = [function] * size
        return function

    def verify_array(array, size, name):
        array = np.asarray(array)
        if array.ndim == 2:
            array = [array] * size
        elif array.ndim == 3:
            if len(array) != size:
                raise ValueError(f"Incorrect size of {name}")
        else:
            raise ValueError(f"Incorrect shape of array {name}")
        return array

    fs = verify_function(fs, n_epoch - 1, 'fs')
    Qs = verify_array(Qs, n_epoch - 1, 'Qs')
    hs = verify_function(hs, n_epoch, 'hs')
    Rs = verify_array(Rs, n_epoch, 'Rs')

    Xf = []
    Pf = []

    X = X0
    P = P0

    for f, Q, Z, h, R in zip(fs, Qs, Zs, hs, Rs):
        Zf, H = h(X)
        x, P, *_ = kf_update(np.zeros(H.shape[1]), P, Zf - Z, H, R)
        X -= x

        Xf.append(X)
        Pf.append(P)

        X, F, G = f(X)
        P = F @ P @ F.T + G @ Q @ G.T

    if all(len(Xi) == len(X) for Xi in Xf):
        Xf = np.asarray(Xf)
        Pf = np.asarray(Pf)

    return Xf, Pf

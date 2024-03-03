import numpy as np


def check_measurements(measurements):
    if measurements is None:
        measurements = []

    result = []
    for epochs, Z, h, R in measurements:
        Z = np.asarray(Z)
        R = np.asarray(R)
        if R.ndim == 2:
            R = np.resize(R, (len(epochs), *R.shape))

        n = len(epochs)
        m = Z.shape[-1]
        if Z.shape != (n, m) or R.shape != (n, m, m):
            raise ValueError("Inconsistent shapes in measurements")

        result.append((epochs, Z, h, R))

    return result

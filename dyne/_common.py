import numpy as np


def check_input_arrays(X0, P0, Q, n_epochs):
    X0 = np.asarray(X0)
    P0 = np.asarray(P0)
    Q = np.asarray(Q)

    n_states = len(X0)
    if Q.ndim == 2:
        Q = np.resize(Q, (n_epochs - 1, *Q.shape))
    n_noises = Q.shape[-1]

    if (X0.shape != (n_states,) or P0.shape != (n_states, n_states) or
            Q.shape != (n_epochs - 1, n_noises, n_noises)):
        raise ValueError("Inconsistent input shapes")

    return X0, P0, Q, n_states, n_noises


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

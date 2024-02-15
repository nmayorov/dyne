import numpy as np
from scipy._lib._util import check_random_state


def generate_linear_pendulum(
    n_epoch=1000,
    x0=np.array([1.0, 0.0]),
    P0=np.diag([0.1**2, 0.05**2]),
    tau=0.1,
    T=10.0,
    eta=0.1,
    qf=0.03,
    sigma_angle=0.2,
    sigma_rate=0.1,
    rng=0,
):
    rng = check_random_state(rng)
    n_states = 2
    n_noises = 1
    n_obs = 2

    xs = np.empty((n_epoch, n_states))
    xs[0] = rng.multivariate_normal(x0, P0)
    Fs = np.empty((n_epoch - 1, n_states, n_states))
    omega = 2 * np.pi / T
    Fs[:] = [[1, tau], [-(omega**2) * tau, 1 - 2 * eta * omega * tau]]
    Gs = np.empty((n_epoch - 1, n_states, n_noises))
    Gs[:] = [[0], [1]]
    Qs = np.empty((n_epoch - 1, n_noises, n_noises))
    Qs[:] = tau * qf**2
    ws = np.empty((n_epoch - 1, n_noises))

    zs = np.empty((n_epoch, n_obs))
    Hs = np.empty((n_epoch, n_obs, n_states))
    Hs[:] = np.eye(2)
    Rs = np.empty((n_epoch, n_obs, n_obs))
    Rs[:] = np.diag([sigma_angle**2, sigma_rate**2])

    for i in range(n_epoch - 1):
        zs[i] = Hs[i] @ xs[i] + rng.multivariate_normal([0, 0], Rs[i])
        ws[i] = rng.multivariate_normal(np.zeros(len(Qs[i])), Qs[i])
        xs[i + 1] = Fs[i] @ xs[i] + Gs[i] @ ws[i]

    return x0, P0, xs, ws, Fs, Gs, Qs, zs, Hs, Rs


def generate_nonlinear_pendulum(
    n_epoch=1000,
    X0=np.array([0.5 * np.pi, 0]),
    P0=np.diag([0.1**2, 0.05**2]),
    tau=0.1,
    T=10.0,
    eta=0.5,
    xi=1.0,
    Q=np.diag([0.1**2, 0.01**2, 0.5**2]),
    sigma_angle=0.1,
    rng=0
):
    rng = check_random_state(rng)
    omega = 2 * np.pi / T

    def f(X, W=None):
        if W is None:
            W = np.zeros(3)

        omega_ = omega + W[0]
        eta_ = eta + W[1]
        X_next = np.array([
            X[0] + tau * X[1],
            X[1] + tau * (-omega_ ** 2 * np.sin(X[0])
                          - 2 * eta_ * omega_ * X[1] * (1 + xi * X[1] ** 2)
                          + W[2])
        ])
        F = np.array([
            [1, tau],
            [-tau * omega_ ** 2 * np.cos(X[0]),
             1 - 2 * tau * eta_ * omega_ * (1 + 3 * xi * X[1] ** 2)]
        ])
        G = np.array([
            [0, 0, 0],
            [-2 * tau * (omega_ * np.sin(X[0]) + eta_ * X[1] * (1 + xi * X[1] ** 2)),
             -2 * tau * omega_ * X[1] * (1 + xi * X[1] ** 2), tau]
        ])
        return X_next, F, G

    def h(X):
        return np.array([np.sin(X[0])]), np.array([[np.cos(X[0]), 0]])

    R = np.array([[sigma_angle ** 2]])

    X = rng.multivariate_normal(X0, P0)
    Xs = np.empty((n_epoch, 2))
    Ws = np.empty((n_epoch - 1, 3))
    Zs = np.empty((n_epoch, 1))

    for epoch in range(n_epoch):
        Xs[epoch] = X
        Zs[epoch] = h(X)[0] + rng.multivariate_normal(np.zeros(len(R)), R)

        if epoch + 1 < n_epoch:
            W = rng.multivariate_normal(np.zeros(len(Q)), Q)
            Ws[epoch] = W
            X, *_ = f(X, W)

    return X0, P0, Xs, Ws, f, Q, Zs, h, R

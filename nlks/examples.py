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

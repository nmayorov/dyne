"""Example of estimation problems."""
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
    """Prepare data for an example of linear pendulum with friction.

    The continuous system model is::

        dx1 / dt = x2
        dx2 / dt = -omega**2 * x1 - 2 * eta * omega * x2 + f

    with ``f`` being an external force. It is discretized with a time step `tau`,
    the external force is modeled as a random white sequence.

    The measurements consist of both x1 and x2 (angle and angular rate).
    """
    rng = check_random_state(rng)
    n_states = 2
    n_noises = 1
    n_obs = 2

    xt = np.empty((n_epoch, n_states))
    xt[0] = rng.multivariate_normal(x0, P0)

    omega = 2 * np.pi / T
    F = np.array([[1, tau], [-(omega ** 2) * tau, 1 - 2 * eta * omega * tau]])
    G = np.array([[0], [1]])
    Q = np.array([[tau * qf**2]])
    wt = np.empty((n_epoch - 1, n_noises))

    R = np.diag([sigma_angle**2, sigma_rate**2])
    H = np.identity(n_obs)
    z = []

    for i in range(n_epoch):
        z.append(H @ xt[i] + rng.multivariate_normal(np.zeros(n_obs), R))
        if i + 1 < n_epoch:
            wt[i] = rng.multivariate_normal(np.zeros(len(Q)), Q)
            xt[i + 1] = F @ xt[i] + G @ wt[i]

    return x0, P0, xt, wt, F, G, Q, n_epoch, [(np.arange(n_epoch), z, H, R)]


def generate_linear_pendulum_as_nl_problem(
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
    """Prepare data for an example of linear pendulum with friction.

    The continuous system model is::

        dx1 / dt = x2
        dx2 / dt = -omega**2 * x1 - 2 * eta * omega * x2 + f

    with ``f`` being an external force. It is discretized with a time step `tau`,
    the external force is modeled as a random white sequence.

    The measurements consist of both x1 and x2 (angle and angular rate).

    This function returns the problem as a general nonlinear problem which can be
    used for testing and verification purposes.
    """
    rng = check_random_state(rng)
    n_states = 2
    n_noises = 1
    n_obs = 2

    xt = np.empty((n_epoch, n_states))
    xt[0] = rng.multivariate_normal(x0, P0)

    omega = 2 * np.pi / T
    F = np.array([[1, tau], [-(omega ** 2) * tau, 1 - 2 * eta * omega * tau]])
    G = np.array([[0], [1]])
    Q = np.array([[tau * qf**2]])
    wt = np.empty((n_epoch - 1, n_noises))

    R = np.diag([sigma_angle**2, sigma_rate**2])
    H = np.identity(n_obs)
    z = []

    def f(k, X, W=None, with_jacobian=True):
        if W is None:
            W = np.zeros(1)
        X_next = F @ X + G @ W
        if not with_jacobian:
            return X_next
        return X_next, F, G

    def h(k, X, with_jacobian=True):
        return (H @ X, H) if with_jacobian else H @ X

    for i in range(n_epoch):
        z.append(H @ xt[i] + rng.multivariate_normal(np.zeros(n_obs), R))
        if i + 1 < n_epoch:
            wt[i] = rng.multivariate_normal(np.zeros(len(Q)), Q)
            xt[i + 1] = F @ xt[i] + G @ wt[i]

    return x0, P0, xt, wt, f, Q, n_epoch, [(np.arange(n_epoch), z, h, R)]


def generate_nonlinear_pendulum(
    n_epochs=1000,
    X0=np.array([0.5 * np.pi, 0]),
    P0=np.diag([0.1**2, 0.05**2]),
    tau=0.1,
    T=10.0,
    eta=0.5,
    xi=1.0,
    sigma_omega=0.1,
    sigma_eta=0.01,
    sigma_f=0.5,
    sigma_measurement_x=0.1,
    rng=0
):
    """Prepare data for an example of nonlinear pendulum with friction.

    The continuous time system model is::

        dx1 / dt = x2
        dx2 / dt = -omega**2 * sin(x1) - 2 * eta * omega * x2 * (1 + xi * x2**2) + f

    with ``f`` being an external force. It is discretized with a time step `tau`.
    It is discretized with a time step `tau`. The parameters ``omega`` and ``eta`` are
    randomly perturbed by noise at each epoch, the external force is modeled as a
    random white sequence.
    """
    rng = check_random_state(rng)
    omega = 2 * np.pi / T
    Q = np.diag([sigma_omega**2, sigma_eta**2, sigma_f**2])

    def f(k, X, W=None, with_jacobian=True):
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
        if not with_jacobian:
            return X_next
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

    def h(k, X, with_jacobian=True):
        Z = np.array([np.sin(X[0])])
        if not with_jacobian:
            return Z
        return Z, np.array([[np.cos(X[0]), 0]])

    R = np.array([[sigma_measurement_x ** 2]])
    X = rng.multivariate_normal(X0, P0)
    Xt = np.empty((n_epochs, 2))
    Wt = np.empty((n_epochs - 1, 3))
    Z = []

    for k in range(n_epochs):
        Xt[k] = X
        Z.append(h(k, X, with_jacobian=False)
                 + rng.multivariate_normal(np.zeros(len(R)), R))

        if k + 1 < n_epochs:
            Wt[k] = rng.multivariate_normal(np.zeros(len(Q)), Q)
            X, *_ = f(k, X, Wt[k])

    return X0, P0, Xt, Wt, f, Q, n_epochs, [(np.arange(n_epochs), Z, h, R)]

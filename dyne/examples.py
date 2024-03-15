"""Example of estimation problems."""
from dataclasses import dataclass
import numpy as np
from scipy._lib._util import check_random_state
from .util import solve_ivp_with_jac
from scipy.integrate import solve_ivp


@dataclass
class LinearProblemExample:
    """Example for a linear estimation problem.

    Parameters
    ----------
    x0 : ndarray, shape (n_states,)
        Initial state.
    P0 : ndarray, shape (n_states, n_states)
        Initial covariance.
    F : ndarray, shape (n_epochs - 1, n_states, n_states) or (n_states, n_states)
        Process matrices.
    G : ndarray, shape (n_epochs - 1, n_states, n_noises) or (n_states, n_noises)
        Noise input matrices.
    Q : ndarray, shape (n_epochs - 1, n_noises, n_noises) or (n_noises, n_noises)
        Process noise covariance matrices.
    n_epochs : int
        Number of epochs for estimation.
    measurements : list
        List of measurement structures.
        See `dyne.run_kalman_smoother` for a detailed definition.
    xt : ndarray, shape (n_epochs, n_states)
        True state for each epoch.
    wt : ndarray, shape (n_epochs - 1, n_noises)
        True noise values.
    """
    x0 : np.ndarray
    P0 : np.ndarray
    F : np.ndarray
    G : np.ndarray
    Q : np.ndarray
    n_epochs : int
    measurements : list | None
    xt : np.ndarray
    wt : np.ndarray


def generate_linear_pendulum(
    n_epochs=1000,
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
    """Generate data for an example of a linear pendulum with friction.

    The continuous system model is::

        dx1 / dt = x2
        dx2 / dt = -omega**2 * x1 - 2 * eta * omega * x2 + f

    with ``f`` being an external force. It is discretized with a time step `tau`,
    the external force is modeled as a random white sequence.

    The measurements consist of both x1 and x2 (angle and angular rate).

    Parameters
    ----------
    n_epochs : int
        Number of epochs for simulation.
    x0 : array_like, shape (2,)
        Initial state.
    P0 : array_like, shape (2, 2)
        Initial state covariance.
    tau : float
        Time step in seconds.
    T : float
        Pendulum period in seconds.
    eta : float
        Dimensionless friction coefficient.
    qf : float
        Intensity of force process in rad/s/sqrt(s)
    sigma_angle : float
        Accuracy of angle measurements in rad.
    sigma_rate : float
        Accuracy of angular rate measurements in rad/s.
    rng : None, int or `numpy.random.RandomState`
        Seed to create or already created RandomState. None (default) corresponds to
        nondeterministic seeding.

    Returns
    -------
    LinearProblemExample
    """
    rng = check_random_state(rng)
    n_states = 2
    n_noises = 1
    n_obs = 2

    x0 = np.asarray(x0)
    P0 = np.asarray(P0)
    xt = np.empty((n_epochs, n_states))
    xt[0] = rng.multivariate_normal(x0, P0)

    omega = 2 * np.pi / T
    F = np.array([[1, tau], [-(omega ** 2) * tau, 1 - 2 * eta * omega * tau]])
    G = np.array([[0], [1]])
    Q = np.array([[tau * qf**2]])
    wt = np.empty((n_epochs - 1, n_noises))

    R = np.diag([sigma_angle**2, sigma_rate**2])
    H = np.identity(n_obs)
    z = []

    for i in range(n_epochs):
        z.append(H @ xt[i] + rng.multivariate_normal(np.zeros(n_obs), R))
        if i + 1 < n_epochs:
            wt[i] = rng.multivariate_normal(np.zeros(len(Q)), Q)
            xt[i + 1] = F @ xt[i] + G @ wt[i]

    return LinearProblemExample(x0, P0, F, G, Q, n_epochs,
                                [(np.arange(n_epochs), z, H, R)], xt, wt)


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


def generate_falling_body(total_time=60, time_step=1,
                          X0t=np.array([3e5, 2e4, 1e-3]),
                          X0=np.array([3e5, 2e4, 3e-5]),
                          P0=np.diag([1e3**2, 2e3**2, 1e-2**2]),
                          rtol=1e-10,
                          rng=0):
    """
    This example is taken from "Optimal Estimation of Dynamic Systems", 2nd edition,
    sec. 3.7, where it is put as a demonstration of supposedly superior performance of
    UKF versus EKF.

    The example models the fall of a body in air with changing density using 3 states:
    altitude, downward velocity and drag coefficient. The gravity is not included as a
    negligible effect for high velocities. The system evolution is described by ODEs::

        dx1 / dt = -x2
        dx2 / dt = -exp(-alpha * x1) * x2**2 * x3
        dx3 / dt = 0

    There is no process noise. Range measurements from a radar to the body are available
    every second.

    The system is discretized in a proper way using Runge-Kutta integration with
    simultaneous Jacobian evaluation.

    As in the book specific true and filter initial conditions are used.
    """
    ALPHA = 5e-5
    RADAR_M = 1e5
    RADAR_Z = 1e5

    rng = check_random_state(rng)

    def ode_rhs(t, X):
        return [-X[1], -np.exp(-ALPHA * X[0]) * X[1] ** 2 * X[2], 0.0]

    def ode_rhs_jac(t, X):
        return np.array([
            [0, -np.exp(ALPHA * X[0]), 0],
            [ALPHA * X[1] ** 2 * X[2], -2 * X[1] * X[2], -X[1] ** 2],
            [0, 0, 0]
        ]) * np.exp(-ALPHA * X[0])

    def f(k, X, W=None, with_jacobian=True):
        _, X, F, _ = solve_ivp_with_jac(ode_rhs, ode_rhs_jac,
                                        [k * time_step, (k + 1) * time_step], X,
                                        rtol=rtol)
        return (X[-1], F[-1], np.empty((3, 0))) if with_jacobian else X[-1]

    def h(k, X, with_jacobian=True):
        Z = np.atleast_1d(np.hypot(RADAR_M, X[0] - RADAR_Z))
        return (Z, np.array([[(X[0] - RADAR_Z) / Z[0], 0, 0]])) if with_jacobian else Z

    Xt = solve_ivp(ode_rhs, [0, total_time], X0t,
                   t_eval=np.arange(0, total_time, time_step), rtol=rtol).y.T
    Z = []
    R = np.array([[1e2**2]])
    for k, X in enumerate(Xt):
        Z.append(h(k, X, with_jacobian=False) + rng.multivariate_normal(np.zeros(1), R))
    n_epochs = len(Z)
    return (X0, P0, Xt, np.empty((n_epochs - 1, 0)), f, np.empty((n_epochs - 1, 0, 0)),
            n_epochs, [(np.arange(n_epochs), Z, h, R)])


def generate_lorenz_system(
    total_time=15,
    time_step=1e-2,
    X0t=np.array([10, -5, 5]),
    X0=None,
    P0=np.diag([0.1, 0.1, 5])**2,
    sigma=10.0,
    beta=8.0/3.0,
    rho=28.0,
    q=np.zeros(3),
    sigma_measurement_x=1.0,
    measurement_subsample=1,
    rtol=1e-10,
    rng=0
):
    """Prepare data for an example of Lorenz system.

    The continuous time system model is::

        dx / dt = sigma * (y - x) + w_x
        dy / dt = x * (rho - z) - y + w_y
        dz / dt = x * y - beta * z + w_z

    The system was studied by Edward Lorenz and relate the thermal properties
    of a two-dimensional fluid layer. With default parameters sigma=10,
    beta=3/8, rho=28 the sytem exhibits chaotic behavior.
    It is discretized with a time step `time_step`. For simplification noise is
    applied in discrete time.
    """
    rng = check_random_state(rng)
    n_epochs = np.round(total_time / time_step).astype(int)

    noises = np.asarray(q)
    n_noises = np.sum(noises > 0)
    G = np.zeros((3, n_noises))
    Q = np.zeros((n_noises, n_noises))
    j = 0
    for i, s in enumerate(noises):
        if s > 0:
            G[i, j] = 1
            Q[j, j] = s**2 * time_step
            j = j + 1

    def lorenz(t, state):
        x, y, z = state
        return np.array([sigma * (y - x),
                         x * (rho - z) - y,
                         x*y - beta*z])

    def lorenz_jacobian(t, state):
        x, y, z = state
        F = np.zeros((3, 3))

        F[0, 0] = -sigma
        F[0, 1] =  sigma
        F[0, 2] =  0
        F[1, 0] =  rho - z
        F[1, 1] = -1
        F[1, 2] = -x
        F[2, 0] =  y
        F[2, 1] =  x
        F[2, 2] = -beta

        return F

    def f(k, X, W=None, with_jacobian=True):
        _, X, F, _ = solve_ivp_with_jac(lorenz, lorenz_jacobian,
                                        [k * time_step, (k + 1) * time_step], X,
                                        rtol=rtol, method='DOP853')
        wk = G @ W if W is not None else 0
        return (X[-1] + wk, F[-1], G) if with_jacobian else X[-1] + wk

    def h(k, X, with_jacobian=True):
        Z = np.atleast_1d(X[0])
        return (Z, np.array([[1, 0, 0]])) if with_jacobian else Z

    X = X0t
    Xt = np.empty((n_epochs, 3))
    Wt = np.empty((n_epochs - 1, n_noises))
    R = np.array([[sigma_measurement_x ** 2]])
    Z = []
    measurement_epochs = []

    for k in range(n_epochs):
        Xt[k] = X
        if k % measurement_subsample == 0:
            vk = rng.multivariate_normal(np.zeros(len(R)), R)
            Z.append(h(k, X, with_jacobian=False) + vk)
            measurement_epochs.append(k)

        if k + 1 < n_epochs:
            if len(Q) > 0:
                Wt[k] = rng.multivariate_normal(np.zeros(len(Q)), Q)
            X, *_ = f(k, X, Wt[k])

    Q = np.array((n_epochs - 1) * [Q])
    if X0 is None:
        X0 = X0t + rng.multivariate_normal(np.zeros(len(P0)), P0)

    return X0, P0, Xt, Wt, f, Q, n_epochs, [(measurement_epochs, Z, h, R)]

"""Example of estimation problems."""
from dataclasses import dataclass
import numpy as np
from scipy._lib._util import check_random_state
from .util import solve_ivp_with_jac
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation


@dataclass
class LinearProblemExample:
    """Example of a linear estimation problem.

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
        List of linear measurement structures.
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
    measurements : list
    n_epochs : int
    xt : np.ndarray
    wt : np.ndarray


@dataclass
class NonlinearProblemExample:
    """Example of a nonlinear estimation problem.

    Parameters
    ----------
    X0 : ndarray, shape (n_states,)
        Initial state.
    P0 : ndarray, shape (n_states, n_states)
        Initial covariance.
    f : callable
        Process function, see `dyne.util.process_callable`.
    Q : ndarray, shape (n_epochs - 1, n_noises, n_noises) or (n_noises, n_noises)
        Process noise covariance matrices.
    n_epochs : int
        Number of epochs for estimation.
    measurements : list
        List of nonlinear measurement structures.
        See `dyne.run_ekf` for a detailed definition.
    Xt : ndarray, shape (n_epochs, n_states)
        True state for each epoch.
    Wt : ndarray, shape (n_epochs - 1, n_noises)
        True noise values.
    """
    X0 : np.ndarray
    P0 : np.ndarray
    f : callable
    Q : np.ndarray
    measurements : list
    n_epochs : int
    Xt : np.ndarray
    Wt : np.ndarray


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

    return LinearProblemExample(x0, P0, F, G, Q, [(np.arange(n_epochs), z, H, R)],
                                n_epochs, xt, wt)


def generate_linear_pendulum_as_nl_problem(
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

    This function returns the problem defined in `generate_linear_pendulum` as a
    general nonlinear problem which can be used for testing and verification purposes.

    Returns
    -------
    NonlinearProblemExample
    """
    lin_problem = generate_linear_pendulum(n_epochs, x0, P0, tau, T, eta, qf,
                                           sigma_angle, sigma_rate, rng)
    meas_epochs, z, H, R = lin_problem.measurements[0]

    def f(k, X, W=None, with_jacobian=True):
        if W is None:
            W = np.zeros(1)
        X_next = lin_problem.F @ X + lin_problem.G @ W
        if not with_jacobian:
            return X_next
        return X_next, lin_problem.F, lin_problem.G

    def h(k, X, with_jacobian=True):
        return (H @ X, H) if with_jacobian else H @ X

    return NonlinearProblemExample(
        lin_problem.x0, lin_problem.P0, f, lin_problem.Q, [(meas_epochs, z, h, R)],
        lin_problem.n_epochs, lin_problem.xt, lin_problem.wt)


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
    sigma_angle=0.1,
    rng=0
):
    """Generate data for an example of a nonlinear pendulum with friction.

    The continuous time system model is::

        dx1 / dt = x2
        dx2 / dt = -omega**2 * sin(x1) - 2 * eta * omega * x2 * (1 + xi * x2**2) + f

    with ``f`` being an external force. It is discretized with a time step `tau`.
    The parameters ``omega`` and ``eta`` are randomly perturbed by noise at each epoch,
    the external force is modeled as a random white sequence.

    The measurements of ``x1`` (angle) are available.

    Parameters
    ----------
    n_epochs : int
        Number of epochs for simulation.
    X0 : array_like, shape (2,)
        Initial state.
    P0 : array_like, shape (2, 2)
        Initial state covariance.
    tau : float
        Time step in seconds.
    T : float
        Pendulum period in seconds.
    eta : float
        Dimensionless friction coefficient.
    xi : float
        Friction nonlinearity coefficient.
    sigma_omega : float
        Standard deviation of ``omega`` disturbance.
    sigma_eta : float
        Standard deviation of ``eta`` disturbance.
    sigma_f : float
        Standard deviation of external force sequence.
    sigma_angle : float
        Accuracy of angle measurements in rad.
    rng : None, int or `numpy.random.RandomState`
        Seed to create or already created RandomState. None (default) corresponds to
        nondeterministic seeding.

    Returns
    -------
    NonlinearProblemExample
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

    R = np.array([[sigma_angle ** 2]])
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

    return NonlinearProblemExample(X0, P0, f, Q, [(np.arange(n_epochs), Z, h, R)],
                                   n_epochs, Xt, Wt)


def generate_falling_body(total_time=60, time_step=1,
                          X0t=np.array([3e5, 2e4, 1e-3]),
                          X0=np.array([3e5, 2e4, 3e-5]),
                          P0=np.diag([1e3**2, 2e3**2, 1e-2**2]),
                          rtol=1e-10,
                          rng=0):
    """Generate data for an example with a falling body in dense air.

    This example is taken from "Optimal Estimation of Dynamic Systems", 2nd edition,
    sec. 3.7, where it is put as a demonstration of supposedly superior performance of
    UKF over EKF.

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

    Parameters
    ----------
    total_time : float
        Total simulation time.
    time_step : float
        Discrete time step.
    X0t : array_like, shape (2,)
        True initial state.
    X0 : array_like, shape (2,) or None
        Initial state for estimation. The concrete value proposed in [1] is set
        by default. If None, then it is randomly generated from `X0t` and `P0`.
    P0 : array_like, shape (3, 3)
        Initial covariance.
    rtol : float
        Tolerance parameter (relative) for Runge-Kutta integrator.
    rng : None, int or `numpy.random.RandomState`
        Seed to create or already created RandomState. None (default) corresponds to
        nondeterministic seeding.

    Returns
    -------
    NonlinearProblemExample
    """
    ALPHA = 5e-5
    RADAR_M = 1e5
    RADAR_Z = 1e5

    rng = check_random_state(rng)
    X0t = np.asarray(X0t)
    P0 = np.asarray(P0)
    if X0 is None:
        X0 = X0t + rng.multivariate_normal(np.zeros_like(X0t), P0)
    else:
        X0 = np.asarray(X0)

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
    return NonlinearProblemExample(
        X0, P0, f, np.empty((n_epochs - 1, 0, 0)), [(np.arange(n_epochs), Z, h, R)],
        n_epochs, Xt, np.empty((n_epochs - 1, 0)))


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
    sigma_x=1.0,
    measurement_subsample=1,
    rtol=1e-10,
    rng=0
):
    """Generate data for an example of Lorenz system.

    The continuous time system model is::

        dx / dt = sigma * (y - x) + w_x
        dy / dt = x * (rho - z) - y + w_y
        dz / dt = x * y - beta * z + w_z

    The system was studied by Edward Lorenz, and it is related to thermal properties
    of a two-dimensional fluid layer. With parameters sigma=10,
    beta=3/8, rho=28 (all set by default) the system exhibits chaotic behavior.

    It is discretized with a time step `time_step` using a "DOP853" Runge-Kutta
    integrator. The process noise is introduce in a discrete fashion at the end
    of each integration step.

    Measurements of ``x`` variable are available.

    Parameters
    ----------
    total_time : float
        Total time of the simulation.
    time_step : float
        Discretization time step.
    X0t : array_like, shape (3,)
        Initial true state.
    X0 : array_like or None
        Initial estimator state. If None, generate it randomly from `X0t` and `P0`.
    P0 : array_like, shape (3, 3)
        Initial covariance matrix.
    sigma, beta, rho : float
        Model parameters.
    q : array_like, shape (3,)
        Additive process noise defined in a continuous sense.
        Standard deviation of the discrete noise is computed as ``q * sqrt(dt)``, where
        ``dt`` is `time_step`.
    sigma_x : float
        Standard deviation of measurement noise of variable ``x``.
    measurement_subsample : int
        Subsample factor for measurement generation, i.e. generate measurement
        every m-th sample, where m is `measurement_subsample`.
    rtol : float
        Tolerance parameter (relative) for ODE integrator.
    rng : None, int or `numpy.random.RandomState`
        Seed to create or already created RandomState. None (default) corresponds to
        nondeterministic seeding.

    Returns
    -------
    NonlinearProblemExample
    """
    rng = check_random_state(rng)
    n_epochs = int(np.round(total_time / time_step))

    q = np.asarray(q)
    n_noises = np.sum(q > 0)
    G = np.zeros((3, n_noises))
    G[q > 0, np.arange(n_noises)] = 1
    Q = np.diag(q[q > 0] ** 2)

    def lorenz(t, X):
        x, y, z = X
        return np.array([sigma * (y - x),
                         x * (rho - z) - y,
                         x*y - beta*z])

    def lorenz_jacobian(t, X):
        x, y, z = X
        return np.array([
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ])

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
    R = np.array([[sigma_x ** 2]])
    Z = []
    measurement_epochs = []

    for k in range(n_epochs):
        Xt[k] = X
        if k % measurement_subsample == 0:
            Z.append(h(k, X, with_jacobian=False) +
                     rng.multivariate_normal(np.zeros(len(R)), R))
            measurement_epochs.append(k)

        if k + 1 < n_epochs:
            if len(Q) > 0:
                Wt[k] = rng.multivariate_normal(np.zeros(len(Q)), Q)
            X = f(k, X, Wt[k], with_jacobian=False)

    if X0 is None:
        X0 = X0t + rng.multivariate_normal(np.zeros(len(P0)), P0)

    return NonlinearProblemExample(X0, P0, f, Q, [(measurement_epochs, Z, h, R)],
                                   n_epochs, Xt, Wt)


def generate_magnetic_heading(
    total_time=120,
    time_step=0.1,
    rph_mean=[3, -5, 0],
    rph_change_amplitude=5,
    rph_change_period=[5, 5, 30],
    rph_change_phase_offset=[0, 90, 45],
    magnetic_field_n=[10, 3, 40],
    mag_bias=[30, -20, 70],
    X0=None,
    P0=np.diag([180, 50, 50, 50])**2,
    q=np.zeros(4),
    measurement_sd=[1, 1, 1],
    rng=0
):
    """Generate data for exapmle of heading estimation using biased magnetometer data.

    Roll, pitch and heading angles are varying according to a given harmonic law.
    Exact values of roll and pitch are known, heading and magnetometer biases
    should be estimated.

    The discrete time system model is::

        h[k+1] = h[k] + dh[k] + w_h[k]
        bx[k+1] = bx[k] + w_x[k]
        by[k+1] = by[k] + w_y[k]
        bz[k+1] = bz[k] + w_z[k]

    Where::

        h  - heading
        dh - known heading increment, used as control
        bx, by, bz - magnetometer biases, components of `mag_bias`
        wx, wy, wz - discrete noise

    The measurement model is::

        [mx, my, mz] = C_bn @ [mn, me, md] + [bx, by, bz]

    Where::

        C_bn - rotation matrix from ``body`` to ``world`` frame
        mn, me, md - magnetic field in ``world`` frame, componets of `magnetic_field_n`
        bx, by, bz - magnetometer biases, components of `mag_bias`
        mx, my, mz - magnetometer measurements

    Parameters
    ----------
    total_time : float
        Total time of the simulation.
    time_step : float
        Discretization time step.
    rph_mean: array_like, shape (3,)
        Mean value of angles in degrees.
    rph_change_amplitude: float or array_like, shape (3,)
        Amplitude of angles variation in degrees.
    rph_change_period: float or array_like, shape (3,)
        Period of sinusoidal angles change in seconds.
    rph_change_phase_offset: array_like, shape (3,)
        Phase offset for sinusoid part in degrees.
    magnetic_field_n: array_like, shape (3,)
        Magnetic field in ``world`` frame.
    mag_bias: array_like, shape (3,)
        Magnetometer measurements bias.
    X0 : array_like, shape (4,) or None
        Initial estimator state. If None, generate it randomly from trajectory and `P0`.
    P0 : array_like, shape (4, 4)
        Initial covariance matrix.
    q : array_like, shape (4,)
        Additive process noise defined in a continuous sense.
        Standard deviation of the discrete noise is computed as ``q * sqrt(dt)``, where
        ``dt`` is `time_step`.
    measurement_sd : array_like, shape (3,)
        Standard deviation of measurement noise.
    rng : None, int or `numpy.random.RandomState`
        Seed to create or already created RandomState. None (default) corresponds to
        nondeterministic seeding.

    Returns
    -------
    NonlinearProblemExample
    """
    rng = check_random_state(rng)
    n_epochs = np.round(total_time / time_step).astype(int)
    time = time_step * np.arange(n_epochs)
    phase = (2 * np.pi * time[:, None] / rph_change_period +
             np.deg2rad(rph_change_phase_offset))
    rph = (np.atleast_2d(rph_mean) +
           np.atleast_2d(rph_change_amplitude) * np.sin(phase))
    mn, me, md = magnetic_field_n

    q = np.asarray(q)
    n_noises = np.sum(q > 0)
    G = np.zeros((4, n_noises))
    G[q > 0, np.arange(n_noises)] = 1
    Q = np.diag(q[q > 0] ** 2)

    def f(k, X, W=None, with_jacobian=True):
        X = np.array(X)
        X[0] += (rph[k+1, 2] - rph[k, 2])
        F = np.eye(4, 4)
        wk = G @ W if W is not None else 0
        return (X + wk, F, G) if with_jacobian else X + wk

    def h(k, X, with_jacobian=True):
        rph_deg = np.array(rph[k])
        rph_deg[2] = X[0]
        C_nb = Rotation.from_euler('xyz', rph_deg, degrees=True).as_matrix()
        Z = C_nb.T @ magnetic_field_n + X[1:]
        rph_rad = np.deg2rad(rph_deg)

        sr = np.sin(rph_rad[0])
        cr = np.cos(rph_rad[0])
        sp = np.sin(rph_rad[1])
        cp = np.cos(rph_rad[1])
        sh = np.sin(rph_rad[2])
        ch = np.cos(rph_rad[2])

        H = np.zeros((3, 4))
        H[0, 0] = me*ch*cp - mn*sh*cp
        H[1, 0] = me*(-sh*cr + sp*sr*cr) + mn*(-sh*sp*sr - ch*cr)
        H[2, 0] = me*( sh*sr + sp*ch*cr) + mn*(-sh*sp*cr + sr*ch)
        H = np.deg2rad(H)
        H[:, 1:] = np.eye(3)

        return (Z, H) if with_jacobian else Z

    X0t = np.zeros(4)
    X0t[0] = rph[0, 2]
    X0t[1:] = mag_bias

    X = X0t
    Xt = np.empty((n_epochs, 4))
    Wt = np.empty((n_epochs - 1, n_noises))
    R = np.diag(measurement_sd)**2
    Z = []
    measurement_epochs = np.arange(n_epochs)

    for k in range(n_epochs):
        Xt[k] = X
        vk = rng.multivariate_normal(np.zeros(len(R)), R)
        Z.append(h(k, X, with_jacobian=False) + vk)

        if k + 1 < n_epochs:
            if len(Q) > 0:
                Wt[k] = rng.multivariate_normal(np.zeros(len(Q)), Q)
            X = f(k, X, Wt[k], with_jacobian=False)

    if X0 is None:
        X0 = X0t + rng.multivariate_normal(np.zeros(len(P0)), P0)

    return NonlinearProblemExample(X0, P0, f, Q, [(measurement_epochs, Z, h, R)],
                                   n_epochs, Xt, Wt)

"""Utility functions."""
import numpy as np
from scipy.integrate import solve_ivp


class Bunch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join(['{}: {}'.format(k.rjust(m), type(v))
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def compute_rms(data):
    """Compute root-mean-square of data along 0 axis."""
    return np.mean(np.square(data), axis=0) ** 0.5


def solve_ivp_with_jac(fun, jac, t_span, X0, method='RK45', t_eval=None, **options):
    """Solve IVP for ODE and compute solution Jacobian w.r.t. initial conditions.

    The Jacobian w.r.t. the initial conditions can be found as the solution to the
    following matrix linear ODE with the initial condition:

        dF / dt = J(t, X) @ F
        F(t0) = I

    It is computed along with the solution ``X`` by considering the augmented ODE with
    ``X`` and (flattened) elements of the Jacobian.

    The function does not support analytical Jacobian for implicit methods,
    because it requires knowledge of the derivatives of ``J`` in the augmented system,
    which is cumbersome. Do not pass it in `options`.

    For the detailed description of the accepted parameters refer to
    `scipy.integrate.solve_ivp`.

    Parameters
    ----------
    fun : callable
        Right-hand side of the ODE, called as ``fun(t, X)``.
    jac : callable
        Jacobian of `fun` with respect to its second argument, i.e. ``X``.
    t_span : tuple with 2 elements
        Start and end integration times.
    X0 : array_like
        Initial state.
    method : string, optional
        One of the methods supported by `solve_ivp`. Default is 'RK45'.
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    options
        Other keyword arguments which will be passed to `scipy.integrate.solve_ivp`.

    Returns
    -------
    t : ndarray, shape (n_points,)
        Time points.
    X : ndarray, shape (n_points, n_states)
        State solution at `t`.
    F : ndarray, shape (n_points, n_states, n_states)
        Jacobian of the solution w.r.t. to the initial conditions at `t`.
    solve_ivp_solution : Bunch object
        Solution object returned from `solve_ivp`.
    """
    n_states = len(X0)

    def fun_augmented(t, y):
        X = y[:n_states]
        F = y[n_states:].reshape(n_states, n_states)
        dXdt = fun(t, X)
        dFdt = np.asarray(jac(t, X)) @ F
        return np.hstack((dXdt, dFdt.ravel()))

    y0 = np.hstack((X0, np.identity(n_states).ravel()))
    solution = solve_ivp(fun_augmented, t_span, y0, method=method, t_eval=t_eval,
                         **options)
    X = solution.y[:n_states].T
    F = solution.y[n_states:].T.reshape(-1, n_states, n_states)
    return solution.t, X, F, solution


def process_callable(k, X, W=None, with_jacobian=True):
    """Process callable interface.

    This function stub is included to conveniently describe the expected interface
    of process callables (denoted as ``f``) used in the estimation algorithms provided
    in the package.

    Parameters
    ----------
    k : int
        Epoch index at which the function is evaluated. That is the function might
        explicitly depend on the epoch index.
    X : ndarray, shape (n_states,)
        State vector.
    W : ndarray, shape (n_noises,) or None, optional
        Noise vector. If None (default) must be interpreted as zeros with appropriate
        size.
    with_jacobian : bool, optional
        Whether to return function Jacobian with respect to X and W.
        Default is True.

    Returns
    -------
    X_next : ndarray, shape (n_states,)
        Computed value of ``f_k(X, W)``.
    F : ndarray, shape (n_states, n_states)
        Jacobian of ``f`` with respect to ``X``. Must be returned only when
        `with_jacobian` is True.
    G : ndarray, shape (n_states, n_noises)
        Jacobian of ``f`` with respect to ``W``. Must be returned only when
        `with_jacobian` is True.
    """
    pass


def measurement_callable(k, X, with_jacobian=True):
    """Measurement callable interface.

    This function stub is included to conveniently describe the expected interface
    of measurement callables (denoted as ``h``) used in the estimation algorithms
    provided in the package.

    Parameters
    ----------
    k : int
        Epoch index at which the function is evaluated, that is the function might
        explicitly depend on the epoch index.
    X : ndarray, shape (n_states,)
        State vector.
    with_jacobian : bool, optional
        Whether to return function Jacobian with respect to X. Default is True.

    Returns
    -------
    Z : ndarray, shape (n_meas,)
        Compute value of ``h_k(X)``.
    H : ndarray, shape (n_meas, n_states)
        Jacobian of ``h`` with respect to ``X``. Must be returned only when
        `with_jacobian` is True.
    """
    pass

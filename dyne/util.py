"""Utility functions."""
import numpy as np


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


def rms(data):
    return np.mean(np.square(data), axis=0) ** 0.5


def process_callable(k, X, W=None, with_jacobian=True):
    """Process callable interface.

    This function stub is included to conveniently describe the expected interface
    of process callables (denoted as ``f``) used in the estimation algorithms provided
    in the package.

    Parameters
    ----------
    k : int
        Epoch index at which the function is evaluated, that is the function might
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
    provided  in the package.

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

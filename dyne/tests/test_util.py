import numpy as np
from numpy.testing import assert_allclose
import dyne


def test_solve_ivp_with_jac():
    y0 = 1

    def ode(t, y):
        return t * y**2 + 2 * t * y

    def jac(t, y):
        return np.atleast_2d(2 * t * y + 2 * t)

    def gt(t):
        return 2 / ((1 + 2/y0) * np.exp(-t**2) - 1)

    def gt_jac(t):
        return 4 * np.exp(-t ** 2) / ((y0 + 2) * np.exp(-t ** 2) - y0) ** 2

    t, y, F, _ = dyne.util.solve_ivp_with_jac(ode, jac, [0, 1], [y0])
    assert_allclose(y.ravel(), gt(t), rtol=1e-3)
    assert_allclose(F.ravel(), gt_jac(t), rtol=1e-3)

import numpy as np
from nlks import examples, run_kalman_smoother, util


def test_kalman_smoother():
    x0, P0, xt, wt, F, G, Q, measurements = examples.generate_linear_pendulum()
    result = run_kalman_smoother(x0, P0, F, G, Q, measurements)

    efn = (result.xf - xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.xo - xt) / np.diagonal(result.Po, axis1=1, axis2=2) ** 0.5

    assert np.all(util.rms(result.xo - xt) < util.rms(result.xf - xt))
    assert np.all(util.rms(efn) > 0.7)
    assert np.all(util.rms(efn) < 1.3)
    assert np.all(util.rms(eon) > 0.7)
    assert np.all(util.rms(eon) < 1.3)

import numpy as np
from nlks import examples, run_ekf, run_kalman_smoother, run_optimization, util


def test_kalman_smoother():
    x0, P0, xt, wt, F, G, Q, measurements = examples.generate_linear_pendulum()
    result = run_kalman_smoother(x0, P0, F, G, Q, measurements)

    efn = (result.xf - xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.x - xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5

    assert np.all(util.rms(result.x - xt) < util.rms(result.xf - xt))
    assert np.all(util.rms(efn) > 0.7)
    assert np.all(util.rms(efn) < 1.3)
    assert np.all(util.rms(eon) > 0.7)
    assert np.all(util.rms(eon) < 1.3)


def test_ekf():
    X0, P0, Xt, _, f, Q, measurements = examples.generate_nonlinear_pendulum()
    Xf, Pf = run_ekf(X0, P0, f, Q, measurements)
    en = (Xf - Xt) / np.diagonal(Pf, axis1=1, axis2=2) ** 0.5
    assert np.all(util.rms(en) > 0.7)
    assert np.all(util.rms(en) < 1.3)


def test_optimization():
    X0, P0, Xt, _, f, Q, measurements = examples.generate_nonlinear_pendulum()
    result = run_optimization(X0, P0, f, Q, measurements)

    efn = (result.Xf - Xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.X - Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5

    assert np.all(util.rms(result.X - Xt) < util.rms(result.Xf - Xt))
    assert np.all(util.rms(efn) > 0.7)
    assert np.all(util.rms(efn) < 1.3)
    assert np.all(util.rms(eon) > 0.7)
    assert np.all(util.rms(eon) < 1.3)

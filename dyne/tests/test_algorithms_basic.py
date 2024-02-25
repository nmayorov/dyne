import numpy as np
import dyne


def test_kalman_smoother():
    x0, P0, xt, wt, F, G, Q, measurements = dyne.examples.generate_linear_pendulum()
    result = dyne.run_kalman_smoother(x0, P0, F, G, Q, measurements)

    efn = (result.xf - xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.x - xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5

    assert np.all(dyne.util.rms(result.x - xt) < dyne.util.rms(result.xf - xt))
    assert np.all(dyne.util.rms(efn) > 0.7)
    assert np.all(dyne.util.rms(efn) < 1.3)
    assert np.all(dyne.util.rms(eon) > 0.7)
    assert np.all(dyne.util.rms(eon) < 1.3)


def test_ekf():
    X0, P0, Xt, _, f, Q, measurements = dyne.examples.generate_nonlinear_pendulum()
    result = dyne.run_ekf(X0, P0, f, Q, measurements)
    en = (result.X - Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5
    assert np.all(dyne.util.rms(en) > 0.7)
    assert np.all(dyne.util.rms(en) < 1.3)


def test_ukf():
    X0, P0, Xt, _, f, Q, measurements = dyne.examples.generate_nonlinear_pendulum()
    for alpha in [1e-2, 1e-1, 1, 2]:
        result = dyne.run_ukf(X0, P0, f, Q, measurements, alpha=alpha)
        en = (result.X - Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5
        assert np.all(dyne.util.rms(en) > 0.7)
        assert np.all(dyne.util.rms(en) < 1.3)


def test_optimization():
    X0, P0, Xt, _, f, Q, measurements = dyne.examples.generate_nonlinear_pendulum()
    result = dyne.run_optimization(X0, P0, f, Q, measurements)

    efn = (result.Xf - Xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.X - Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5

    assert np.all(dyne.util.rms(result.X - Xt) < dyne.util.rms(result.Xf - Xt))
    assert np.all(dyne.util.rms(efn) > 0.7)
    assert np.all(dyne.util.rms(efn) < 1.3)
    assert np.all(dyne.util.rms(eon) > 0.7)
    assert np.all(dyne.util.rms(eon) < 1.3)

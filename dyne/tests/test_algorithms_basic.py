import numpy as np
import dyne


def test_kalman_smoother():
    x0, P0, xt, wt, F, G, Q, n_epochs, measurements = (
        dyne.examples.generate_linear_pendulum())
    result = dyne.run_kalman_smoother(x0, P0, F, G, Q, n_epochs, measurements)

    efn = (result.xf - xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.x - xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5

    assert np.all(dyne.util.compute_rms(result.x - xt) <
                  dyne.util.compute_rms(result.xf - xt))
    assert np.all(dyne.util.compute_rms(efn) > 0.7)
    assert np.all(dyne.util.compute_rms(efn) < 1.3)
    assert np.all(dyne.util.compute_rms(eon) > 0.7)
    assert np.all(dyne.util.compute_rms(eon) < 1.3)

    for k in range(n_epochs - 1):
        error = F @ result.x[k] + G @ result.w[k] - result.x[k + 1]
        assert np.all(np.abs(error) < 1e-15)


def test_ekf():
    X0, P0, Xt, _, f, Q, n_epochs, measurements = (
        dyne.examples.generate_nonlinear_pendulum())
    result = dyne.run_ekf(X0, P0, f, Q, n_epochs, measurements)
    en = (result.X - Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5
    assert np.all(dyne.util.compute_rms(en) > 0.7)
    assert np.all(dyne.util.compute_rms(en) < 1.3)


def test_ukf():
    X0, P0, Xt, _, f, Q, n_epochs, measurements = (
        dyne.examples.generate_nonlinear_pendulum())
    for alpha in [1e-2, 1e-1, 1, 2]:
        result = dyne.run_ukf(X0, P0, f, Q, n_epochs, measurements, alpha=alpha)
        en = (result.X - Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5
        assert np.all(dyne.util.compute_rms(en) > 0.7)
        assert np.all(dyne.util.compute_rms(en) < 1.3)


def test_optimization():
    X0, P0, Xt, _, f, Q, n_epochs, measurements = (
        dyne.examples.generate_nonlinear_pendulum())
    result = dyne.run_optimization(X0, P0, f, Q, n_epochs, measurements)

    efn = (result.Xf - Xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.X - Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5

    assert np.all(dyne.util.compute_rms(result.X - Xt) < dyne.util.compute_rms(result.Xf - Xt))
    assert np.all(dyne.util.compute_rms(efn) > 0.7)
    assert np.all(dyne.util.compute_rms(efn) < 1.3)
    assert np.all(dyne.util.compute_rms(eon) > 0.7)
    assert np.all(dyne.util.compute_rms(eon) < 1.3)


def test_mhf():
    X0, P0, Xt, _, f, Q, n_epochs, measurements = (
        dyne.examples.generate_nonlinear_pendulum())

    for window in [1, 3, 5]:
        result = dyne.run_mhf(X0, P0, f, Q, n_epochs, measurements, window)
        en = (result.X - Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5
        assert np.all(dyne.util.compute_rms(en) > 0.7)
        assert np.all(dyne.util.compute_rms(en) < 1.3)

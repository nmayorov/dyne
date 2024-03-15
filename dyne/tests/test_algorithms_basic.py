import numpy as np
import dyne


def test_kalman_smoother():
    p = dyne.examples.generate_linear_pendulum()
    result = dyne.run_kalman_smoother(p.x0, p.P0, p.F, p.G, p.Q, p.n_epochs,
                                      p.measurements)

    efn = (result.xf - p.xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.x - p.xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5

    assert np.all(dyne.util.compute_rms(result.x - p.xt) <
                  dyne.util.compute_rms(result.xf - p.xt))
    assert np.all(dyne.util.compute_rms(efn) > 0.7)
    assert np.all(dyne.util.compute_rms(efn) < 1.3)
    assert np.all(dyne.util.compute_rms(eon) > 0.7)
    assert np.all(dyne.util.compute_rms(eon) < 1.3)

    for k in range(p.n_epochs - 1):
        error = p.F @ result.x[k] + p.G @ result.w[k] - result.x[k + 1]
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


def test_linear_equivalence():
    p_lin = dyne.examples.generate_linear_pendulum()
    *_, f, _, _, measurements = dyne.examples.generate_linear_pendulum_as_nl_problem()

    kf_result = dyne.run_kalman_smoother(p_lin.x0, p_lin.P0, p_lin.F, p_lin.G, p_lin.Q,
                                         p_lin.n_epochs, p_lin.measurements)

    for algorithm, options in zip([dyne.run_ekf, dyne.run_ukf, dyne.run_mhf],
                                  [{}, {}, dict(window=1)]):
        result = algorithm(p_lin.x0, p_lin.P0, f, p_lin.Q, p_lin.n_epochs,
                           measurements, **options)
        difference = result.X - kf_result.xf
        assert np.all(np.abs(difference) < 1e-15)

    result_opt = dyne.run_optimization(p_lin.x0, p_lin.P0, f, p_lin.Q, p_lin.n_epochs,
                                       measurements)
    difference = result_opt.X - kf_result.x
    assert np.all(np.abs(difference) < 1e-15)

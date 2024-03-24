import numpy as np
import dyne


def test_kalman_smoother():
    p = dyne.examples.generate_linear_pendulum()
    result = dyne.run_kalman_smoother(p.x0, p.P0, p.F, p.G, p.Q, p.measurements,
                                      p.n_epochs)

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
    p = dyne.examples.generate_nonlinear_pendulum()
    result = dyne.run_ekf(p.X0, p.P0, p.f, p.Q, p.measurements, p.n_epochs)
    en = (result.X - p.Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5
    assert np.all(dyne.util.compute_rms(en) > 0.7)
    assert np.all(dyne.util.compute_rms(en) < 1.3)


def test_ukf():
    p = dyne.examples.generate_nonlinear_pendulum()
    for alpha in [1e-2, 1e-1, 1, 2]:
        result = dyne.run_ukf(p.X0, p.P0, p.f, p.Q, p.measurements, p.n_epochs,
                              alpha=alpha)
        en = (result.X - p.Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5
        assert np.all(dyne.util.compute_rms(en) > 0.7)
        assert np.all(dyne.util.compute_rms(en) < 1.3)


def test_optimization():
    p = dyne.examples.generate_nonlinear_pendulum()
    result = dyne.run_optimization(p.X0, p.P0, p.f, p.Q, p.measurements, p.n_epochs)

    efn = (result.Xf - p.Xt) / np.diagonal(result.Pf, axis1=1, axis2=2) ** 0.5
    eon = (result.X - p.Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5

    assert np.all(dyne.util.compute_rms(result.X - p.Xt) <
                  dyne.util.compute_rms(result.Xf - p.Xt))
    assert np.all(dyne.util.compute_rms(efn) > 0.7)
    assert np.all(dyne.util.compute_rms(efn) < 1.3)
    assert np.all(dyne.util.compute_rms(eon) > 0.7)
    assert np.all(dyne.util.compute_rms(eon) < 1.3)


def test_mhf():
    p = dyne.examples.generate_nonlinear_pendulum()

    for window in [1, 3, 5]:
        result = dyne.run_mhf(p.X0, p.P0, p.f, p.Q, p.measurements, p.n_epochs, window)
        en = (result.X - p.Xt) / np.diagonal(result.P, axis1=1, axis2=2) ** 0.5
        assert np.all(dyne.util.compute_rms(en) > 0.7)
        assert np.all(dyne.util.compute_rms(en) < 1.3)


def test_linear_equivalence():
    p_lin = dyne.examples.generate_linear_pendulum()
    p_nl = dyne.examples.generate_linear_pendulum_as_nl_problem()

    kf_result = dyne.run_kalman_smoother(p_lin.x0, p_lin.P0, p_lin.F, p_lin.G, p_lin.Q,
                                         p_lin.measurements, p_lin.n_epochs)

    for algorithm, options in zip([dyne.run_ekf, dyne.run_ukf, dyne.run_mhf],
                                  [{}, {}, dict(window=1)]):
        result = algorithm(p_nl.X0, p_nl.P0, p_nl.f, p_nl.Q, p_nl.measurements,
                           p_nl.n_epochs, **options)
        difference = result.X - kf_result.xf
        assert np.all(np.abs(difference) < 1e-15)

    result_opt = dyne.run_optimization(p_nl.X0, p_nl.P0, p_nl.f, p_nl.Q,
                                       p_nl.measurements, p_nl.n_epochs)
    difference = result_opt.X - kf_result.x
    assert np.all(np.abs(difference) < 1e-15)

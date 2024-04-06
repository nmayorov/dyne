import numpy as np
from numpy.testing import assert_allclose
from dyne import ukf
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag


def test_generate_sigma_points():
    R = Rotation.random().as_matrix()
    eigenvalues = np.array([0.1, 1.0, 2.0])
    P1 = R @ np.diag(eigenvalues) @ R.transpose()
    P1_expected = P1

    eigenvalues[0] = -0.01
    P2 = R @ np.diag(eigenvalues) @ R.transpose()

    eigenvalues[0] = 0.0
    P2_expected = R @ np.diag(eigenvalues) @ R.transpose()

    for P, P_expected in [(P1, P1_expected), (P2, P2_expected)]:
        for P_second in [None, np.empty((0, 0)), np.eye(2)]:
            if P_second is None:
                P_blocks = [P]
                P_expected_full = P_expected
            else:
                P_blocks = [P, P_second]
                P_expected_full = block_diag(P_expected, P_second)

            for alpha in [0.1, 1.0, 2.0]:
                sigma_points, w_mean, w_cov = ukf.generate_sigma_points(P_blocks, alpha)
                assert_allclose(len(sigma_points) * w_mean, 1.0, rtol=1e-16)
                assert_allclose(w_mean * np.sum(sigma_points, axis=0), 0, atol=1e-15)
                assert_allclose(w_cov * sigma_points.T @ sigma_points, P_expected_full,
                                rtol=1e-13)

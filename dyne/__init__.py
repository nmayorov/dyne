"""dyne: Optimal estimation of dynamic systems.

The package contains estimation algorithms for discrete-time stochastic dynamics
systems with measurements of the form::

    X_{k + 1} = f_k(X_k, W_k)
    Z_k = h_k(X_k) + V_k

Where

    - k   - integer epoch index
    - X_k - state vector
    - W_k - process noise vector
    - Z_k - measurement vector
    - V_k - measurement noise vector
    - f_k - process function
    - h_k - measurement function

Used references on estimation theory in general and basic known algorithms are [1] and
[2]. References to specific algorithms and nuances are given in individual docstrings if
necessary.

To run the estimation algorithms, process and measurement functions must be implemented
in a specific way according to `dyne.util.process_callable` and
`dyne.util.measurement_callable` signatures. The structure containing measurements must
be carefully prepared according to docstrings. Refer to `dyne.examples` for examples
of correctly defined problems.

References
----------
.. [1] J. L. Crassidis, J. L. Junkins, "Optimal Estimation of Dynamic Systems",
   2nd edition
.. [2] P. S. Maybeck, "Stochastic Models, Estimation and Control", volumes 1 and 2
"""
from . import examples, util
from .linear import run_kalman_smoother
from .ekf import run_ekf
from .optimize import run_optimization, run_mhf
from .ukf import run_ukf

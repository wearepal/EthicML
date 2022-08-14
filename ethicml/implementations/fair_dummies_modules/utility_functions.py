"""Fair Dummies utility functions."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import KernelDensity


def density_estimation(
    y: np.ndarray, *, a: np.ndarray, y_test: np.ndarray | None = None
) -> tuple[list[float], list[float]]:
    """Estimate the distribusion of P{A|Y}."""
    if y_test is None:
        y_test = np.array([])
    assert y_test is not None
    bandwidth = np.sqrt(max(np.median(np.abs(y)), 0.01))

    kde_0 = KernelDensity(kernel='linear', bandwidth=bandwidth).fit(y[a == 0][:, np.newaxis])
    kde_1 = KernelDensity(kernel='linear', bandwidth=bandwidth).fit(y[a == 1][:, np.newaxis])

    log_dens_0 = np.exp(np.squeeze(kde_0.score_samples(y[:, np.newaxis])))
    log_dens_1 = np.exp(np.squeeze(kde_1.score_samples(y[:, np.newaxis])))
    p_0 = np.sum(a == 0) / a.shape[0]
    p_1 = 1 - p_0

    # p(A=1|y) = p(y|A=1)p(A=1) / (p(y|A=1)p(A=1) + p(y|A=0)p(A=0))
    p_success = (log_dens_1 * p_1) / (log_dens_1 * p_1 + log_dens_0 * p_0 + 1e-10)

    p_success_test = []
    if len(y_test) > 0:
        log_dens_0_test = np.exp(np.squeeze(kde_0.score_samples(y_test[:, np.newaxis])))
        log_dens_1_test = np.exp(np.squeeze(kde_1.score_samples(y_test[:, np.newaxis])))
        p_success_test = (log_dens_1_test * p_1) / (
            log_dens_1_test * p_1 + log_dens_0_test * p_0 + 1e-10
        )

    return p_success, p_success_test

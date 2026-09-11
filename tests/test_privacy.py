import math

from seal.privacy import calibrate_sigma, spent_epsilon, rho_to_epsilon, epsilon_to_rho


def test_epsilon_rho_roundtrip():
    for eps in [0.05, 0.1, 0.5, 1.0, 3.0]:
        rho = epsilon_to_rho(eps, 1e-5)
        eps_back = rho_to_epsilon(rho, 1e-5)
        assert math.isclose(eps, eps_back, rel_tol=1e-6)


def test_calibrate_sigma_spends_target_epsilon():
    budget = calibrate_sigma(epsilon=0.5, delta=1e-5, n_queries=50, sensitivity=8.0)
    spent = spent_epsilon(budget.sigma, n_queries=50, sensitivity=8.0, delta=1e-5)
    assert math.isclose(spent, 0.5, rel_tol=1e-6)


def test_more_queries_need_more_noise_for_same_budget():
    b1 = calibrate_sigma(epsilon=1.0, delta=1e-5, n_queries=10, sensitivity=8.0)
    b2 = calibrate_sigma(epsilon=1.0, delta=1e-5, n_queries=100, sensitivity=8.0)
    assert b2.sigma > b1.sigma


def test_larger_epsilon_needs_less_noise():
    b1 = calibrate_sigma(epsilon=0.1, delta=1e-5, n_queries=50, sensitivity=8.0)
    b2 = calibrate_sigma(epsilon=2.0, delta=1e-5, n_queries=50, sensitivity=8.0)
    assert b2.sigma < b1.sigma


def test_zero_epsilon_is_infinite_noise():
    budget = calibrate_sigma(epsilon=0.0, delta=1e-5, n_queries=50, sensitivity=8.0)
    assert math.isinf(budget.sigma)

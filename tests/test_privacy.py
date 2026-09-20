# import numpy as np
# import pytest

# from fedpriv.privacy.accountant import PrivacyAccountant
# from fedpriv.privacy.clipping import clip_update
# from fedpriv.privacy.mechanisms import gaussian_mechanism, laplace_mechanism


# def test_clip_update_no_op_when_under_norm():
#     v = np.array([0.1, 0.1, 0.1])
#     clipped, norm = clip_update(v, clip_norm=1.0)
#     assert np.allclose(clipped, v)


# def test_clip_update_scales_down_when_over_norm():
#     v = np.array([3.0, 4.0])  # norm = 5
#     clipped, norm = clip_update(v, clip_norm=1.0)
#     assert norm == pytest.approx(5.0)
#     assert np.linalg.norm(clipped) == pytest.approx(1.0)


# def test_gaussian_mechanism_adds_noise_with_correct_scale():
#     rng = np.random.default_rng(0)
#     v = np.zeros(200_000)
#     noisy = gaussian_mechanism(v, sigma=2.0, rng=rng)
#     assert noisy.std() == pytest.approx(2.0, rel=0.05)


# def test_laplace_mechanism_adds_noise_with_correct_scale():
#     rng = np.random.default_rng(0)
#     v = np.zeros(200_000)
#     noisy = laplace_mechanism(v, scale=3.0, rng=rng)
#     # Laplace(0, b) has variance 2*b^2
#     assert noisy.std() == pytest.approx(3.0 * np.sqrt(2), rel=0.05)


# @pytest.mark.parametrize("accountant_type", ["basic_composition", "analytic_gaussian"])
# def test_accountant_smaller_epsilon_means_more_noise(accountant_type):
#     acc_tight = PrivacyAccountant("gaussian", epsilon=0.1, delta=1e-5, clip_norm=1.0,
#                                    total_rounds=10, accountant=accountant_type)
#     acc_loose = PrivacyAccountant("gaussian", epsilon=5.0, delta=1e-5, clip_norm=1.0,
#                                    total_rounds=10, accountant=accountant_type)
#     assert acc_tight.noise_param > acc_loose.noise_param


# def test_analytic_gaussian_is_tighter_than_basic_composition():
#     acc_basic = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
#                                    total_rounds=20, accountant="basic_composition")
#     acc_zcdp = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
#                                   total_rounds=20, accountant="analytic_gaussian")
#     # zCDP composition should require LESS noise for the same end-to-end budget
#     assert acc_zcdp.noise_param < acc_basic.noise_param


# def test_privacy_spend_grows_monotonically_with_round():
#     acc = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
#                              total_rounds=10, accountant="analytic_gaussian")
#     spends = [acc.spend_at_round(r).epsilon for r in range(1, 11)]
#     assert all(spends[i] <= spends[i + 1] + 1e-9 for i in range(len(spends) - 1))
#     assert spends[-1] == pytest.approx(1.0, rel=1e-6)


# def test_laplace_scale_inversely_proportional_to_epsilon():
#     acc = PrivacyAccountant("laplace", epsilon=2.0, delta=0, clip_norm=1.0,
#                              total_rounds=1, accountant="basic_composition")
#     assert acc.noise_param == pytest.approx(0.5)




import numpy as np
import pytest

from fedpriv.privacy.accountant import PrivacyAccountant
from fedpriv.privacy.clipping import clip_update
from fedpriv.privacy.mechanisms import gaussian_mechanism, laplace_mechanism


def test_clip_update_no_op_when_under_norm():
    v = np.array([0.1, 0.1, 0.1])
    clipped, norm = clip_update(v, clip_norm=1.0)
    assert np.allclose(clipped, v)


def test_clip_update_scales_down_when_over_norm():
    v = np.array([3.0, 4.0])  # norm = 5
    clipped, norm = clip_update(v, clip_norm=1.0)
    assert norm == pytest.approx(5.0)
    assert np.linalg.norm(clipped) == pytest.approx(1.0)


def test_gaussian_mechanism_adds_noise_with_correct_scale():
    rng = np.random.default_rng(0)
    v = np.zeros(200_000)
    noisy = gaussian_mechanism(v, sigma=2.0, rng=rng)
    assert noisy.std() == pytest.approx(2.0, rel=0.05)


def test_laplace_mechanism_adds_noise_with_correct_scale():
    rng = np.random.default_rng(0)
    v = np.zeros(200_000)
    noisy = laplace_mechanism(v, scale=3.0, rng=rng)
    # Laplace(0, b) has variance 2*b^2
    assert noisy.std() == pytest.approx(3.0 * np.sqrt(2), rel=0.05)


@pytest.mark.parametrize("accountant_type", ["basic_composition", "analytic_gaussian"])
def test_accountant_smaller_epsilon_means_more_noise(accountant_type):
    acc_tight = PrivacyAccountant("gaussian", epsilon=0.1, delta=1e-5, clip_norm=1.0,
                                   total_rounds=10, accountant=accountant_type)
    acc_loose = PrivacyAccountant("gaussian", epsilon=5.0, delta=1e-5, clip_norm=1.0,
                                   total_rounds=10, accountant=accountant_type)
    assert acc_tight.noise_param > acc_loose.noise_param


def test_analytic_gaussian_is_tighter_than_basic_composition():
    acc_basic = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                                   total_rounds=20, accountant="basic_composition")
    acc_zcdp = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                                  total_rounds=20, accountant="analytic_gaussian")
    # zCDP composition should require LESS noise for the same end-to-end budget
    assert acc_zcdp.noise_param < acc_basic.noise_param


def test_privacy_spend_grows_monotonically_with_round():
    acc = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                             total_rounds=10, accountant="analytic_gaussian")
    spends = [acc.spend_at_round(r).epsilon for r in range(1, 11)]
    assert all(spends[i] <= spends[i + 1] + 1e-9 for i in range(len(spends) - 1))
    assert spends[-1] == pytest.approx(1.0, rel=1e-6)


def test_laplace_scale_inversely_proportional_to_epsilon():
    acc = PrivacyAccountant("laplace", epsilon=2.0, delta=0, clip_norm=1.0,
                             total_rounds=1, accountant="basic_composition")
    assert acc.noise_param == pytest.approx(0.5)


def test_default_sampling_rate_reproduces_old_behavior():
    """sampling_rate defaults to 1.0 (no amplification) so every existing
    result computed before this feature existed stays exactly reproducible."""
    acc_explicit = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                                      total_rounds=10, accountant="analytic_gaussian",
                                      sampling_rate=1.0)
    acc_default = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                                     total_rounds=10, accountant="analytic_gaussian")
    assert acc_default.noise_param == pytest.approx(acc_explicit.noise_param)


def test_amplification_reduces_noise_for_analytic_gaussian():
    acc_full = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                                  total_rounds=10, accountant="analytic_gaussian", sampling_rate=1.0)
    acc_subsampled = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                                        total_rounds=10, accountant="analytic_gaussian", sampling_rate=0.3)
    assert acc_subsampled.noise_param < acc_full.noise_param
    assert acc_subsampled.noise_param == pytest.approx(0.3 * acc_full.noise_param)


def test_amplification_reduces_noise_for_laplace():
    acc_full = PrivacyAccountant("laplace", epsilon=1.0, delta=0, clip_norm=1.0,
                                  total_rounds=10, accountant="basic_composition", sampling_rate=1.0)
    acc_subsampled = PrivacyAccountant("laplace", epsilon=1.0, delta=0, clip_norm=1.0,
                                        total_rounds=10, accountant="basic_composition", sampling_rate=0.3)
    assert acc_subsampled.noise_param == pytest.approx(0.3 * acc_full.noise_param)


def test_amplification_does_not_affect_basic_composition_gaussian():
    """basic_composition is the deliberately-naive baseline for the
    accountant-tightness ablation; amplification is intentionally NOT
    applied to it so that comparison stays clean."""
    acc_full = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                                  total_rounds=10, accountant="basic_composition", sampling_rate=1.0)
    acc_subsampled = PrivacyAccountant("gaussian", epsilon=1.0, delta=1e-5, clip_norm=1.0,
                                        total_rounds=10, accountant="basic_composition", sampling_rate=0.3)
    assert acc_subsampled.noise_param == pytest.approx(acc_full.noise_param)

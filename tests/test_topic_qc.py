from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pycisTopic.topic_qc import _proportion_ztest_larger, gini_coefficient


@pytest.mark.parametrize(
    ("count", "nobs", "value"), [(30, 100, 0.2), (5, 100, 0.2), (50, 60, 0.5)]
)
def test_proportion_ztest_larger(count, nobs, value):
    # statsmodels.stats.proportion.proportions_ztest(alternative="larger") uses the
    # observed proportion to estimate the variance.
    prop = count / nobs
    z = (prop - value) / np.sqrt(prop * (1 - prop) / nobs)
    assert _proportion_ztest_larger(count, nobs, value) == pytest.approx(norm.sf(z))


@pytest.mark.parametrize(("count", "expected"), [(0, 1.0), (10, 0.0)])
def test_proportion_ztest_larger_zero_variance(count, expected):
    assert _proportion_ztest_larger(count, 10, 0.5) == expected


def test_gini_coefficient():
    assert gini_coefficient(np.ones(10)) == pytest.approx(0.0)
    x = np.zeros(100)
    x[0] = 1
    assert gini_coefficient(x) == pytest.approx(0.99)

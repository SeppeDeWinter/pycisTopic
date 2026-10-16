from __future__ import annotations

import numpy as np
import pytest

from pycisTopic.topic_binarization import (
    _gaussian_mixture_1d_means,
    binarize_topics,
    smooth_topics_distributions,
    threshold_aucell,
    threshold_li,
    threshold_ntop,
    threshold_otsu,
    threshold_yen,
)


@pytest.fixture
def bimodal(rng):
    return np.concatenate([rng.normal(0.2, 0.03, 900), rng.normal(0.8, 0.03, 100)])


def _between_class_variances(values, nbins):
    """Between-class variance for each split of the histogram (class 1 = bins <= k)."""
    hist, edges = np.histogram(values, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    variances = {}
    for k in range(nbins - 1):
        w1, w2 = hist[: k + 1].sum(), hist[k + 1 :].sum()
        if w1 == 0 or w2 == 0:
            continue
        m1 = (hist[: k + 1] * centers[: k + 1]).sum() / w1
        m2 = (hist[k + 1 :] * centers[k + 1 :]).sum() / w2
        variances[centers[k]] = w1 * w2 * (m1 - m2) ** 2
    return variances


@pytest.mark.parametrize("nbins", [20, 100, 5000])
@pytest.mark.parametrize("high_mode_mean", [0.8, 0.35])
def test_threshold_otsu_maximizes_between_class_variance(rng, nbins, high_mode_mean):
    values = np.concatenate(
        [rng.normal(0.2, 0.05, 900), rng.normal(high_mode_mean, 0.05, 100)]
    )
    variances = _between_class_variances(values, nbins)
    threshold = threshold_otsu(values, nbins=nbins)
    # Compare variances instead of thresholds: splits in empty bins are ties.
    best = max(variances.values())
    achieved = variances[min(variances, key=lambda c: abs(c - threshold))]
    assert achieved == pytest.approx(best, rel=1e-9)


@pytest.mark.parametrize("threshold_function", [threshold_otsu, threshold_yen, threshold_li])
def test_threshold_keeps_high_mode(bimodal, threshold_function):
    threshold = threshold_function(bimodal, nbins=100)
    assert bimodal.min() < threshold < 0.7
    # All values of the (small) high mode are above the threshold.
    assert np.all(bimodal[900:] > threshold)


def test_threshold_ntop(rng):
    values = rng.random(50)
    threshold = threshold_ntop(values, ntop=5)
    assert (values > threshold).sum() == 5


def test_smooth_topics_distributions():
    x = np.array([[0.5, 0.5], [0.9, 0.1]])
    smoothed = smooth_topics_distributions(x)
    # Uniform rows are smoothed to 0; specific entries are rewarded.
    np.testing.assert_allclose(smoothed[0], 0, atol=1e-12)
    expected = x[1] * (np.log(x[1]) - np.log(x[1]).mean())
    np.testing.assert_allclose(smoothed[1], expected)


class TestBinarizeTopics:
    @pytest.fixture
    def topic_prob(self, rng):
        prob = rng.dirichlet(np.full(4, 0.2), size=200)
        names = [f"region{i}" for i in range(200)]
        return prob, names

    @pytest.mark.parametrize("method", ["otsu", "yen", "li", "aucell"])
    def test_output_structure(self, topic_prob, method):
        prob, names = topic_prob
        selected, scores, thresholds = binarize_topics(prob, names, method=method)
        assert len(selected) == len(scores) == len(thresholds) == 4
        for topic_names, topic_scores in zip(selected, scores):
            assert len(topic_names) == len(topic_scores) > 0
            # Sorted from highest to lowest score.
            assert np.all(np.diff(topic_scores) <= 0)

    def test_ntop(self, topic_prob):
        prob, names = topic_prob
        selected, scores, _ = binarize_topics(
            prob, names, method="ntop", ntop=10, smooth_topics=False
        )
        for topic_idx, topic_names in enumerate(selected):
            expected = [names[i] for i in np.argsort(prob[:, topic_idx])[::-1][:10]]
            assert topic_names == expected
            np.testing.assert_allclose(scores[topic_idx], np.sort(prob[:, topic_idx])[::-1][:10])

    def test_errors(self, topic_prob):
        prob, names = topic_prob
        with pytest.raises(ValueError, match="rows"):
            binarize_topics(prob, names[:-1])
        with pytest.raises(ValueError, match="duplicates"):
            binarize_topics(prob, names[:-1] + names[:1])
        with pytest.raises(ValueError, match="ntop"):
            binarize_topics(prob, names, method="ntop")
        with pytest.raises(ValueError, match="method"):
            binarize_topics(prob, names, method="median")


class TestThresholdAucell:
    def test_unimodal_is_mean_plus_two_std(self, rng):
        values = rng.normal(0.3, 0.05, 2000)
        assert threshold_aucell(values) == pytest.approx(
            values.mean() + 2 * values.std()
        )

    # With a small high mode (e.g. 5%), the density between the modes is flat with
    # several local minima and the bounded Brent search (as in pySCENIC) does not
    # always end in a minimum, so only clear troughs are tested here.
    @pytest.mark.parametrize("fraction", [0.2, 0.5])
    def test_bimodal_is_density_minimum_between_modes(self, rng, fraction):
        from scipy.stats import gaussian_kde

        n_high = int(2000 * fraction)
        values = np.concatenate(
            [rng.normal(0.2, 0.06, 2000 - n_high), rng.normal(0.7, 0.08, n_high)]
        )
        threshold = threshold_aucell(values)
        assert 0.3 < threshold < 0.6
        # The threshold is a (local) minimum of the kernel density between both modes.
        kde = gaussian_kde(values)
        assert kde(threshold)[0] <= kde(threshold - 0.005)[0]
        assert kde(threshold)[0] <= kde(threshold + 0.005)[0]

    def test_gaussian_mixture_means(self, rng):
        values = np.concatenate([rng.normal(-2, 0.5, 700), rng.normal(3, 1, 300)])
        means = np.sort(_gaussian_mixture_1d_means(values))
        np.testing.assert_allclose(means, [-2, 3], atol=0.15)

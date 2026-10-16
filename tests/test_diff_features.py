from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from pycisTopic.diff_features import (
    calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc,
    find_diff_accessible_regions,
    find_highly_variable_regions,
    get_log2_fc,
    get_marker_regions_for_contrast,
    get_nonzero_row_indices,
    mean_axis1,
    p_adjust_bh,
    rankdata_average_numba,
    ranksums_numba,
    ranksums_numba_multiple,
    subset_array_second_axis,
)


class TestNumbaHelpers:
    def test_rankdata_average_matches_scipy(self, rng):
        x = rng.integers(0, 10, 200).astype(np.float64)  # many ties
        np.testing.assert_allclose(rankdata_average_numba(x), stats.rankdata(x))

    def test_ranksums_matches_scipy(self, rng):
        x = rng.normal(0.3, 1, 40)
        y = rng.normal(0, 1, 60)
        z, p = ranksums_numba(x, y)
        expected = stats.ranksums(x, y)
        assert z == pytest.approx(expected.statistic)
        assert p == pytest.approx(expected.pvalue)

    def test_ranksums_multiple(self, rng):
        x = rng.integers(0, 50, (5, 30)).astype(np.uint32)
        y = rng.integers(0, 50, (5, 20)).astype(np.uint32)
        z, p = ranksums_numba_multiple(x, y)
        for i in range(5):
            expected = stats.ranksums(x[i], y[i])
            assert z[i] == pytest.approx(expected.statistic)
            assert p[i] == pytest.approx(expected.pvalue)

    def test_ranksums_multiple_shape_errors(self):
        with pytest.raises(ValueError):
            ranksums_numba_multiple(np.zeros((2, 3)), np.zeros((3, 3)))

    def test_p_adjust_bh_matches_scipy(self, rng):
        p = rng.random(500) ** 3
        np.testing.assert_allclose(
            p_adjust_bh(p), stats.false_discovery_control(p, method="bh")
        )

    def test_get_nonzero_row_indices(self):
        x = np.array([[0, 0], [0, 1], [0, 0], [2, 0]], dtype=np.float32)
        np.testing.assert_array_equal(get_nonzero_row_indices(x), [1, 3])

    def test_mean_axis1_and_log2_fc(self, rng):
        fg = rng.random((10, 7))
        bg = rng.random((10, 4))
        np.testing.assert_allclose(mean_axis1(fg), fg.mean(axis=1))
        np.testing.assert_allclose(
            get_log2_fc(fg, bg),
            np.log2((fg.mean(axis=1) + 1e-12) / (bg.mean(axis=1) + 1e-12)),
        )
        with pytest.raises(ValueError, match="different first dimension"):
            get_log2_fc(fg, bg[:5])

    def test_subset_array_second_axis(self, rng):
        x = rng.random((6, 10)).astype(np.float32)
        idx = np.array([9, 0, 3, 3], dtype=np.int64)
        np.testing.assert_array_equal(subset_array_second_axis(x, idx), x[:, idx])


def _reference_mean_and_dispersion(region_topic, cell_topic, region_names, sf1, sf2):
    imputed = np.floor((region_topic @ cell_topic) * np.float32(sf1))
    keep = np.flatnonzero(imputed.any(axis=1))
    imputed = imputed[keep].astype(np.float64)
    per_cell_sum = imputed.sum(axis=0) / sf2
    normalized = np.log1p(imputed / per_cell_sum)
    mean = normalized.mean(axis=1)
    variance = normalized.var(axis=1)
    with np.errstate(divide="ignore"):
        dispersion = np.log(variance / np.where(mean == 0, 1e-12, mean))
    dispersion[variance == 0] = np.nan
    return [region_names[i] for i in keep], mean, dispersion


class TestMeanAndDispersion:
    @pytest.fixture
    def model_with_inaccessible_regions(self, topic_model):
        region_topic, cell_topic, region_names, _ = topic_model
        region_topic = region_topic.copy()
        region_topic[[3, 50, 51]] = 0.0
        return region_topic, cell_topic, region_names

    @pytest.mark.parametrize("chunk_size", [37, 400, 1000])
    def test_matches_reference(self, model_with_inaccessible_regions, chunk_size):
        region_topic, cell_topic, region_names = model_with_inaccessible_regions
        names, mean, dispersion = (
            calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc(
                region_topic, cell_topic, region_names, regions_chunk_size=chunk_size
            )
        )
        ref_names, ref_mean, ref_dispersion = _reference_mean_and_dispersion(
            region_topic, cell_topic, region_names, 10**6, 10**4
        )
        assert names == ref_names
        assert len(names) == len(region_names) - 3
        np.testing.assert_allclose(mean, ref_mean, rtol=1e-4)
        np.testing.assert_allclose(dispersion, ref_dispersion, rtol=1e-4, atol=1e-4)


class TestHighlyVariableRegions:
    @pytest.fixture
    def mean_dispersion(self, topic_model):
        region_topic, cell_topic, region_names, _ = topic_model
        return calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc(
            region_topic, cell_topic, region_names
        )

    def test_n_top_features(self, mean_dispersion):
        names, mean, dispersion = mean_dispersion
        hvr = find_highly_variable_regions(
            names, mean, dispersion, n_top_features=25, plot=False
        )
        assert len(hvr) == 25
        assert set(hvr) <= set(names)

    def test_thresholds(self, mean_dispersion):
        names, mean, dispersion = mean_dispersion
        hvr = find_highly_variable_regions(
            names, mean, dispersion, min_mean=0.0, max_mean=np.inf, plot=False
        )
        selected = np.isin(names, hvr)
        assert 0 < selected.sum() < len(names)
        assert np.all(mean[selected] > 0)


def _reference_markers(region_topic, cell_topic, region_names, cell_names, regions, fg, bg):
    r_idx = [region_names.index(r) for r in regions]
    c_idx = [cell_names.index(c) for c in fg + bg]
    imputed = np.floor(
        (region_topic[r_idx] @ cell_topic[:, c_idx]) * np.float32(10**6)
    )
    fg_acc, bg_acc = imputed[:, : len(fg)], imputed[:, len(fg) :]
    pvalues = np.array([stats.ranksums(a, b).pvalue for a, b in zip(fg_acc, bg_acc)])
    adjusted = stats.false_discovery_control(pvalues, method="bh")
    log2fc = np.log2((fg_acc.mean(axis=1) + 1e-12) / (bg_acc.mean(axis=1) + 1e-12))
    return dict(zip(regions, zip(log2fc, adjusted)))


class TestMarkerRegions:
    @pytest.fixture
    def contrast(self, topic_model):
        region_topic, cell_topic, region_names, cell_names = topic_model
        # Foreground: cells dominated by topic 0.
        fg_mask = cell_topic[0] > np.quantile(cell_topic[0], 0.8)
        fg = [c for c, m in zip(cell_names, fg_mask) if m]
        bg = [c for c, m in zip(cell_names, fg_mask) if not m]
        return region_topic, cell_topic, region_names, cell_names, region_names, fg, bg

    def _markers(self, contrast, **kwargs):
        region_topic, cell_topic, region_names, cell_names, regions, fg, bg = contrast
        return get_marker_regions_for_contrast(
            region_topic,
            cell_topic,
            region_names,
            cell_names,
            highly_variable_regions=regions,
            contrast_name="topic0",
            selected_foreground_cells=fg,
            selected_background_cells=bg,
            **kwargs,
        )

    def test_matches_reference_in_single_chunk(self, contrast):
        markers = self._markers(contrast, regions_chunk_size=10_000)
        reference = _reference_markers(*contrast)
        assert markers.height > 0
        assert set(markers["Contrast"]) == {"topic0"}
        for region, log2fc, adjusted_pval in markers.select(
            "RegionNames", "Log2FC", "Adjusted_pval"
        ).rows():
            assert log2fc == pytest.approx(reference[region][0], rel=1e-5)
            assert adjusted_pval == pytest.approx(reference[region][1], rel=1e-5)
            assert log2fc >= math.log2(1.5)
            assert adjusted_pval <= 0.05
        expected = {
            r for r, (fc, p) in reference.items() if fc >= math.log2(1.5) and p <= 0.05
        }
        assert set(markers["RegionNames"]) == expected
        # Sorted by decreasing log2 fold change.
        assert markers["Log2FC"].is_sorted(descending=True)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: Benjamini-Hochberg correction is applied per chunk of regions "
        "instead of over all tested regions.",
    )
    def test_results_do_not_depend_on_chunk_size(self, contrast):
        assert self._markers(contrast, regions_chunk_size=10_000).equals(
            self._markers(contrast, regions_chunk_size=50)
        )

    def test_never_accessible_region(self, contrast):
        region_topic, *rest = contrast
        region_topic = region_topic.copy()
        region_topic[0] = 0
        with pytest.raises(ValueError, match="never accessible"):
            self._markers((region_topic, *rest))

    def test_find_diff_accessible_regions(self, contrast):
        region_topic, cell_topic, region_names, cell_names, regions, fg, bg = contrast
        markers = find_diff_accessible_regions(
            region_topic,
            cell_topic,
            region_names,
            cell_names,
            regions,
            contrasts={"fg_vs_bg": (fg, bg), "bg_vs_fg": (bg, fg)},
        )
        assert set(markers) == {"fg_vs_bg", "bg_vs_fg"}
        assert not set(markers["fg_vs_bg"]["RegionNames"]) & set(
            markers["bg_vs_fg"]["RegionNames"]
        )

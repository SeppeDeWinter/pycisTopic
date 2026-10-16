from __future__ import annotations

import numpy as np
import pytest

from pycisTopic.imputed_accessibility import (
    impute_accessibility_chunked,
    rank_imputed_accessibility,
)


class TestImputeAccessibilityChunked:
    @pytest.mark.parametrize("chunk_along", ["cell", "region"])
    @pytest.mark.parametrize("chunk_size", [1, 7, 120, 1000])
    def test_chunks_reconstruct_full_matrix(self, topic_model, chunk_along, chunk_size):
        region_topic, cell_topic, _, _ = topic_model
        full = region_topic @ cell_topic
        result = np.empty_like(full)
        for (start, end), chunk in impute_accessibility_chunked(
            region_topic,
            cell_topic,
            chunk_size=chunk_size,
            chunk_along=chunk_along,
            return_start_end=True,
        ):
            if chunk_along == "cell":
                result[:, start:end] = chunk
            else:
                result[start:end] = chunk
        np.testing.assert_allclose(result, full, rtol=1e-6)

    def test_dimension_mismatch(self, topic_model):
        region_topic, cell_topic, _, _ = topic_model
        with pytest.raises(ValueError, match="dimensions do not match"):
            next(impute_accessibility_chunked(region_topic, cell_topic.T, 10, "cell"))

    def test_invalid_chunk_along(self, topic_model):
        region_topic, cell_topic, _, _ = topic_model
        with pytest.raises(ValueError, match="chunk along"):
            next(impute_accessibility_chunked(region_topic, cell_topic, 10, "topic"))


class TestRankImputedAccessibility:
    @pytest.mark.parametrize("engine", ["numpy", "polars"])
    def test_rankings_without_ties(self, topic_model, engine):
        region_topic, cell_topic, _, _ = topic_model
        imputed = region_topic @ cell_topic[:, :20]
        rankings = rank_imputed_accessibility(imputed, seed=1, engine=engine)
        assert rankings.dtype == np.int32
        # Rank 0 is the most accessible region of a cell.
        expected = np.argsort(np.argsort(-imputed, axis=0, kind="stable"), axis=0)
        np.testing.assert_array_equal(rankings, expected)

    @pytest.mark.parametrize("engine", ["numpy", "polars"])
    def test_ties_are_broken_randomly_but_reproducibly(self, engine):
        imputed = np.zeros((100, 3), dtype=np.float32)
        imputed[:5] = 1.0
        rankings = rank_imputed_accessibility(imputed, seed=7, engine=engine)
        for cell_idx in range(3):
            assert sorted(rankings[:, cell_idx]) == list(range(100))
            assert set(rankings[:5, cell_idx]) == set(range(5))
        np.testing.assert_array_equal(
            rankings, rank_imputed_accessibility(imputed, seed=7, engine=engine)
        )
        assert not np.array_equal(
            rankings, rank_imputed_accessibility(imputed, seed=8, engine=engine)
        )

    def test_invalid_engine(self):
        with pytest.raises(ValueError, match="Invalid engine"):
            rank_imputed_accessibility(np.zeros((2, 2), dtype=np.float32), engine="r")

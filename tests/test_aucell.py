from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pycisTopic.aucell import aucell, derive_rank_cutoff
from pycisTopic.signature_enrichment import signature_enrichment


def _reference_aucs(rankings, region_names, signature, auc_threshold):
    """Recovery curve based AUC, as implemented in ctxcore.recovery.weighted_auc1d."""
    n_regions = rankings.shape[0]
    rank_cutoff = round(auc_threshold * n_regions) - 1
    idx = np.flatnonzero(np.isin(region_names, signature))
    max_auc = (rank_cutoff + 1) * len(idx)
    aucs = []
    for cell_idx in range(rankings.shape[1]):
        ranks = rankings[idx, cell_idx]
        x = np.sort(ranks[ranks < rank_cutoff])
        x = np.concatenate([x, [rank_cutoff]])
        y = np.arange(1, len(x))
        aucs.append(np.sum(np.diff(x) * y) / max_auc)
    return np.array(aucs)


@pytest.fixture
def rankings(rng):
    n_regions, n_cells = 2000, 40
    rankings = np.stack([rng.permutation(n_regions) for _ in range(n_cells)], axis=1)
    names = np.array([f"r{i}" for i in range(n_regions)])
    return rankings.astype(np.int32), names


class TestDeriveRankCutoff:
    def test_rounding(self):
        # round(0.05 * 1010) = 50 (banker's rounding of 50.5), minus one.
        assert derive_rank_cutoff(0.05, 1010) == 49
        assert derive_rank_cutoff(0.05, 2000) == 99

    @pytest.mark.parametrize("auc_threshold", [0.0, 1.5])
    def test_invalid_threshold(self, auc_threshold):
        with pytest.raises(ValueError, match="fraction"):
            derive_rank_cutoff(auc_threshold, 100)

    def test_too_few_regions(self):
        with pytest.raises(ValueError, match="top ranked regions"):
            derive_rank_cutoff(0.001, 100)


class TestAucell:
    @pytest.mark.parametrize("auc_threshold", [0.01, 0.05, 0.2])
    def test_matches_recovery_curve_reference(self, rankings, rng, auc_threshold):
        rankings, names = rankings
        signatures = {
            f"sig{i}": list(rng.choice(names, size=size, replace=False))
            for i, size in enumerate([3, 50, 400])
        }
        result = aucell(rankings, names, signatures, auc_threshold=auc_threshold)
        assert result.shape == (rankings.shape[1], 3)
        for sig_idx, signature in enumerate(signatures.values()):
            np.testing.assert_allclose(
                result[:, sig_idx],
                _reference_aucs(rankings, names, signature, auc_threshold),
            )

    def test_perfect_signature_has_auc_close_to_one(self, rankings):
        rankings, names = rankings
        # Signature = top ranked regions of cell 0.
        top = list(names[np.argsort(rankings[:, 0])[:5]])
        assert aucell(rankings, names, {"top": top})[0, 0] > 0.95

    def test_missing_regions(self, rankings):
        rankings, names = rankings
        signatures = {
            "mostly_missing": list(names[:7]) + [f"missing{i}" for i in range(3)],
            "present": list(names[:9]) + ["missing"],
            "all_missing": ["missing"],
        }
        result = aucell(rankings, names, signatures)
        # Less than 80% of the signature present: AUC is 0.
        assert np.all(result[:, 0] == 0)
        assert np.all(result[:, 2] == 0)
        np.testing.assert_allclose(
            result[:, 1], _reference_aucs(rankings, names, list(names[:9]), 0.05)
        )

    def test_duplicate_signature_regions_are_ignored(self, rankings):
        rankings, names = rankings
        signature = list(names[:20])
        np.testing.assert_array_equal(
            aucell(rankings, names, {"s": signature * 2}),
            aucell(rankings, names, {"s": signature}),
        )

    def test_shape_mismatch(self, rankings):
        rankings, names = rankings
        with pytest.raises(ValueError, match="does not match"):
            aucell(rankings, names[:-1], {"s": list(names[:5])})


class TestSignatureEnrichment:
    @pytest.fixture
    def model(self, topic_model):
        region_topic, cell_topic, region_names, _ = topic_model
        granges = pl.DataFrame(
            {
                "Chromosome": [n.split(":")[0] for n in region_names],
                "Start": [int(n.split(":")[1].split("-")[0]) for n in region_names],
                "End": [int(n.split("-")[1]) for n in region_names],
            }
        )
        return region_topic, cell_topic, granges

    def test_matches_aucell_on_full_rankings(self, model):
        from pycisTopic.imputed_accessibility import rank_imputed_accessibility

        region_topic, cell_topic, granges = model
        signature = granges[10:60]
        # Signature regions shifted by 100 bp: still >= 50% overlap with consensus.
        shifted = signature.with_columns(pl.col("Start") + 100, pl.col("End") + 100)
        result = signature_enrichment(
            region_topic,
            cell_topic,
            granges,
            {"sig": shifted},
            chunk_size=cell_topic.shape[1],
            normalize=False,
            min_frac_consensus=0.5,
            min_frac_signature=0.5,
            seed=3,
        )
        rankings = rank_imputed_accessibility(region_topic @ cell_topic, seed=3)
        region_names = [f"chr1:{s}-{e}" for s, e in granges.select("Start", "End").rows()]
        expected = aucell(
            rankings,
            region_names,
            {"sig": region_names[10:60]},
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_normalize_and_chunking(self, model):
        region_topic, cell_topic, granges = model
        kwargs = {
            "signatures": {"a": granges[:30], "b": granges[200:260]},
            "normalize": True,
            "min_frac_consensus": 0.5,
            "min_frac_signature": 0.5,
            "seed": 1,
        }
        result = signature_enrichment(
            region_topic, cell_topic, granges, chunk_size=17, **kwargs
        )
        assert result.shape == (cell_topic.shape[1], 2)
        np.testing.assert_allclose(result.max(axis=0), 1.0)
        # Rankings are generated per chunk, so with ties only the shape is invariant;
        # without ties chunking does not change the results.
        other = signature_enrichment(
            region_topic, cell_topic, granges, chunk_size=cell_topic.shape[1], **kwargs
        )
        np.testing.assert_allclose(result, other, rtol=1e-6)

    def test_region_count_mismatch(self, model):
        region_topic, cell_topic, granges = model
        with pytest.raises(ValueError, match="does not match"):
            signature_enrichment(
                region_topic, cell_topic, granges[:-1], {}, 10, False, 0.5, 0.5, 1
            )

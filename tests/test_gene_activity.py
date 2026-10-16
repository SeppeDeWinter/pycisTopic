from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from pycisTopic.gene_activity import get_gene_activity, get_region_gene_weights

GENE_BODY_WEIGHT = 1 + math.exp(-1)


def _decay_weight(distance: int, decay_rate: float = 1.0) -> float:
    return math.exp(-decay_rate * abs(distance) / 5000) + math.exp(-1)


@pytest.fixture
def annotation():
    # G1 (+ strand) chr1:10000-20000 with 2 transcripts.
    # G2 (- strand) chr1:40000-50000 (TSS at 49999).
    return pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr1"],
            "Gene_start": [10_000, 10_000, 40_000],
            "Gene_end": [20_000, 20_000, 50_000],
            "Gene": ["G1", "G1", "G2"],
            "Strand": ["+", "+", "-"],
            "TSS_start": [10_000, 12_000, 49_999],
        }
    )


@pytest.fixture
def chrom_sizes():
    return pl.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [1_000_000]})


def _regions(*coordinates):
    return pl.DataFrame(
        {
            "Chromosome": ["chr1"] * len(coordinates),
            "Start": [s for s, _ in coordinates],
            "End": [e for _, e in coordinates],
            "Name": [f"chr1:{s}-{e}" for s, e in coordinates],
        }
    )


def _weights_by_link(weights: pl.DataFrame) -> dict[tuple[str, str], dict]:
    return {(row["Name"], row["Gene"]): row for row in weights.iter_rows(named=True)}


class TestRegionGeneWeights:
    def test_distances_without_boundaries(self, annotation, chrom_sizes):
        regions = _regions(
            (6_990, 7_010),  # 3 kb upstream of G1 (midpoint 7000)
            (14_990, 15_010),  # in G1
            (21_990, 22_010),  # 2 kb downstream of G1, 18 kb downstream of G2
            (50_990, 51_010),  # 1 kb upstream of G2 (- strand)
        )
        weights = get_region_gene_weights(
            regions, annotation, chrom_sizes, use_gene_boundaries=False
        )
        links = _weights_by_link(weights)
        expected = {
            ("chr1:6990-7010", "G1"): 3_000,
            ("chr1:14990-15010", "G1"): 0,
            ("chr1:21990-22010", "G1"): -2_000,
            ("chr1:21990-22010", "G2"): -18_000,
            ("chr1:14990-15010", "G2"): -25_000,
            ("chr1:6990-7010", "G2"): -33_000,
            ("chr1:50990-51010", "G2"): 1_000,
            ("chr1:50990-51010", "G1"): -31_000,
        }
        assert {k: v["Distance"] for k, v in links.items()} == expected
        assert links[("chr1:6990-7010", "G1")]["Distance_weight"] == pytest.approx(
            GENE_BODY_WEIGHT
        )
        assert links[("chr1:21990-22010", "G1")]["Distance_weight"] == pytest.approx(
            _decay_weight(-2_000)
        )
        assert set(weights["Gene_size_weight"]) == {1.0}
        assert "Distance_upstream" not in weights.columns

    def test_gene_boundaries_limit_search_space(self, annotation, chrom_sizes):
        regions = _regions((21_990, 22_010), (35_000, 35_020), (60_000, 60_020))
        weights = get_region_gene_weights(regions, annotation, chrom_sizes)
        links = _weights_by_link(weights)
        # G1 search space downstream stops at the promoter of G2 (49989-50009):
        # nearest non-overlapping promoter downstream of G1 end.
        g1 = weights.filter(pl.col("Gene") == "G1").row(0, named=True)
        assert g1["Distance_downstream"] == 49_989 - 20_000 + 1
        # Upstream of G1 there is no other promoter: limited by the chromosome start.
        assert g1["Distance_upstream"] == 10_000 - 1 + 1
        # G2 (- strand): upstream is towards higher coordinates (chromosome end),
        # downstream towards G1's promoters.
        g2 = weights.filter(pl.col("Gene") == "G2").row(0, named=True)
        assert g2["Distance_upstream"] == 100_000
        assert g2["Distance_downstream"] == 40_000 - 12_010 + 1
        assert set(links) == {
            ("chr1:21990-22010", "G1"),
            ("chr1:35000-35020", "G1"),
            ("chr1:21990-22010", "G2"),
            ("chr1:35000-35020", "G2"),
            ("chr1:60000-60020", "G2"),
        }

    def test_minimum_search_space(self, chrom_sizes):
        # Two genes with promoters 100 bp apart: the minimum search space applies.
        annotation = pl.DataFrame(
            {
                "Chromosome": ["chr1", "chr1"],
                "Gene_start": [10_000, 10_150],
                "Gene_end": [10_050, 20_000],
                "Gene": ["A", "B"],
                "Strand": ["+", "+"],
                "TSS_start": [10_000, 10_150],
            }
        )
        weights = get_region_gene_weights(
            _regions((10_500, 10_520)), annotation, chrom_sizes, downstream=(1_000, 100_000)
        )
        a = weights.filter(pl.col("Gene") == "A").row(0, named=True)
        assert a["Distance_downstream"] == 1_000

    def test_predefined_boundaries(self, annotation, chrom_sizes):
        boundaries = pl.DataFrame({"Chromosome": ["chr1"], "Start": [25_000], "End": [25_001]})
        weights = get_region_gene_weights(
            _regions((30_000, 30_020)),
            annotation,
            chrom_sizes,
            predefined_boundaries=boundaries,
        )
        assert weights["Gene"].to_list() == ["G2"]

    def test_remove_promoters(self, annotation, chrom_sizes):
        # Region overlapping the promoter of G2 is removed as a distal region of G1,
        # but kept for G2 itself (distance 0).
        regions = _regions((49_995, 50_005))
        kept = get_region_gene_weights(
            regions, annotation, chrom_sizes, use_gene_boundaries=False
        )
        removed = get_region_gene_weights(
            regions,
            annotation,
            chrom_sizes,
            use_gene_boundaries=False,
            remove_promoters=True,
        )
        assert set(kept["Gene"]) == {"G1", "G2"}
        assert removed["Gene"].to_list() == ["G2"]

    def test_gene_size_weight(self, annotation, chrom_sizes):
        weights = get_region_gene_weights(
            _regions((14_990, 15_010)),
            annotation,
            chrom_sizes,
            use_gene_boundaries=False,
            gene_size_weight=True,
            gene_size_scale_factor=5_000,
        )
        assert weights["Gene_size_weight"].to_list() == [0.5, 0.5]

    def test_distance_weight_disabled(self, annotation, chrom_sizes):
        weights = get_region_gene_weights(
            _regions((21_990, 22_010)),
            annotation,
            chrom_sizes,
            use_gene_boundaries=False,
            distance_weight=False,
        )
        assert set(weights["Distance_weight"]) == {1.0}


class TestGeneActivity:
    @pytest.fixture
    def model(self, rng):
        region_starts = np.arange(0, 120_000, 1_500)
        region_names = [f"chr1:{s}-{s + 500}" for s in region_starts]
        region_topic = rng.dirichlet(np.full(len(region_names), 0.5), size=5).T
        cell_topic = rng.dirichlet(np.full(5, 0.5), size=30).T
        return region_topic.astype(np.float32), cell_topic.astype(np.float32), region_names

    @pytest.mark.parametrize("gini_weight", [True, False])
    @pytest.mark.parametrize("average_scores", [True, False])
    def test_activity_is_weighted_sum_of_imputed_accessibility(
        self, model, annotation, chrom_sizes, gini_weight, average_scores
    ):
        region_topic, cell_topic, region_names = model
        activity, genes, weights = get_gene_activity(
            region_topic,
            cell_topic,
            region_names,
            annotation,
            chrom_sizes,
            gini_weight=gini_weight,
            average_scores=average_scores,
            chunk_size=7,
        )
        assert genes == ["G1", "G2"]
        assert activity.shape == (cell_topic.shape[1], 2)
        imputed = (region_topic @ cell_topic).astype(np.float64)
        for gene_idx, gene in enumerate(genes):
            gene_weights = weights.filter(pl.col("Gene") == gene)
            idx = [region_names.index(n) for n in gene_weights["Name"]]
            expected = gene_weights["Weight"].to_numpy() @ imputed[idx]
            if average_scores:
                expected /= len(idx)
            np.testing.assert_allclose(activity[:, gene_idx], expected, rtol=1e-5)
        if not gini_weight:
            assert set(weights["Gini_weight"]) == {1.0}

    def test_gini_weight(self, model, annotation, chrom_sizes):
        region_topic, cell_topic, region_names = model
        _, _, weights = get_gene_activity(
            region_topic, cell_topic, region_names, annotation, chrom_sizes
        )
        imputed = (region_topic @ cell_topic).astype(np.float64)
        name, gini_weight = weights.select("Name", "Gini_weight").row(0)
        x = np.sort(imputed[region_names.index(name)])
        n = len(x)
        gini = np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum())
        assert gini_weight == pytest.approx(math.exp(1 - gini) + math.exp(-1), rel=1e-5)

    def test_scale_factor_rounds_and_drops_empty_genes(self, model, annotation, chrom_sizes):
        region_topic, cell_topic, region_names = model
        activity, genes, _ = get_gene_activity(
            region_topic, cell_topic, region_names, annotation, chrom_sizes, scale_factor=1e-9
        )
        assert activity.shape == (cell_topic.shape[1], 0)
        assert genes == []
        activity, _, _ = get_gene_activity(
            region_topic, cell_topic, region_names, annotation, chrom_sizes, scale_factor=1e6
        )
        np.testing.assert_array_equal(activity, np.round(activity))

    def test_regions_without_accessibility_are_ignored(self, model, annotation, chrom_sizes):
        region_topic, cell_topic, region_names = model
        region_topic = region_topic.copy()
        region_topic[10] = 0
        _, _, weights = get_gene_activity(
            region_topic, cell_topic, region_names, annotation, chrom_sizes
        )
        assert region_names[10] not in weights["Name"].to_list()

    def test_input_validation(self, model, annotation, chrom_sizes):
        region_topic, cell_topic, region_names = model
        with pytest.raises(ValueError, match="Number of region names"):
            get_gene_activity(region_topic, cell_topic, region_names[:-1], annotation, chrom_sizes)
        with pytest.raises(ValueError, match="dimensions do not match"):
            get_gene_activity(region_topic, cell_topic.T, region_names, annotation, chrom_sizes)

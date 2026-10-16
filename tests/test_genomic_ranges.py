from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pycisTopic.genomic_ranges import extend, intersection, nearest, overlap


def _random_regions(rng, n, chroms=("chr1", "chr2"), max_pos=5000, max_len=300):
    starts = rng.integers(0, max_pos, n)
    return pl.DataFrame(
        {
            "Chromosome": rng.choice(chroms, n),
            "Start": starts,
            "End": starts + rng.integers(1, max_len, n),
            "Strand": rng.choice(["+", "-"], n),
            "ID": [f"id{i}" for i in range(n)],
        }
    )


def _brute_force_overlapping_pairs(df1, df2):
    pairs = set()
    for i, (c1, s1, e1) in enumerate(df1.select("Chromosome", "Start", "End").rows()):
        for j, (c2, s2, e2) in enumerate(
            df2.select("Chromosome", "Start", "End").rows()
        ):
            # Half-open intervals.
            if c1 == c2 and s1 < e2 and s2 < e1:
                pairs.add((i, j))
    return pairs


@pytest.fixture
def regions1():
    return pl.DataFrame(
        {
            "Chromosome": ["chr1"] * 3,
            "Start": [1, 4, 10],
            "End": [3, 9, 11],
            "ID": ["a", "b", "c"],
        }
    )


@pytest.fixture
def regions2():
    return pl.DataFrame(
        {
            "Chromosome": ["chr1"] * 3,
            "Start": [2, 2, 9],
            "End": [3, 9, 10],
            "Name": ["reg1", "reg2", "reg3"],
        }
    )


class TestIntersection:
    def test_docstring_example(self, regions1, regions2):
        result = intersection(regions1, regions2)
        assert result.select("Start", "End", "ID").rows() == [
            (2, 3, "a"),
            (2, 3, "a"),
            (4, 9, "b"),
        ]

    def test_bookended_regions_do_not_overlap(self):
        df1 = pl.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [10]})
        df2 = pl.DataFrame({"Chromosome": ["chr1"], "Start": [10], "End": [20]})
        assert intersection(df1, df2).height == 0

    def test_no_shared_chromosome_returns_empty(self, regions1):
        df2 = pl.DataFrame({"Chromosome": ["chrX"], "Start": [0], "End": [100]})
        assert intersection(regions1, df2).height == 0

    def test_add_overlap_size(self):
        df1 = pl.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [100]})
        df2 = pl.DataFrame({"Chromosome": ["chr1"], "Start": [50], "End": [250]})
        result = intersection(df1, df2, add_overlap_size=True)
        assert result.select("Start", "End").row(0) == (50, 100)
        assert result["fr1_inter"][0] == pytest.approx(0.5)
        assert result["fr2_inter"][0] == pytest.approx(0.25)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        df1 = _random_regions(rng, 60)
        df2 = _random_regions(rng, 40)
        result = intersection(
            df1.with_row_index("i1"),
            df2.with_row_index("i2"),
            regions1_info=True,
            regions2_info=True,
        )
        found = set(zip(result["i1"].to_list(), result["i2@2"].to_list()))
        assert found == _brute_force_overlapping_pairs(df1, df2)


class TestOverlap:
    def test_first(self, regions1, regions2):
        assert overlap(regions1, regions2)["ID"].to_list() == ["a", "b"]

    def test_invert(self, regions1, regions2):
        assert overlap(regions1, regions2, invert=True)["ID"].to_list() == ["c"]

    def test_invert_keeps_chromosomes_without_overlap(self):
        # Regression test: chromosomes not present in the second set of regions, or
        # without any overlap, were dropped when invert=True.
        df1 = pl.DataFrame(
            {
                "Chromosome": ["chr1", "chr1", "chr2", "chr3"],
                "Start": [120, 500, 5, 5],
                "End": [121, 501, 6, 6],
                "ID": ["overlaps", "chr1_no_overlap", "chr2_no_overlap", "chr3"],
            }
        )
        df2 = pl.DataFrame(
            {"Chromosome": ["chr1", "chr2"], "Start": [100, 1000], "End": [200, 2000]}
        )
        assert overlap(df1, df2, invert=True)["ID"].to_list() == [
            "chr1_no_overlap",
            "chr2_no_overlap",
            "chr3",
        ]
        assert overlap(df1, df2)["ID"].to_list() == ["overlaps"]

    @pytest.mark.parametrize("seed", [0, 1])
    def test_overlap_and_invert_partition_first_set(self, seed):
        rng = np.random.default_rng(seed)
        df1 = _random_regions(rng, 80).with_row_index("i1")
        df2 = _random_regions(rng, 30, chroms=("chr1",))
        overlapping = set(overlap(df1, df2)["i1"].to_list())
        not_overlapping = set(overlap(df1, df2, invert=True)["i1"].to_list())
        expected = {i for i, _ in _brute_force_overlapping_pairs(df1, df2)}
        assert overlapping == expected
        assert not_overlapping == set(range(df1.height)) - expected


class TestExtend:
    def test_strand_aware_fixed(self):
        df = pl.DataFrame(
            {
                "Chromosome": ["chr1"] * 3,
                "Start": [100, 100, 100],
                "End": [200, 200, 200],
                "Strand": ["+", "-", "."],
            }
        )
        result = extend(df, upstream=10, downstream=5)
        assert result.select("Start", "End").rows() == [(90, 205), (95, 210), (90, 205)]

    def test_per_region_columns_and_dtype(self):
        df = pl.DataFrame(
            {
                "Chromosome": ["chr1", "chr1"],
                "Start": pl.Series([100, 100], dtype=pl.Int32),
                "End": pl.Series([200, 200], dtype=pl.Int32),
                "Strand": ["+", "-"],
                "up": [1, 2],
                "down": [10, 20],
            }
        )
        result = extend(df, upstream="up", downstream="down")
        assert result.select("Start", "End").rows() == [(99, 210), (80, 202)]
        assert result.schema["Start"] == pl.Int32


class TestNearest:
    @pytest.fixture
    def genes(self):
        return pl.DataFrame(
            {
                "Chromosome": ["c1", "c1", "c1"],
                "Start": [100, 100, 1000],
                "End": [200, 200, 1100],
                "Strand": ["+", "-", "+"],
            }
        )

    @pytest.fixture
    def others(self):
        return pl.DataFrame(
            {
                "Chromosome": ["c1"] * 5,
                "Start": [10, 60, 150, 250, 95],
                "End": [50, 99, 400, 320, 105],
            }
        )

    def test_upstream_matches_pyranges(self, genes, others):
        # Expected values obtained with pyranges 0.1.4 nearest(overlap=False).
        result = nearest(genes, others, how="upstream")
        assert result["Distance"].to_list() == [2, 51, 601]

    def test_downstream_matches_pyranges(self, genes, others):
        result = nearest(genes, others, how="downstream")
        assert result["Distance"].to_list() == [51, 2, None]
        assert result["Start@2"].to_list() == [250, 60, None]

    def test_bookended_distance_is_one(self):
        genes = pl.DataFrame(
            {"Chromosome": ["c1"], "Start": [100], "End": [200], "Strand": ["+"]}
        )
        others = pl.DataFrame({"Chromosome": ["c1"], "Start": [90], "End": [100]})
        assert nearest(genes, others, how="upstream")["Distance"].to_list() == [1]

    def test_missing_chromosome_and_order(self, genes, others):
        genes = pl.concat(
            [
                genes,
                pl.DataFrame(
                    {"Chromosome": ["c2"], "Start": [1], "End": [2], "Strand": ["+"]}
                ),
            ]
        ).reverse()
        result = nearest(genes, others, how="previous")
        assert result.select("Chromosome", "Start").rows() == genes.select(
            "Chromosome", "Start"
        ).rows()
        assert result["Distance"].to_list() == [None, 601, 2, 2]

    def test_invalid_how(self, genes, others):
        with pytest.raises(ValueError, match="Invalid value for how"):
            nearest(genes, others, how="sideways")

    @pytest.mark.parametrize("how", ["upstream", "downstream", "previous", "next"])
    def test_matches_brute_force(self, how):
        rng = np.random.default_rng(3)
        df1 = _random_regions(rng, 50)
        df2 = _random_regions(rng, 30)
        result = nearest(df1, df2, how=how)
        for (c1, s1, e1, strand), distance in zip(
            df1.select("Chromosome", "Start", "End", "Strand").rows(),
            result["Distance"].to_list(),
        ):
            look_previous = {
                "previous": True,
                "next": False,
                "upstream": strand != "-",
                "downstream": strand == "-",
            }[how]
            candidates = [
                s1 - e2 + 1 if look_previous else s2 - e1 + 1
                for c2, s2, e2 in df2.select("Chromosome", "Start", "End").rows()
                if c2 == c1 and ((e2 <= s1) if look_previous else (s2 >= e1))
            ]
            assert distance == (min(candidates) if candidates else None)

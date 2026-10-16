from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pycisTopic.consensus_peaks import (
    calculate_center_peaks_and_extend,
    get_consensus_peaks,
    iterative_peak_filtering,
    normalize_peak_scores,
    read_chrom_sizes,
    read_peaks_bed,
    remove_peaks_overlapping_blacklisted_regions,
    write_peaks_bed,
)

DATA = Path(__file__).parent / "data" / "consensus_peaks"
SAMPLES = ("B_cells", "Mono", "T_cells")
CHROM_SIZES = {"chr1": 10_000, "chr2": 5_000}


def _peaks(rows):
    return pl.DataFrame(
        rows,
        schema={
            "Chromosome": pl.Utf8,
            "Start": pl.Int64,
            "End": pl.Int64,
            "Name": pl.Utf8,
            "Score": pl.Float64,
        },
        orient="row",
    )


def _read_reference(path):
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["Chromosome", "Start", "End", "Name", "Score"],
        schema_overrides={"Start": pl.Int64, "End": pl.Int64, "Score": pl.Float64},
    )


class TestReferenceScripts:
    """Compare with output of https://github.com/aertslab/iterative_peak_filtering."""

    @pytest.fixture
    def chrom_sizes(self):
        return read_chrom_sizes(DATA / "chrom.sizes")

    @pytest.mark.parametrize("sample", SAMPLES)
    def test_per_sample(self, chrom_sizes, sample):
        result = normalize_peak_scores(
            iterative_peak_filtering(
                calculate_center_peaks_and_extend(
                    read_peaks_bed(DATA / f"{sample}_summits.bed"), chrom_sizes, 250
                ),
                chrom_sizes,
            )
        )
        expected = _read_reference(
            DATA / f"{sample}.normalized.iterative_peak_filtering_scripts.bed"
        )
        assert result.equals(expected)

    def test_consensus(self, chrom_sizes, tmp_path):
        result = get_consensus_peaks(
            {s: read_peaks_bed(DATA / f"{s}_summits.bed") for s in SAMPLES},
            chrom_sizes,
            peak_half_width=250,
        )
        expected = _read_reference(
            DATA / "consensus_peaks.iterative_peak_filtering_scripts.bed"
        )
        assert result.equals(expected)
        assert set(result.select(pl.col("End") - pl.col("Start")).to_series()) == {500}

        write_peaks_bed(result, tmp_path / "consensus.bed")
        assert _read_reference(tmp_path / "consensus.bed").equals(expected)


def test_read_peaks_bed_narrow_peak(tmp_path):
    narrow_peak = tmp_path / "peaks.narrowPeak"
    narrow_peak.write_text("chr1\t10\t20\tpeak1\t30\t.\t2.5\t7.25\t3.1\t5\n")
    peaks = read_peaks_bed(narrow_peak)
    # -log10(pvalue) (column 8) is used as score.
    assert peaks.row(0) == ("chr1", 10, 20, "peak1", 7.25)
    (tmp_path / "bad.bed").write_text("chr1\t10\t20\n")
    with pytest.raises(ValueError, match="at least 5 columns"):
        read_peaks_bed(tmp_path / "bad.bed")


def test_read_chrom_sizes(tmp_path):
    (tmp_path / "genome.fa.fai").write_text("chr2\t500\t6\t60\t61\nchr1\t300\t512\t60\t61\n")
    assert read_chrom_sizes(tmp_path / "genome.fa.fai") == {"chr2": 500, "chr1": 300}


class TestCalculateCenterPeaksAndExtend:
    def test_center_and_extend(self):
        peaks = _peaks(
            [
                ("chr2", 1000, 1001, "summit", 1.0),
                ("chr1", 1000, 1011, "odd_length", 2.0),  # center 1005
                ("chr1", 1000, 1010, "even_length", 3.0),  # center 1005 (left)
            ]
        )
        result = calculate_center_peaks_and_extend(peaks, CHROM_SIZES, 100)
        assert result.select("Chromosome", "Start", "End", "Name").rows() == [
            ("chr1", 905, 1106, "odd_length"),
            ("chr1", 905, 1106, "even_length"),
            ("chr2", 900, 1101, "summit"),
        ]

    def test_peaks_at_chromosome_ends_are_removed(self):
        peaks = _peaks(
            [
                ("chr1", 99, 100, "too_close_to_start", 1.0),
                ("chr1", 100, 101, "start_ok", 1.0),
                ("chr1", 9899, 9900, "end_ok", 1.0),
                ("chr1", 9900, 9901, "too_close_to_end", 1.0),
            ]
        )
        result = calculate_center_peaks_and_extend(peaks, CHROM_SIZES, 100)
        assert result["Name"].to_list() == ["start_ok", "end_ok"]

    def test_unknown_chromosomes_are_removed(self):
        peaks = _peaks([("chrUn", 1000, 1001, "x", 1.0), ("chr1", 1000, 1001, "y", 1.0)])
        with pytest.warns(UserWarning, match="Removed 1 peaks"):
            result = calculate_center_peaks_and_extend(peaks, CHROM_SIZES, 100)
        assert result["Name"].to_list() == ["y"]


class TestIterativePeakFiltering:
    def test_single_and_bookended_pairs(self):
        peaks = _peaks(
            [
                ("chr1", 0, 100, "alone", 1.0),
                # Bookended peaks are merged: only the best one is kept.
                ("chr1", 200, 300, "pair_low", 1.0),
                ("chr1", 300, 400, "pair_high", 5.0),
                # Ties: the first peak is kept.
                ("chr1", 500, 600, "tie_first", 2.0),
                ("chr1", 550, 650, "tie_second", 2.0),
                ("chr2", 0, 100, "other_chrom", 1.0),
            ]
        )
        result = iterative_peak_filtering(peaks, CHROM_SIZES)
        assert result["Name"].to_list() == ["alone", "pair_high", "tie_first", "other_chrom"]

    def test_cluster_of_three_or_more(self):
        peaks = _peaks(
            [
                ("chr1", 1000, 1101, "a", 1.0),
                ("chr1", 1050, 1151, "b", 9.0),  # best: removes a and c (adjacent)
                ("chr1", 1120, 1221, "c", 8.0),
                ("chr1", 1200, 1301, "d", 7.0),  # does not overlap b: kept
                ("chr1", 1290, 1391, "e", 6.0),  # adjacent to d: removed
                ("chr1", 1352, 1453, "f", 5.0),  # does not overlap d: kept
            ]
        )
        result = iterative_peak_filtering(peaks, CHROM_SIZES)
        assert result["Name"].to_list() == ["b", "d", "f"]

    def test_peaks_bookended_with_selected_peak_are_removed(self):
        peaks = _peaks(
            [
                ("chr1", 0, 100, "a", 1.0),
                ("chr1", 50, 150, "b", 9.0),
                ("chr1", 100, 200, "c", 0.5),  # adjacent to b
                ("chr1", 150, 250, "d", 2.0),  # bookended with b: overlap
                ("chr1", 251, 351, "e", 0.1),  # merged into the cluster via d
            ]
        )
        result = iterative_peak_filtering(peaks, CHROM_SIZES)
        assert result["Name"].to_list() == ["b", "e"]

    def test_input_order_does_not_matter(self):
        peaks = _peaks(
            [
                ("chr2", 0, 100, "x", 1.0),
                ("chr1", 1050, 1151, "b", 9.0),
                ("chr1", 1000, 1101, "a", 1.0),
            ]
        )
        result = iterative_peak_filtering(peaks, CHROM_SIZES)
        assert result["Name"].to_list() == ["b", "x"]


def test_normalize_peak_scores():
    peaks = _peaks([("chr1", 0, 1, "a", 1.0), ("chr1", 5, 6, "b", 2.0)])
    result = normalize_peak_scores(peaks)
    # Rounded to 6 significant digits (as printed by gawk).
    assert result["Score"].to_list() == [333333.0, 666667.0]


def test_remove_peaks_overlapping_blacklisted_regions():
    peaks = _peaks(
        [
            ("chr1", 0, 100, "a", 1.0),
            ("chr1", 200, 300, "partial_overlap", 1.0),
            ("chr2", 0, 100, "b", 1.0),
        ]
    )
    blacklist = pl.DataFrame({"Chromosome": ["chr1"], "Start": [290], "End": [400]})
    result = remove_peaks_overlapping_blacklisted_regions(peaks, blacklist, CHROM_SIZES)
    assert result["Name"].to_list() == ["a", "b"]


def test_consensus_with_blacklist():
    summits = {
        "s1": _peaks([("chr1", 1000, 1001, "s1_a", 5.0), ("chr1", 5000, 5001, "s1_b", 1.0)]),
        "s2": _peaks([("chr1", 1100, 1101, "s2_a", 1.0), ("chr2", 2000, 2001, "s2_b", 1.0)]),
    }
    result = get_consensus_peaks(summits, CHROM_SIZES, peak_half_width=100)
    # s1_a and s2_a overlap: normalized scores 833333 (s1_a) vs 500000 (s2_a).
    assert result.select("Chromosome", "Start", "End", "Name").rows() == [
        ("chr1", 900, 1100, "s1_a"),
        ("chr1", 4900, 5100, "s1_b"),
        ("chr2", 1900, 2100, "s2_b"),
    ]
    blacklist = pl.DataFrame({"Chromosome": ["chr2"], "Start": [0], "End": [5000]})
    result = get_consensus_peaks(summits, CHROM_SIZES, 100, blacklist=blacklist)
    assert result["Name"].to_list() == ["s1_a", "s1_b"]

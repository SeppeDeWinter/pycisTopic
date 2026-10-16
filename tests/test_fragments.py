from __future__ import annotations

import gzip

import numpy as np
import polars as pl
import pytest

from pycisTopic.fragments import (
    create_fragment_matrix_from_fragments,
    filter_fragments_by_cb,
    get_cbs_passing_filter,
    get_fragments_in_peaks,
    get_fragments_per_cb,
    get_insert_size_distribution,
    read_barcodes_file_to_polars_series,
    read_bed_to_polars_df,
    read_fragments_to_polars_df,
    region_ids_to_bed_df_pl,
)

BED_PARSER_ENGINES = ["polars_lazy", "polars", "pyarrow"]


def _as_str(df: pl.DataFrame, *columns: str) -> pl.DataFrame:
    return df.with_columns(pl.col(c).cast(pl.Utf8) for c in columns)


class TestReadBed:
    @pytest.mark.parametrize("engine", BED_PARSER_ENGINES)
    def test_read_bed(self, small_files, engine):
        df = read_bed_to_polars_df(small_files["regions"], bed_parser_engine=engine)
        assert df.columns == ["Chromosome", "Start", "End"]
        assert df.schema["Start"] == pl.Int32
        assert _as_str(df, "Chromosome").rows() == [
            ("chr1", 100, 200),
            ("chr1", 300, 400),
            ("chr1", 1000, 1100),
            ("chr2", 50, 150),
        ]

    def test_read_gzipped_bed(self, small_files, tmp_path):
        gz = tmp_path / "regions.bed.gz"
        with gzip.open(gz, "wt") as fh:
            fh.write(small_files["regions"].read_text())
        assert read_bed_to_polars_df(gz).height == 4

    def test_too_few_columns(self, small_files):
        with pytest.raises(ValueError, match="at least 4 columns"):
            read_bed_to_polars_df(small_files["regions"], min_column_count=4)

    def test_invalid_engine(self, small_files):
        with pytest.raises(ValueError, match="Unsupported bed_parser_engine"):
            read_bed_to_polars_df(small_files["regions"], bed_parser_engine="awk")

    def test_region_ids_to_bed_df_pl(self):
        df = region_ids_to_bed_df_pl(["chr1:1000-2000", "chr2:1500-2500"])
        assert _as_str(df, "Chromosome", "RegionID").rows() == [
            ("chr1", 1000, 2000, "chr1:1000-2000"),
            ("chr2", 1500, 2500, "chr2:1500-2500"),
        ]


class TestReadFragments:
    @pytest.mark.parametrize("engine", BED_PARSER_ENGINES)
    def test_read_fragments(self, small_files, engine):
        df = read_fragments_to_polars_df(
            small_files["fragments"], bed_parser_engine=engine
        )
        assert df.columns == ["Chromosome", "Start", "End", "CB", "CB_count"]
        assert df.height == 7
        assert df["CB_count"].sum() == 10

    def test_sample_id_and_cb_end(self, small_files):
        df = read_fragments_to_polars_df(
            small_files["fragments"], sample_id="s1", cb_end_to_remove="-1"
        )
        assert sorted(df["CB"].cast(pl.Utf8).unique().to_list()) == [
            "AAA___s1",
            "BBB___s1",
            "CCC___s1",
        ]

    def test_four_column_fragments_are_counted(self, tmp_path):
        fragments = tmp_path / "fragments4.tsv"
        fragments.write_text(
            "chr1\t10\t20\tA\nchr1\t10\t20\tA\nchr1\t30\t40\tA\nchr1\t30\t40\tB\n"
        )
        df = (
            _as_str(read_fragments_to_polars_df(fragments), "Chromosome", "CB")
            .sort("Start", "CB")
            .select("Start", "CB", "CB_count")
        )
        assert df.rows() == [(10, "A", 2), (30, "A", 1), (30, "B", 1)]

    def test_read_barcodes(self, small_files):
        cbs = read_barcodes_file_to_polars_series(
            small_files["barcodes"], sample_id="s1", cb_end_to_remove="-1"
        )
        assert cbs.cast(pl.Utf8).to_list() == ["AAA___s1", "BBB___s1"]


class TestFragmentStats:
    @pytest.fixture
    def fragments(self, small_files):
        return read_fragments_to_polars_df(small_files["fragments"])

    def test_get_fragments_per_cb(self, fragments):
        stats = _as_str(
            get_fragments_per_cb(fragments, min_fragments_per_cb=0), "CB"
        ).sort("CB")
        assert stats["CB"].to_list() == ["AAA-1", "BBB-1", "CCC-1"]
        assert stats["total_fragments_count"].to_list() == [4, 5, 1]
        assert stats["unique_fragments_count"].to_list() == [3, 3, 1]
        assert stats["duplication_count"].to_list() == [1, 2, 0]
        # AAA-1 fragment lengths: 60, 200, 100 -> 2 nucleosome free, 1 mononucleosome.
        assert stats["nucleosome_signal"][0] == pytest.approx(0.5)
        # Barcode rank by unique fragments (ties keep input order).
        assert stats.sort("barcode_rank")["CB"].to_list()[-1] == "CCC-1"

    def test_min_fragments_per_cb_is_exclusive(self, fragments):
        stats = get_fragments_per_cb(fragments, min_fragments_per_cb=1)
        assert "CCC-1" not in stats["CB"].cast(pl.Utf8).to_list()

    def test_get_cbs_passing_filter_min_fragments(self, fragments):
        stats = get_fragments_per_cb(fragments, min_fragments_per_cb=0)
        cbs, _ = get_cbs_passing_filter(stats, min_fragments_per_cb=3)
        assert sorted(cbs.cast(pl.Utf8).to_list()) == ["AAA-1", "BBB-1"]

    def test_get_cbs_passing_filter_top_x(self, fragments):
        stats = get_fragments_per_cb(
            fragments, min_fragments_per_cb=0, collapse_duplicates=False
        )
        cbs, _ = get_cbs_passing_filter(
            stats, keep_top_x_cbs=1, collapse_duplicates=False
        )
        assert cbs.cast(pl.Utf8).to_list() == ["BBB-1"]

    def test_get_cbs_passing_filter_cb_list(self, fragments):
        stats = get_fragments_per_cb(fragments, min_fragments_per_cb=0)
        cbs, _ = get_cbs_passing_filter(stats, cbs=["CCC-1", "AAA-1"])
        assert sorted(cbs.cast(pl.Utf8).to_list()) == ["AAA-1", "CCC-1"]

    def test_get_cbs_passing_filter_cb_series(self, fragments):
        stats = get_fragments_per_cb(fragments, min_fragments_per_cb=0)
        cbs, _ = get_cbs_passing_filter(stats, cbs=pl.Series(["CCC-1", "AAA-1"]))
        assert sorted(cbs.cast(pl.Utf8).to_list()) == ["AAA-1", "CCC-1"]

    def test_get_cbs_passing_filter_requires_criterion(self, fragments):
        stats = get_fragments_per_cb(fragments, min_fragments_per_cb=0)
        with pytest.raises(ValueError, match="Provide a minimal number"):
            get_cbs_passing_filter(stats)

    def test_filter_fragments_by_cb(self, fragments):
        filtered = filter_fragments_by_cb(fragments, ["BBB-1"])
        assert set(filtered["CB"].cast(pl.Utf8).to_list()) == {"BBB-1"}
        assert filtered.height == 3

    def test_insert_size_distribution(self, fragments):
        dist = get_insert_size_distribution(fragments).sort("insert_size")
        assert dict(zip(dist["insert_size"], dist["fragments_count"])) == {
            10: 4,
            60: 1,
            100: 1,
            200: 1,
        }
        assert dist["fragments_ratio"].sum() == pytest.approx(1.0)

    def test_fragments_in_peaks(self, small_files, fragments):
        regions = read_bed_to_polars_df(small_files["regions"])
        in_peaks = _as_str(get_fragments_in_peaks(fragments, regions), "CB").sort("CB")
        assert in_peaks.rows() == [
            # AAA-1: chr1:120-180 (x2) and chr1:150-350; chr1:500-600 is outside.
            ("AAA-1", 3, 2),
            # BBB-1: all 3 fragments overlap a region.
            ("BBB-1", 5, 3),
            ("CCC-1", 1, 1),
        ]


class TestFragmentMatrix:
    def _matrix_as_dict(self, matrix, cbs, region_ids):
        coo = matrix.tocoo()
        return {
            (region_ids[r], cbs[c]): int(v)
            for r, c, v in zip(coo.row, coo.col, coo.data)
            if v != 0
        }

    @pytest.mark.parametrize("engine", BED_PARSER_ENGINES)
    def test_binary_matrix(self, small_files, engine):
        matrix, cbs, region_ids = create_fragment_matrix_from_fragments(
            fragments_bed_filename=small_files["fragments"],
            regions_bed_filename=small_files["regions"],
            barcodes_tsv_filename=small_files["barcodes"],
            sample_id="s1",
            bed_parser_engine=engine,
        )
        assert cbs == ["AAA___s1", "BBB___s1"]
        assert sorted(region_ids) == sorted(
            ["chr1:100-200", "chr1:300-400", "chr1:1000-1100", "chr2:50-150"]
        )
        assert matrix.shape == (4, 2)
        assert self._matrix_as_dict(matrix, cbs, region_ids) == {
            ("chr1:100-200", "AAA___s1"): 1,
            ("chr1:300-400", "AAA___s1"): 1,
            ("chr1:300-400", "BBB___s1"): 1,
            ("chr1:1000-1100", "BBB___s1"): 1,
            ("chr2:50-150", "BBB___s1"): 1,
        }

    def test_count_matrix_counts_unique_fragments(self, small_files):
        matrix, cbs, region_ids = create_fragment_matrix_from_fragments(
            fragments_bed_filename=small_files["fragments"],
            regions_bed_filename=small_files["regions"],
            barcodes_tsv_filename=small_files["barcodes"],
            sample_id="s1",
            fragment_matrix_type="count",
        )
        counts = self._matrix_as_dict(matrix, cbs, region_ids)
        # Two different fragments of AAA-1 overlap chr1:100-200.
        assert counts[("chr1:100-200", "AAA___s1")] == 2
        assert counts[("chr1:1000-1100", "BBB___s1")] == 1

    def test_blacklist_removes_regions(self, small_files):
        matrix, _, region_ids = create_fragment_matrix_from_fragments(
            fragments_bed_filename=small_files["fragments"],
            regions_bed_filename=small_files["regions"],
            barcodes_tsv_filename=small_files["barcodes"],
            blacklist_bed_filename=small_files["blacklist"],
            sample_id="s1",
        )
        assert "chr1:1000-1100" not in region_ids
        assert matrix.shape == (3, 2)

    def test_matrix_rows_match_region_ids(self, small_files):
        matrix, _, region_ids = create_fragment_matrix_from_fragments(
            fragments_bed_filename=small_files["fragments"],
            regions_bed_filename=small_files["regions"],
            barcodes_tsv_filename=small_files["barcodes"],
            sample_id="s1",
        )
        row_sums = dict(zip(region_ids, np.asarray(matrix.sum(axis=1)).ravel()))
        assert row_sums == {
            "chr1:100-200": 1,
            "chr1:300-400": 2,
            "chr1:1000-1100": 1,
            "chr2:50-150": 1,
        }

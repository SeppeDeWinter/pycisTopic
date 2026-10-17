from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from scipy import io as sp_io

from pycisTopic.count_matrix import (
    check_accessibility_matrix_files,
    create_count_matrix_from_fragment_files,
    read_mapping,
    subset_accessibility_matrix,
)


def _read_matrix(prefix):
    matrix = sp_io.mmread(f"{prefix}.matrix.mtx").toarray()
    with open(f"{prefix}.cell_barcodes.tsv") as fh:
        cbs = [line.strip() for line in fh]
    with open(f"{prefix}.region_ids.tsv") as fh:
        region_ids = [line.strip() for line in fh]
    return matrix, cbs, region_ids


@pytest.fixture
def mapping_files(small_files, tmp_path):
    # Two samples sharing the same fragments file.
    sample_fragments = tmp_path / "sample_fragments.tsv"
    sample_fragments.write_text(
        "sample\tfragments_filename\n"
        f"s1\t{small_files['fragments']}\n"
        f"s2\t{small_files['fragments']}\n"
    )
    sample_barcodes = tmp_path / "sample_barcodes.tsv"
    sample_barcodes.write_text(
        "sample\tcell_barcodes_filename\n"
        f"s1\t{small_files['barcodes']}\n"
        f"s2\t{small_files['barcodes']}\n"
    )
    return sample_fragments, sample_barcodes


@pytest.fixture
def count_matrix_prefix(small_files, mapping_files, tmp_path):
    sample_fragments, sample_barcodes = mapping_files
    prefix = str(tmp_path / "out")
    create_count_matrix_from_fragment_files(
        sample_to_fragments_file_mapping_tsv_filename=str(sample_fragments),
        sample_to_cell_barcodes_file_mapping_tsv_filenames=str(sample_barcodes),
        regions_bed_filename=str(small_files["regions"]),
        output_prefix=prefix,
        cb_end_to_remove="-1",
    )
    return prefix


class TestReadMapping:
    def test_read_mapping(self, mapping_files):
        mapping = read_mapping(str(mapping_files[0]), "sample", "fragments_filename", "\t")
        assert list(mapping) == ["s1", "s2"]

    def test_missing_column(self, mapping_files):
        with pytest.raises(KeyError, match="must have columns"):
            read_mapping(str(mapping_files[0]), "sample", "barcodes", "\t")

    def test_duplicate_key(self, tmp_path):
        mapping = tmp_path / "dup.tsv"
        mapping.write_text("sample\tfragments_filename\ns1\ta\ns1\tb\n")
        with pytest.raises(ValueError, match="Duplicate s1"):
            read_mapping(str(mapping), "sample", "fragments_filename", "\t")


class TestCreateCountMatrix:
    def test_merged_matrix(self, count_matrix_prefix):
        matrix, cbs, region_ids = _read_matrix(count_matrix_prefix)
        assert cbs == ["AAA___s1", "BBB___s1", "AAA___s2", "BBB___s2"]
        assert matrix.shape == (4, 4)
        # Both samples use the same fragments, so their columns are identical.
        np.testing.assert_array_equal(matrix[:, :2], matrix[:, 2:])
        accessible = {
            (region_ids[r], cbs[c]) for r, c in zip(*np.nonzero(matrix[:, :2]))
        }
        assert accessible == {
            ("chr1:100-200", "AAA___s1"),
            ("chr1:300-400", "AAA___s1"),
            ("chr1:300-400", "BBB___s1"),
            ("chr1:1000-1100", "BBB___s1"),
            ("chr2:50-150", "BBB___s1"),
        }

    def test_ruranges_engine(self, small_files, mapping_files, count_matrix_prefix, tmp_path):
        pytest.importorskip("ruranges")
        sample_fragments, sample_barcodes = mapping_files
        prefix = str(tmp_path / "out_ruranges")
        create_count_matrix_from_fragment_files(
            sample_to_fragments_file_mapping_tsv_filename=str(sample_fragments),
            sample_to_cell_barcodes_file_mapping_tsv_filenames=str(sample_barcodes),
            regions_bed_filename=str(small_files["regions"]),
            output_prefix=prefix,
            cb_end_to_remove="-1",
            intersection_engine="ruranges",
        )
        expected = _read_matrix(count_matrix_prefix)
        result = _read_matrix(prefix)
        np.testing.assert_array_equal(result[0], expected[0])
        assert result[1:] == expected[1:]

    def test_sample_without_barcodes(self, small_files, mapping_files, tmp_path):
        sample_fragments, _ = mapping_files
        sample_barcodes = tmp_path / "only_s1.tsv"
        sample_barcodes.write_text(
            f"sample\tcell_barcodes_filename\ns1\t{small_files['barcodes']}\n"
        )
        with pytest.raises(ValueError, match='Sample "s2" was not present'):
            create_count_matrix_from_fragment_files(
                sample_to_fragments_file_mapping_tsv_filename=str(sample_fragments),
                sample_to_cell_barcodes_file_mapping_tsv_filenames=str(sample_barcodes),
                regions_bed_filename=str(small_files["regions"]),
                output_prefix=str(tmp_path / "out"),
            )


class TestCheckAccessibilityMatrixFiles:
    def test_all_present(self, count_matrix_prefix):
        assert check_accessibility_matrix_files(f"{count_matrix_prefix}.matrix.mtx")

    def test_wrong_suffix(self, tmp_path):
        with pytest.raises(ValueError, match=r"matrix\.mtx"):
            check_accessibility_matrix_files(str(tmp_path / "out.mtx"))

    def test_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"cell_barcodes\.tsv"):
            check_accessibility_matrix_files(str(tmp_path / "out.matrix.mtx"))


class TestSubsetAccessibilityMatrix:
    def test_subset_cells_and_regions(self, count_matrix_prefix, tmp_path):
        matrix, cbs, region_ids = _read_matrix(count_matrix_prefix)
        keep_cbs = ["BBB___s2", "AAA___s1"]
        keep_regions = ["chr2:50-150", "chr1:300-400"]
        subset_prefix = str(tmp_path / "subset")
        subset_accessibility_matrix(
            f"{count_matrix_prefix}.matrix.mtx",
            subset_prefix,
            cell_barcodes_to_keep=keep_cbs,
            region_ids_to_keep=keep_regions,
        )
        sub_matrix, sub_cbs, sub_region_ids = _read_matrix(subset_prefix)
        assert sub_cbs == keep_cbs
        assert sub_region_ids == keep_regions
        expected = matrix[
            np.ix_(
                [region_ids.index(r) for r in keep_regions],
                [cbs.index(c) for c in keep_cbs],
            )
        ]
        np.testing.assert_array_equal(sub_matrix, expected)

    def test_keep_all(self, count_matrix_prefix, tmp_path):
        subset_prefix = str(tmp_path / "subset")
        subset_accessibility_matrix(
            f"{count_matrix_prefix}.matrix.mtx", subset_prefix, None, None
        )
        for original, subset in zip(
            _read_matrix(count_matrix_prefix), _read_matrix(subset_prefix)
        ):
            np.testing.assert_array_equal(np.asarray(original), np.asarray(subset))

    def test_unknown_barcode(self, count_matrix_prefix, tmp_path):
        with pytest.raises(ValueError, match='Cell barcode "nope" not found'):
            subset_accessibility_matrix(
                f"{count_matrix_prefix}.matrix.mtx", str(tmp_path / "s"), ["nope"]
            )


CLI = [sys.executable, "-m", "pycisTopic.cli.pycistopic", "count_matrix"]


def test_cli_create(small_files, mapping_files, tmp_path):
    sample_fragments, sample_barcodes = mapping_files
    prefix = tmp_path / "cli"
    subprocess.run(
        [
            *CLI,
            "create",
            "-f", str(sample_fragments),
            "-c", str(sample_barcodes),
            "-r", str(small_files["regions"]),
            "-o", str(prefix),
            "-d", "-1",
            "-b", str(small_files["blacklist"]),
        ],
        check=True,
        capture_output=True,
    )
    matrix, _, region_ids = _read_matrix(prefix)
    assert matrix.shape == (3, 4)
    assert "chr1:1000-1100" not in region_ids


def test_cli_subset(count_matrix_prefix, tmp_path):
    keep = tmp_path / "keep_cbs.txt"
    keep.write_text("AAA___s2\n")
    subprocess.run(
        [
            *CLI,
            "subset",
            "-m", f"{count_matrix_prefix}.matrix.mtx",
            "-o", str(tmp_path / "sub"),
            "-c", str(keep),
        ],
        check=True,
        capture_output=True,
    )
    sub_matrix, sub_cbs, _ = _read_matrix(tmp_path / "sub")
    assert sub_cbs == ["AAA___s2"]
    assert sub_matrix.shape == (4, 1)

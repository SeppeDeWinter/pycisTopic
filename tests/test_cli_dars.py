from __future__ import annotations

import polars as pl

from pycisTopic.cli.subcommand.dars import get_contrast_barcodes_lists


def _cell_data(labels):
    return pl.DataFrame(
        {"cell_barcode": [f"c{i}" for i in range(len(labels))], "cell_type": labels}
    )


def test_one_vs_all_ignores_cells_without_annotation():
    contrasts = get_contrast_barcodes_lists(
        _cell_data(["A", "B", None, "A", "C"]), "cell_barcode", "cell_type"
    )
    assert contrasts == {
        "fg_A_vs_bg_others": (["c0", "c3"], ["c1", "c4"]),
        "fg_B_vs_bg_others": (["c1"], ["c0", "c3", "c4"]),
        "fg_C_vs_bg_others": (["c4"], ["c0", "c1", "c3"]),
    }


def test_contrasts_tsv(tmp_path):
    contrasts_tsv = tmp_path / "contrasts.tsv"
    contrasts_tsv.write_text("foreground\tbackground\nA\t\nA,B\tC\n")
    contrasts = get_contrast_barcodes_lists(
        _cell_data(["A", "B", None, "A", "C"]),
        "cell_barcode",
        "cell_type",
        contrasts_tsv=contrasts_tsv,
    )
    assert contrasts == {
        "fg_A_vs_bg_others": (["c0", "c3"], ["c1", "c4"]),
        "fg_A_B_vs_bg_C": (["c0", "c3", "c1"], ["c4"]),
    }


def test_numeric_annotations(tmp_path):
    cell_data_tsv = tmp_path / "cell_data.tsv"
    _cell_data([1, 2, 1]).write_csv(cell_data_tsv, separator="\t")
    contrasts = get_contrast_barcodes_lists(cell_data_tsv, "cell_barcode", "cell_type")
    assert contrasts == {
        "fg_1_vs_bg_others": (["c0", "c2"], ["c1"]),
        "fg_2_vs_bg_others": (["c1"], ["c0", "c2"]),
    }

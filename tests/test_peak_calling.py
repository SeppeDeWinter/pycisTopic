from __future__ import annotations

import gzip
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pycisTopic.adata import create_anndata
from pycisTopic.peak_calling import (
    call_peaks_macs3,
    export_pseudobulk,
    find_pseudobulk_fragment_files,
    get_cell_type_barcodes_per_sample,
    macs3_callpeak_command,
    pseudobulk_name,
)

DATA = Path(__file__).parent / "data"
SAMPLE_TO_FRAGMENTS = {
    s: str(DATA / f"{s}.fragments.tsv.gz") for s in ("sample1", "sample2")
}
CHROM_SIZES = {"chr1": 10_000, "chr2": 5_000}
CLI = [sys.executable, "-m", "pycisTopic.cli.pycistopic"]


def _barcodes_in_fragments(filename):
    with gzip.open(filename, "rt") as fh:
        lines = fh.read().splitlines()
    return len(lines), sorted({line.split("\t")[3] for line in lines})


@pytest.fixture
def cell_data():
    return pd.DataFrame(
        {"cell_type": ["T cells/NK", "B", "T cells/NK", np.nan]},
        index=["AAA___sample1", "BBB___sample1", "BBB___sample2", "CCC___sample2"],
    )


class TestCellTypeBarcodes:
    def test_mapping(self):
        mapping = get_cell_type_barcodes_per_sample(
            ["AAA___s1", "BBB___s1", "AAA___s2", "CCC___s2"],
            ["T cells", "B", "T cells", None],
            ["s1", "s2"],
            cb_end_to_add="-1",
        )
        assert mapping == {
            "s1": {"T_cells": ["AAA-1"], "B": ["BBB-1"]},
            "s2": {"T_cells": ["AAA-1"]},
        }

    def test_single_sample_without_sample_id(self):
        mapping = get_cell_type_barcodes_per_sample(["AAA", "BBB"], ["x", "y"], ["s1"])
        assert mapping == {"s1": {"x": ["AAA"], "y": ["BBB"]}}

    def test_errors(self):
        with pytest.raises(ValueError, match="multiple samples"):
            get_cell_type_barcodes_per_sample(["AAA"], ["x"], ["s1", "s2"])
        with pytest.raises(ValueError, match='No fragments file for sample "s3"'):
            get_cell_type_barcodes_per_sample(["AAA___s3"], ["x"], ["s1"])
        with pytest.raises(ValueError, match="differ"):
            get_cell_type_barcodes_per_sample(["AAA___s1"], [], ["s1"])


def test_pseudobulk_name():
    assert pseudobulk_name("CD4+ T cells/NK") == "CD4+_T_cells_NK"


class TestExportPseudobulk:
    def test_pandas(self, cell_data, tmp_path):
        files = export_pseudobulk(
            cell_data,
            "cell_type",
            SAMPLE_TO_FRAGMENTS,
            CHROM_SIZES,
            tmp_path / "pseudobulk",
            cb_end_to_add="-1",
        )
        assert files == {
            "B": str(tmp_path / "pseudobulk" / "B.fragments.tsv.gz"),
            "T_cells_NK": str(tmp_path / "pseudobulk" / "T_cells_NK.fragments.tsv.gz"),
        }
        # B: BBB-1 of sample1 (3 fragments).
        assert _barcodes_in_fragments(files["B"]) == (3, ["BBB-1"])
        # T cells/NK: AAA-1 of sample1 (3) + BBB-1 of sample2 (3).
        assert _barcodes_in_fragments(files["T_cells_NK"]) == (6, ["AAA-1", "BBB-1"])
        assert find_pseudobulk_fragment_files(tmp_path / "pseudobulk") == files

    def test_polars(self, cell_data, tmp_path):
        polars_cell_data = pl.from_pandas(cell_data.reset_index(names="barcode"))
        files = export_pseudobulk(
            polars_cell_data,
            "cell_type",
            SAMPLE_TO_FRAGMENTS,
            CHROM_SIZES,
            tmp_path,
            barcode_column="barcode",
            cb_end_to_add="-1",
        )
        assert sorted(files) == ["B", "T_cells_NK"]

    def test_name_collision(self, tmp_path):
        cell_data = pd.DataFrame(
            {"cell_type": ["a b", "a_b"]}, index=["AAA___sample1", "BBB___sample1"]
        )
        with pytest.raises(ValueError, match="same name"):
            export_pseudobulk(
                cell_data, "cell_type", SAMPLE_TO_FRAGMENTS, CHROM_SIZES, tmp_path
            )


@pytest.fixture
def fake_macs3(tmp_path):
    """Fake macs3 executable which records its arguments and writes a summits file."""
    script = tmp_path / "bin" / "macs3"
    script.parent.mkdir()
    script.write_text(
        f"""#!{sys.executable}
import sys, os
args = sys.argv[1:]
name = args[args.index("--name") + 1]
if name == "fail":
    sys.exit("macs3 error for fail")
outdir = args[args.index("--outdir") + 1]
with open(os.path.join(outdir, name + "_summits.bed"), "w") as fh:
    fh.write("chr1\\t1000\\t1001\\t" + name + "_peak_1\\t12.5\\n")
with open(os.path.join(outdir, name + ".args"), "w") as fh:
    fh.write(" ".join(args))
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


class TestCallPeaksMacs3:
    def test_command(self):
        command = macs3_callpeak_command("B.fragments.tsv.gz", "B", "out", "hs")
        assert command == [
            "macs3", "callpeak", "--treatment", "B.fragments.tsv.gz", "--name", "B",
            "--outdir", "out", "--format", "FRAG", "--gsize", "hs", "--qvalue", "0.05",
            "--max-count", "1", "--nomodel", "--shift", "73", "--extsize", "146",
            "--nolambda", "--call-summits",
        ]  # fmt: skip

    def test_call_peaks(self, fake_macs3, tmp_path):
        summits = call_peaks_macs3(
            {"B": "B.fragments.tsv.gz", "T": "T.fragments.tsv.gz"},
            tmp_path / "macs",
            genome_size=2_913_022_398,
            qvalue=0.01,
            n_jobs=2,
            macs3_path=fake_macs3,
        )
        assert summits == {
            name: str(tmp_path / "macs" / f"{name}_summits.bed") for name in ("B", "T")
        }
        args = (tmp_path / "macs" / "B.args").read_text().split()
        assert args[args.index("--gsize") + 1] == "2913022398"
        assert args[args.index("--qvalue") + 1] == "0.01"

    def test_macs3_failure(self, fake_macs3, tmp_path):
        with pytest.raises(RuntimeError, match="macs3 error for fail"):
            call_peaks_macs3({"fail": "x"}, tmp_path, "hs", macs3_path=fake_macs3)

    def test_macs3_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="bioconda macs3"):
            call_peaks_macs3({"B": "x"}, tmp_path, "hs", macs3_path="no_such_macs3")


def test_cli_pseudobulk_callpeaks_consensus(cell_data, fake_macs3, tmp_path):
    adata = create_anndata(
        np.full((4, 2), 0.5),
        np.full((3, 2), 0.5),
        cell_data.index.tolist(),
        ["chr1:0-100", "chr1:200-300", "chr2:0-100"],
        obs=cell_data,
    )
    adata.write_h5ad(tmp_path / "model.h5ad")
    sample_fragments = tmp_path / "sample_fragments.tsv"
    sample_fragments.write_text(
        "sample\tfragments_filename\n"
        + "".join(f"{s}\t{f}\n" for s, f in SAMPLE_TO_FRAGMENTS.items())
    )
    chrom_sizes = tmp_path / "chrom.sizes"
    chrom_sizes.write_text("chr1\t10000\nchr2\t5000\n")

    def run(*args):
        result = subprocess.run([*CLI, *map(str, args)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    run(
        "pseudobulk", "-a", tmp_path / "model.h5ad", "-g", "cell_type",
        "-f", sample_fragments, "-c", chrom_sizes, "-o", tmp_path / "pseudobulk",
        "-d", "-1",
    )  # fmt: skip
    assert sorted(p.name for p in (tmp_path / "pseudobulk").glob("*.fragments.tsv.gz")) == [
        "B.fragments.tsv.gz",
        "T_cells_NK.fragments.tsv.gz",
    ]

    run(
        "callpeaks", "-i", tmp_path / "pseudobulk", "-o", tmp_path / "macs",
        "-g", "hs", "-n", "2", "-e", fake_macs3,
    )  # fmt: skip
    assert sorted(p.name for p in (tmp_path / "macs").glob("*_summits.bed")) == [
        "B_summits.bed",
        "T_cells_NK_summits.bed",
    ]

    run(
        "consensus_peaks", "-i", tmp_path / "macs", "-c", chrom_sizes,
        "-o", tmp_path / "consensus.bed", "-w", "250",
    )  # fmt: skip
    # Both fake summits are at chr1:1000: one consensus peak of 500 bp.
    lines = (tmp_path / "consensus.bed").read_text().splitlines()
    assert [line.split("\t")[:3] for line in lines] == [["chr1", "750", "1250"]]

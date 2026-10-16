"""End-to-end tests of CLI commands that use the pycisTopic AnnData file."""

from __future__ import annotations

import subprocess
import sys

import anndata
import numpy as np
import pandas as pd
import pytest

from pycisTopic.adata import create_anndata

CLI = [sys.executable, "-m", "pycisTopic.cli.pycistopic"]


def _run(*args):
    result = subprocess.run([*CLI, *map(str, args)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def adata_file(tmp_path, topic_model):
    region_topic, cell_topic, region_names, cell_names = topic_model
    groups = np.where(cell_topic[0] > np.quantile(cell_topic[0], 0.8), "high", "low")
    adata = create_anndata(
        cell_topic.T,
        region_topic,
        cell_names,
        region_names,
        obs=pd.DataFrame({"cell_type": groups}, index=cell_names),
    )
    path = tmp_path / "model.h5ad"
    adata.write_h5ad(path)
    return path


def test_signature_enrichment(adata_file, tmp_path, topic_model):
    _, _, region_names, _ = topic_model
    bed_files = []
    for i, regions in enumerate((region_names[:20], region_names[100:150])):
        bed = tmp_path / f"signature{i}.bed"
        bed.write_text(
            "".join(r.replace(":", "\t").replace("-", "\t") + "\n" for r in regions)
        )
        bed_files.append(bed)

    out1 = tmp_path / "out1.h5ad"
    _run("signature_enrichment", "-a", adata_file, "-n", "s0", "-s", bed_files[0], "-o", out1)
    adata = anndata.read_h5ad(out1)
    assert adata.obsm["AUCell"].shape == (adata.n_obs, 1)
    assert list(adata.uns["AUCell signature names"]) == ["s0"]

    # Adding signatures to a file which already contains AUCell values.
    out2 = tmp_path / "out2.h5ad"
    _run("signature_enrichment", "-a", out1, "-n", "s1", "-s", bed_files[1], "-o", out2)
    adata = anndata.read_h5ad(out2)
    assert adata.obsm["AUCell"].shape == (adata.n_obs, 2)
    assert list(adata.uns["AUCell signature names"]) == ["s0", "s1"]
    assert "X_topic" in adata.obsm


def test_dars(adata_file, tmp_path):
    hv_bed = tmp_path / "hv_regions.bed"
    _run("dars", "compute_hv_regions", "-a", adata_file, "--output-bed", hv_bed)
    hv_regions = hv_bed.read_text().splitlines()
    assert len(hv_regions) > 0

    # Cell annotations are taken from .obs of the AnnData file.
    output_dir = tmp_path / "dars"
    _run(
        "dars",
        "compute_dars",
        "-a",
        adata_file,
        "--hv-regions-bed",
        hv_bed,
        "--output-dir",
        output_dir,
    )
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "DARs_fg_high_vs_bg_others.bed",
        "DARs_fg_low_vs_bg_others.bed",
    ]
    dars_high = (output_dir / "DARs_fg_high_vs_bg_others.bed").read_text().splitlines()
    assert len(dars_high) > 0


def test_dars_missing_annotation_column(adata_file, tmp_path):
    result = subprocess.run(
        [
            *CLI,
            "dars",
            "compute_dars",
            "-a",
            str(adata_file),
            "--annotation-column",
            "missing",
            "--output-dir",
            str(tmp_path / "dars"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'Column "missing" not found' in result.stderr


@pytest.mark.parametrize("target", ["cell", "region"])
def test_binarize(adata_file, tmp_path, topic_model, target):
    from pycisTopic.topic_binarization import binarize_topics

    region_topic, cell_topic, region_names, cell_names = topic_model
    out_dir = tmp_path / "binarized"
    _run(
        "topic_modeling", "mallet", "binarize",
        "-a", target, "-m", "ntop", "-n", 10, "-i", adata_file, "-p", out_dir,
    )
    prob, names = (
        (cell_topic.T, cell_names) if target == "cell" else (region_topic, region_names)
    )
    expected, _, _ = binarize_topics(prob, names, method="ntop", ntop=10)
    n_topics = prob.shape[1]
    assert len((out_dir / f"{target}_thresholds.tsv").read_text().splitlines()) == n_topics
    for topic_idx, expected_names in enumerate(expected):
        suffix = "txt" if target == "cell" else "bed"
        lines = (
            (out_dir / f"{target}_Topic_{topic_idx + 1}_binarized.{suffix}")
            .read_text()
            .splitlines()
        )
        if target == "cell":
            names_in_file = [line.split("\t")[0] for line in lines]
        else:
            names_in_file = [
                "{}:{}-{}".format(*line.split("\t")[:3]) for line in lines
            ]
        assert names_in_file == expected_names


def test_binarize_aucell(adata_file, tmp_path):
    out_dir = tmp_path / "binarized"
    _run(
        "topic_modeling", "mallet", "binarize",
        "-a", "cell", "-m", "aucell", "-i", adata_file, "-p", out_dir,
    )
    assert (out_dir / "cell_Topic_1_binarized.txt").read_text()


def test_topic_qc(adata_file, tmp_path):
    output = tmp_path / "with_qc.h5ad"
    table = tmp_path / "topics.tsv"
    _run(
        "topic_qc", "-a", adata_file, "-o", output, "--annotation", "cell_type",
        "--table", table,
    )  # fmt: skip
    topics = anndata.read_h5ad(output).uns["pycistopic"]["topics"]
    assert {"Gini_index", "cell_type", "is_general"} <= set(topics.columns)
    table_df = pd.read_csv(table, sep="\t", index_col=0)
    assert list(table_df.index) == list(topics.index)


def test_topic_qc_missing_annotation(adata_file, tmp_path):
    result = subprocess.run(
        [*CLI, "topic_qc", "-a", str(adata_file), "-o", str(tmp_path / "x.h5ad"),
         "--annotation", "missing"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert result.returncode != 0
    assert 'Column "missing" not found' in result.stderr


def test_gene_activity(tmp_path, rng):
    import polars as pl

    from pycisTopic.gene_annotation import write_tss_annotation_to_bed

    region_names = [f"chr1:{s}-{s + 500}" for s in range(0, 60_000, 1_000)]
    cells = [f"cell{i}" for i in range(20)]
    adata = create_anndata(
        rng.dirichlet(np.ones(3), size=len(cells)),
        rng.dirichlet(np.ones(len(region_names)), size=3).T,
        cells,
        region_names,
        obs=pd.DataFrame({"cell_type": ["a", "b"] * 10}, index=cells),
    )
    adata.obsm["X_umap"] = rng.random((len(cells), 2))
    adata.write_h5ad(tmp_path / "model.h5ad")
    write_tss_annotation_to_bed(
        pl.DataFrame(
            {
                "Chromosome": ["chr1", "chr1"],
                "Gene_start": [10_000, 30_000],
                "Gene_end": [20_000, 40_000],
                "Gene": ["GeneA", "GeneB"],
                "Score": [".", "."],
                "Strand": ["+", "-"],
                "TSS_start": [10_000, 39_999],
                "TSS_end": [10_001, 40_000],
                "Transcript_type": ["protein_coding"] * 2,
                "Ensembl_gene_id": ["ENSG1", "ENSG2"],
            }
        ),
        str(tmp_path / "tss.bed"),
    )
    (tmp_path / "chrom.sizes").write_text("chr1\t100000\n")
    _run(
        "gene_activity", "-a", tmp_path / "model.h5ad", "-t", tmp_path / "tss.bed",
        "-c", tmp_path / "chrom.sizes", "-o", tmp_path / "ga.h5ad",
        "--weights", tmp_path / "weights.tsv", "--upstream", 1000, 5000,
    )  # fmt: skip
    gene_activity = anndata.read_h5ad(tmp_path / "ga.h5ad")
    assert list(gene_activity.var_names) == ["GeneA", "GeneB"]
    assert list(gene_activity.obs_names) == cells
    assert "X_umap" in gene_activity.obsm
    assert gene_activity.uns["pycistopic"]["gene_activity"]["upstream"].tolist() == [
        1000,
        5000,
    ]
    weights = pl.read_csv(tmp_path / "weights.tsv", separator="\t")
    assert set(weights["Gene"]) == {"GeneA", "GeneB"}

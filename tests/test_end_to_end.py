"""
End-to-end test of the pycisTopic CLI on a small synthetic dataset.

Runs QC, count matrix creation, Mallet topic modeling, model statistics, AnnData
creation, topic binarization, topic QC, DARs, signature enrichment and gene activity.

Requires Java and Mallet, so it only runs when ``PYCISTOPIC_TEST_MALLET`` is set to
the Mallet executable, e.g.::

    PYCISTOPIC_TEST_MALLET=/path/to/mallet/bin/mallet pytest tests/test_end_to_end.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import anndata
import numpy as np
import pandas as pd
import polars as pl
import pytest

from pycisTopic.gene_annotation import write_tss_annotation_to_bed

MALLET = os.environ.get("PYCISTOPIC_TEST_MALLET")
CLI = [sys.executable, "-m", "pycisTopic.cli.pycistopic"]
SAMPLES = ("sampleA", "sampleB")
CELL_TYPES = "ABC"

pytestmark = pytest.mark.skipif(
    not MALLET, reason="PYCISTOPIC_TEST_MALLET (Mallet executable) is not set."
)


def _make_dataset(path, rng):
    """Synthetic data: 3 cell types with specific regions and shared promoters."""
    chrom_sizes = {"chr1": 2_000_000, "chr2": 1_000_000}
    (path / "chrom.sizes").write_text(
        "".join(f"{c}\t{s}\n" for c, s in chrom_sizes.items())
    )

    genes = []
    for i in range(40):
        chrom = "chr1" if i < 26 else "chr2"
        start = int(rng.integers(20_000, chrom_sizes[chrom] - 60_000))
        end = start + int(rng.integers(2_000, 40_000))
        strand = "+" if i % 2 else "-"
        tss = start if strand == "+" else end - 1
        genes.append(
            (chrom, start, end, f"Gene{i}", ".", strand, tss, tss + 1,
             "protein_coding", f"ENSG{i}")
        )  # fmt: skip
    write_tss_annotation_to_bed(
        pl.DataFrame(
            genes,
            schema=[
                "Chromosome", "Gene_start", "Gene_end", "Gene", "Score", "Strand",
                "TSS_start", "TSS_end", "Transcript_type", "Ensembl_gene_id",
            ],
            orient="row",
        ),
        str(path / "tss.bed"),
    )  # fmt: skip

    promoters = [(g[0], g[6] - 250, g[6] + 250) for g in genes]
    specific = {}
    for cell_type in CELL_TYPES:
        specific[cell_type] = []
        for _ in range(150):
            chrom = "chr1" if rng.random() < 0.66 else "chr2"
            start = int(rng.integers(1_000, chrom_sizes[chrom] - 1_000))
            specific[cell_type].append((chrom, start, start + 500))
    regions = sorted(set(promoters).union(*specific.values()))
    (path / "regions.bed").write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in regions))

    metadata = []
    for sample in SAMPLES:
        fragments, barcodes = [], []
        for cell_type in CELL_TYPES:
            for i in range(40):
                cb = f"{cell_type}{sample[-1]}{i:03d}-1"
                barcodes.append(cb)
                metadata.append((f"{cb[:-2]}___{sample}", cell_type))
                for _ in range(int(rng.integers(400, 900))):
                    u = rng.random()
                    if u < 0.55:
                        c, s, e = specific[cell_type][rng.integers(150)]
                    elif u < 0.8:
                        c, s, e = promoters[rng.integers(len(promoters))]
                    else:
                        c = "chr1" if rng.random() < 0.66 else "chr2"
                        s = int(rng.integers(0, chrom_sizes[c] - 600))
                        e = s + 500
                    start = max(int(rng.integers(s - 100, e - 50)), 0)
                    end = start + int(rng.integers(50, 400))
                    fragments.append((c, start, end, cb, 1))
        fragments.sort()
        (path / f"{sample}.fragments.tsv").write_text(
            "".join("\t".join(map(str, f)) + "\n" for f in fragments)
        )
        (path / f"{sample}.barcodes.tsv").write_text("".join(f"{cb}\n" for cb in barcodes))
    (path / "sample_fragments.tsv").write_text(
        "sample\tfragments_filename\n"
        + "".join(f"{s}\t{path / s}.fragments.tsv\n" for s in SAMPLES)
    )
    (path / "sample_barcodes.tsv").write_text(
        "sample\tcell_barcodes_filename\n"
        + "".join(f"{s}\t{path / s}.barcodes.tsv\n" for s in SAMPLES)
    )
    return pd.DataFrame(metadata, columns=["cell_barcode", "cell_type"]).set_index(
        "cell_barcode"
    )


def _run(*args, cwd):
    result = subprocess.run(
        [*CLI, *map(str, args)], capture_output=True, text=True, cwd=cwd
    )
    assert result.returncode == 0, f"{' '.join(map(str, args))}\n{result.stderr}"


def test_end_to_end(tmp_path):
    metadata = _make_dataset(tmp_path, np.random.default_rng(0))

    def run(*args):
        _run(*args, cwd=tmp_path)

    for sample in SAMPLES:
        run(
            "qc", "run", "-f", f"{sample}.fragments.tsv", "-r", "regions.bed",
            "-t", "tss.bed", "-o", f"qc/{sample}", "--tss_flank_window", 1000,
        )  # fmt: skip
    run(
        "count_matrix", "create", "-f", "sample_fragments.tsv",
        "-c", "sample_barcodes.tsv", "-r", "regions.bed", "-o", "matrix", "-d", "-1",
    )  # fmt: skip
    run(
        "topic_modeling", "mallet", "create_corpus_with_mallet",
        "-i", "matrix.matrix.mtx", "-o", "corpus", "-b", MALLET, "-m", 1,
    )  # fmt: skip
    run(
        "topic_modeling", "mallet", "run", "-i", "corpus", "-o", "model",
        "-t", 3, "-p", 2, "-n", 200, "-s", 555, "-m", 1, "-b", MALLET,
    )  # fmt: skip
    run("topic_modeling", "mallet", "stats", "-i", "matrix.matrix.mtx", "-o", "model", "-t", 3)
    run(
        "topic_modeling", "mallet", "create_anndata", "-i", "matrix.matrix.mtx",
        "-o", "model", "-t", 3, "--qc_dir", "qc", "-d", "-1", "--store_matrix",
    )  # fmt: skip

    adata = anndata.read_h5ad(tmp_path / "model.3_topics.h5ad")
    assert adata.shape[0] == len(metadata)
    assert adata.obs["tss_enrichment"].notna().all()
    assert set(adata.uns["pycistopic"]["model"]["metrics"]) >= {"loglikelihood"}
    adata.obs = adata.obs.join(metadata)
    adata.write_h5ad(tmp_path / "model.3_topics.h5ad")

    # Each topic corresponds to one cell type.
    cell_topic = pd.DataFrame(adata.obsm["X_topic"], index=adata.obs_names)
    mean_per_cell_type = cell_topic.groupby(adata.obs["cell_type"].values).mean()
    assert sorted(mean_per_cell_type.to_numpy().argmax(axis=1)) == [0, 1, 2]
    assert mean_per_cell_type.to_numpy().max(axis=1).min() > 0.5

    model = "model.3_topics.h5ad"
    run("topic_modeling", "mallet", "binarize", "-a", "region", "-m", "otsu", "-i", model, "-p", "bin")
    run("topic_modeling", "mallet", "binarize", "-a", "cell", "-m", "aucell", "-i", model, "-p", "bin")
    run("topic_qc", "-a", model, "-o", model, "--annotation", "cell_type", "--table", "topics.tsv")
    topics = pd.read_csv(tmp_path / "topics.tsv", sep="\t", index_col=0)
    assert sorted(topics["cell_type"]) == list(CELL_TYPES)

    run("dars", "compute_hv_regions", "-a", model, "--output-bed", "hv.bed")
    run(
        "dars", "compute_dars", "-a", model, "--hv-regions-bed", "hv.bed",
        "--annotation-column", "cell_type", "--output-dir", "dars",
    )  # fmt: skip
    for cell_type in CELL_TYPES:
        assert (tmp_path / "dars" / f"DARs_fg_{cell_type}_vs_bg_others.bed").read_text()

    topic_of_a = int(mean_per_cell_type.loc["A"].to_numpy().argmax()) + 1
    run(
        "signature_enrichment", "-a", model, "-n", "topic_of_A",
        "-s", f"bin/region_Topic_{topic_of_a}_binarized.bed", "-o", "signatures.h5ad",
        "--auc-threshold", 0.5,
    )  # fmt: skip
    signatures = anndata.read_h5ad(tmp_path / "signatures.h5ad")
    auc = pd.Series(signatures.obsm["AUCell"][:, 0], index=signatures.obs_names)
    auc_per_cell_type = auc.groupby(signatures.obs["cell_type"].values).mean()
    assert auc_per_cell_type.idxmax() == "A"

    run("gene_activity", "-a", model, "-t", "tss.bed", "-c", "chrom.sizes", "-o", "ga.h5ad")
    gene_activity = anndata.read_h5ad(tmp_path / "ga.h5ad")
    assert gene_activity.shape == (len(metadata), 40)
    assert np.isfinite(gene_activity.X).all()

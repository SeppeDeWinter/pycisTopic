from __future__ import annotations

import polars as pl
import pytest

from pycisTopic.gene_annotation import (
    change_chromosome_source_in_bed,
    find_most_likely_chromosome_source_in_bed,
    get_chrom_sizes_and_alias_mapping_from_file,
    read_tss_annotation_from_bed,
    write_tss_annotation_to_bed,
)


@pytest.fixture
def tss_annotation():
    return pl.DataFrame(
        {
            "Chromosome": ["1", "1", "MT"],
            "Gene_start": pl.Series([100, 100, 5], dtype=pl.Int32),
            "Gene_end": pl.Series([900, 900, 50], dtype=pl.Int32),
            "Gene": ["A", "A", "MT-X"],
            "Score": [".", ".", "."],
            "Strand": ["+", "+", "-"],
            "TSS_start": pl.Series([100, 300, 49], dtype=pl.Int32),
            "TSS_end": pl.Series([101, 301, 50], dtype=pl.Int32),
            "Transcript_type": ["protein_coding"] * 3,
            "Ensembl_gene_id": ["ENSG1", "ENSG1", "ENSG2"],
        }
    )


@pytest.fixture
def alias_mapping(tmp_path):
    path = tmp_path / "chrom_alias.tsv"
    path.write_text(
        "# ucsc\tensembl\tgenbank\trefseq\tlength\n"
        "chr1\t1\tCM000663.2\tNC_000001.11\t248956422\n"
        "chrM\tMT\tJ01415.2\tNC_012920.1\t16569\n"
    )
    return get_chrom_sizes_and_alias_mapping_from_file(path)


def test_tss_annotation_roundtrip(tss_annotation, tmp_path):
    path = tmp_path / "tss.bed"
    write_tss_annotation_to_bed(tss_annotation, str(path))
    assert path.read_text().startswith("# Chromosome\tGene_start")
    result = read_tss_annotation_from_bed(str(path))
    assert result.columns == tss_annotation.columns
    assert result.with_columns(pl.col("Chromosome").cast(pl.Utf8)).equals(
        tss_annotation
    )


def test_chrom_alias_file(alias_mapping):
    assert alias_mapping.columns[:4] == ["ucsc", "ensembl", "genbank", "refseq"]


def test_find_most_likely_chromosome_source(alias_mapping, tss_annotation):
    source, stats = find_most_likely_chromosome_source_in_bed(
        alias_mapping, tss_annotation
    )
    assert source == "ensembl"
    assert stats["ensembl"][0] == 2


def test_change_chromosome_source(alias_mapping, tss_annotation):
    tss_annotation = pl.concat(
        [
            tss_annotation,
            tss_annotation.head(1).with_columns(pl.lit("unplaced").alias("Chromosome")),
        ]
    )
    result = change_chromosome_source_in_bed(
        alias_mapping, tss_annotation, "ensembl", "ucsc"
    )
    assert result["Chromosome"].cast(pl.Utf8).to_list() == [
        "chr1",
        "chr1",
        "chrM",
        "unplaced",  # unknown chromosome names are kept
    ]
    assert result.columns == tss_annotation.columns

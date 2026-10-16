from __future__ import annotations

import json
import math

import anndata
import numpy as np
import pytest
import scipy.sparse

from pycisTopic.topic_modeling.create_anndata import create_anndata_from_mallet
from pycisTopic.topic_modeling.mallet_models import LDAMallet, LDAMalletFilenames
from pycisTopic.topic_modeling.stats import (
    calculate_model_evaluation_stats,
    loglikelihood,
)

N_TOPICS, N_REGIONS, N_CELLS = 4, 30, 12


@pytest.fixture
def mallet_output(tmp_path, rng):
    """Fake Mallet output files for a 4 topic model."""
    filenames = LDAMalletFilenames(str(tmp_path / "model"), N_TOPICS)
    cell_topic = rng.dirichlet(np.ones(N_TOPICS), size=N_CELLS)
    with open(filenames.cell_topic_probabilities_txt_filename, "w") as fh:
        for cell_idx, probabilities in enumerate(cell_topic):
            fh.write(
                f"{cell_idx}\tcell{cell_idx}\t"
                + "\t".join(f"{p:.6f}" for p in probabilities)
                + "\n"
            )
    region_topic_counts = rng.integers(0, 20, size=(N_TOPICS, N_REGIONS))
    region_topic_counts[:, 0] = [5, 0, 0, 0]  # region with counts in a single topic
    with open(filenames.region_topic_counts_txt_filename, "w") as fh:
        for region_idx in range(N_REGIONS):
            counts = " ".join(
                f"{topic}:{count}"
                for topic, count in enumerate(region_topic_counts[:, region_idx])
                if count > 0
            )
            fh.write(f"{region_idx} {region_idx} {counts}\n")
    with open(filenames.parameters_json_filename, "w") as fh:
        json.dump(
            {
                "n_topics": N_TOPICS,
                "alpha": 50,
                "alpha_by_topic": True,
                "eta": 0.1,
                "eta_by_topic": False,
            },
            fh,
        )
    LDAMallet.convert_cell_topic_probabilities_txt_to_parquet(
        filenames.cell_topic_probabilities_txt_filename,
        filenames.cell_topic_probabilities_parquet_filename,
    )
    LDAMallet.convert_region_topic_counts_txt_to_parquet(
        filenames.region_topic_counts_txt_filename,
        filenames.region_topic_counts_parquet_filename,
    )
    return filenames, cell_topic, region_topic_counts


def test_filenames():
    filenames = LDAMalletFilenames("out/sample", 10)
    assert filenames.parameters_json_filename == "out/sample.10_topics.parameters.json"
    assert filenames.model_filename_prefix == "out/sample.10_topics.model"
    assert filenames.model_stats_filename == "out/sample.10_topics.stats.json"
    assert (
        filenames.anndata_cell_topic_filename
        == "out/sample.10_topics_cell_topic_adata.h5ad"
    )


class TestMalletOutputConversion:
    def test_cell_topic_probabilities(self, mallet_output):
        filenames, cell_topic, _ = mallet_output
        result = LDAMallet.read_cell_topic_probabilities_parquet_file(
            filenames.cell_topic_probabilities_parquet_filename
        )
        assert result.shape == (N_CELLS, N_TOPICS)
        np.testing.assert_allclose(result, cell_topic, atol=1e-6)

    def test_region_topic_counts(self, mallet_output):
        filenames, _, region_topic_counts = mallet_output
        counts = LDAMallet.read_region_topic_counts_parquet_file(
            filenames.region_topic_counts_parquet_filename
        )
        np.testing.assert_array_equal(counts, region_topic_counts)
        probabilities = (
            LDAMallet.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
                filenames.region_topic_counts_parquet_filename
            )
        )
        # Topic-region distribution: each topic sums to one over all regions.
        assert probabilities.shape == (N_TOPICS, N_REGIONS)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-6)

    def test_read_parameters(self, mallet_output):
        filenames, _, _ = mallet_output
        parameters = LDAMallet.read_parameters_json_filename(
            filenames.parameters_json_filename
        )
        assert parameters["n_topics"] == N_TOPICS


def _reference_loglikelihood(nzw, ndz, alpha, eta):
    """Log likelihood of a LDA model as in the `lda` package (`lda._lda`)."""
    n_docs, n_topics = ndz.shape
    vocab_size = nzw.shape[1]
    ll = 0.0
    ll += (math.lgamma(eta * vocab_size) - vocab_size * math.lgamma(eta)) * n_topics
    for k in range(n_topics):
        ll += sum(math.lgamma(c + eta) for c in nzw[k] if c > 0)
        ll -= math.lgamma(eta * vocab_size + nzw[k].sum())
    ll += (math.lgamma(alpha * n_topics) - n_topics * math.lgamma(alpha)) * n_docs
    for d in range(n_docs):
        ll += sum(math.lgamma(c + alpha) for c in ndz[d] if c > 0)
        ll -= math.lgamma(alpha * n_topics + ndz[d].sum())
    return ll


@pytest.mark.xfail(
    strict=True,
    reason="Bug: loglikelihood overwrites (=) instead of accumulates (+=) the "
    "per word/document terms (also in pycisTopic master).",
)
def test_loglikelihood_matches_reference(rng):
    nzw = rng.integers(0, 10, (3, 8))
    ndz = rng.integers(0, 10, (5, 3))
    assert loglikelihood(nzw, ndz, 0.5, 0.1) == pytest.approx(
        _reference_loglikelihood(nzw, ndz, 0.5, 0.1)
    )


def test_calculate_model_evaluation_stats(mallet_output, rng):
    filenames, _, _ = mallet_output
    binary_matrix = scipy.sparse.csr_matrix(
        (rng.random((N_REGIONS, N_CELLS)) < 0.4).astype(np.int32)
    )
    calculate_model_evaluation_stats(
        binary_matrix, filenames.output_prefix, N_TOPICS, top_topics_coh=2
    )
    with open(filenames.model_stats_filename) as fh:
        metrics = json.load(fh)
    assert set(metrics) == {
        "Arun_2010",
        "Cao_Juan_2009",
        "Mimno_2011",
        "loglikelihood",
        "coherence",
        "marg_topic",
        "assignments",
    }
    assert len(metrics["coherence"]) == N_TOPICS
    assert metrics["Mimno_2011"] == pytest.approx(
        np.mean(sorted(metrics["coherence"])[-2:])
    )
    assert np.sum(metrics["marg_topic"]) == pytest.approx(1.0)
    assert all(math.isfinite(metrics[k]) for k in ("Arun_2010", "Cao_Juan_2009"))


def test_calculate_model_evaluation_stats_topic_mismatch(mallet_output, rng):
    filenames, _, _ = mallet_output
    other = LDAMalletFilenames(filenames.output_prefix, N_TOPICS + 1)
    for src, dst in [
        (filenames.parameters_json_filename, other.parameters_json_filename),
        (filenames.cell_topic_probabilities_parquet_filename, other.cell_topic_probabilities_parquet_filename),
        (filenames.region_topic_counts_parquet_filename, other.region_topic_counts_parquet_filename),
    ]:
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            fout.write(fin.read())
    with pytest.raises(ValueError, match="Number of topics does not match"):
        calculate_model_evaluation_stats(
            scipy.sparse.csr_matrix(np.ones((N_REGIONS, N_CELLS))),
            filenames.output_prefix,
            N_TOPICS + 1,
        )


def test_create_anndata_from_mallet(mallet_output):
    filenames, cell_topic, region_topic_counts = mallet_output
    cell_barcodes = [f"cell{i}" for i in range(N_CELLS)]
    region_ids = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(N_REGIONS)]
    create_anndata_from_mallet(filenames.output_prefix, N_TOPICS, cell_barcodes, region_ids)
    cell_adata = anndata.read_h5ad(filenames.anndata_cell_topic_filename)
    region_adata = anndata.read_h5ad(filenames.anndata_region_topic_filename)
    assert list(cell_adata.obs_names) == cell_barcodes
    assert list(cell_adata.var_names) == [f"Topic{i}" for i in range(1, N_TOPICS + 1)]
    np.testing.assert_allclose(cell_adata.X, cell_topic, atol=1e-6)
    assert list(region_adata.obs_names) == region_ids
    np.testing.assert_allclose(
        region_adata.X, (region_topic_counts / region_topic_counts.sum(axis=1)[:, None]).T,
        rtol=1e-6,
    )

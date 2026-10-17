from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from pycisTopic.topic_modeling.lda import (
    LDACorpus,
    _log_likelihood,
    run_lda_topic_modeling,
    train_lda,
)
from pycisTopic.topic_modeling.mallet_models import LDAMallet, LDAMalletFilenames

N_TOPICS = 4
N_REGIONS = 400
N_CELLS = 300


@pytest.fixture(scope="module")
def synthetic():
    """Binary region x cell matrix generated from LDA with known topics."""
    rng = np.random.default_rng(0)
    region_topic = np.full((N_TOPICS, N_REGIONS), 0.2 / N_REGIONS)
    size = N_REGIONS // N_TOPICS
    for k in range(N_TOPICS):
        region_topic[k, k * size : (k + 1) * size] += 0.8 / size
    region_topic /= region_topic.sum(axis=1, keepdims=True)
    cell_topic = rng.dirichlet(np.full(N_TOPICS, 0.3), size=N_CELLS)
    rows, cols = [], []
    for cell in range(N_CELLS):
        regions = np.unique(
            rng.choice(N_REGIONS, size=120, p=cell_topic[cell] @ region_topic)
        )
        rows.append(regions)
        cols.append(np.full(regions.shape[0], cell))
    rows = np.concatenate(rows)
    matrix = sp.coo_array(
        (np.ones(rows.shape[0], dtype=np.int32), (rows, np.concatenate(cols))),
        shape=(N_REGIONS, N_CELLS),
    )
    return matrix, region_topic, cell_topic


def _matched_correlations(expected, result):
    corr = np.corrcoef(expected, result)[: expected.shape[0], expected.shape[0] :]
    rows, cols = linear_sum_assignment(-corr)
    return corr[rows, cols], cols


@pytest.mark.parametrize("n_partitions", [1, 3])
def test_corpus(synthetic, n_partitions):
    matrix = sp.csr_array(synthetic[0])
    corpus = LDACorpus(matrix, n_partitions=n_partitions)
    assert corpus.n_tokens == matrix.nnz
    assert corpus.block_run_offsets.shape[0] == n_partitions**2 + 1
    assert corpus.run_starts[-1] == corpus.n_tokens
    # All tokens (region, cell) are present once (in renumbered indices).
    words = np.repeat(corpus.run_words, np.diff(corpus.run_starts))
    tokens = set(
        zip(
            corpus.word_order[words].tolist(),
            corpus.doc_order[corpus.docs].tolist(),
            strict=True,
        )
    )
    assert tokens == set(
        zip(matrix.tocoo().row.tolist(), matrix.tocoo().col.tolist(), strict=True)
    )


@pytest.mark.parametrize("n_partitions", [1, 3])
def test_recovers_topics(synthetic, n_partitions):
    matrix, region_topic, cell_topic = synthetic
    corpus = LDACorpus(matrix, n_partitions=n_partitions, random_seed=1)
    model = train_lda(corpus, n_topics=N_TOPICS, iterations=200, random_seed=1)

    # Counts are consistent with the matrix.
    np.testing.assert_array_equal(
        model.doc_topic_counts.sum(axis=1), np.asarray(matrix.sum(axis=0)).ravel()
    )
    np.testing.assert_array_equal(
        model.word_topic_counts.sum(axis=1), np.asarray(matrix.sum(axis=1)).ravel()
    )

    region_topic_counts = model.region_topic_counts
    assert region_topic_counts.shape == (N_TOPICS, N_REGIONS)
    correlations, order = _matched_correlations(
        region_topic,
        region_topic_counts / region_topic_counts.sum(axis=1, keepdims=True),
    )
    assert correlations.min() > 0.95

    cell_topic_probabilities = model.cell_topic_probabilities
    np.testing.assert_allclose(cell_topic_probabilities.sum(axis=1), 1.0)
    assert (
        np.corrcoef(cell_topic_probabilities[:, order].ravel(), cell_topic.ravel())[
            0, 1
        ]
        > 0.9
    )


def test_reproducible(synthetic):
    corpus = LDACorpus(synthetic[0], n_partitions=3, random_seed=5)
    models = [
        train_lda(corpus, n_topics=N_TOPICS, iterations=20, random_seed=5, n_threads=t)
        for t in (1, 3)
    ]
    # Same random seed and number of partitions: same result for any number of
    # threads.
    np.testing.assert_array_equal(
        models[0].doc_topic_counts, models[1].doc_topic_counts
    )
    other_seed = train_lda(corpus, n_topics=N_TOPICS, iterations=20, random_seed=6)
    assert not np.array_equal(models[0].doc_topic_counts, other_seed.doc_topic_counts)


def test_log_likelihood_matches_formula():
    rng = np.random.default_rng(3)
    doc_topic_counts = rng.integers(0, 5, size=(6, 3)).astype(np.int32)
    word_topic_counts = rng.integers(0, 4, size=(8, 3)).astype(np.int32)
    topic_counts = word_topic_counts.sum(axis=0).astype(np.int64)
    alpha, beta = 0.5, 0.1
    n_words, n_topics = word_topic_counts.shape

    expected = 0.0
    for counts in doc_topic_counts:
        expected += math.lgamma(alpha * n_topics) - math.lgamma(
            alpha * n_topics + counts.sum()
        )
        expected += sum(math.lgamma(alpha + c) - math.lgamma(alpha) for c in counts)
    for k in range(n_topics):
        expected += math.lgamma(beta * n_words) - math.lgamma(
            beta * n_words + topic_counts[k]
        )
        expected += sum(
            math.lgamma(beta + c) - math.lgamma(beta) for c in word_topic_counts[:, k]
        )

    assert _log_likelihood(
        doc_topic_counts, word_topic_counts, topic_counts, alpha, beta
    ) == pytest.approx(expected)


def test_log_likelihood_increases(synthetic):
    corpus = LDACorpus(synthetic[0], n_partitions=2)
    model = train_lda(
        corpus, n_topics=N_TOPICS, iterations=100, log_likelihood_interval=10
    )
    log_likelihoods = [ll for _, ll in model.log_likelihoods]
    assert len(log_likelihoods) == 10
    assert log_likelihoods[-1] > log_likelihoods[0]


def test_run_lda_topic_modeling_output_files(synthetic, tmp_path):
    prefix = str(tmp_path / "model")
    run_lda_topic_modeling(
        synthetic[0], prefix, n_topics=[2, N_TOPICS], iterations=10, n_threads=2
    )
    for n_topics in (2, N_TOPICS):
        filenames = LDAMalletFilenames(prefix, n_topics)
        cell_topic = LDAMallet.read_cell_topic_probabilities_parquet_file(
            filenames.cell_topic_probabilities_parquet_filename
        )
        assert cell_topic.shape == (N_CELLS, n_topics)
        region_topic = LDAMallet.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
            filenames.region_topic_counts_parquet_filename
        )
        assert region_topic.shape == (n_topics, N_REGIONS)
        np.testing.assert_allclose(region_topic.sum(axis=1), 1.0, rtol=1e-5)
        with open(filenames.parameters_json_filename) as fh:
            parameters = json.load(fh)
        assert parameters["n_topics"] == n_topics
        assert parameters["alpha"] == 50
        assert parameters["eta"] == 0.1
        assert pl.read_parquet(
            filenames.cell_topic_probabilities_parquet_filename
        ).columns == ["cell_topic_probabilities"]


def test_invalid_number_of_topics(synthetic):
    corpus = LDACorpus(synthetic[0], n_partitions=1)
    with pytest.raises(ValueError, match="Number of topics"):
        train_lda(corpus, n_topics=0)


def test_stats_log_likelihood_matches_sampler(synthetic, tmp_path):
    from pycisTopic.topic_modeling.stats import calculate_model_evaluation_stats

    prefix = str(tmp_path / "model")
    run_lda_topic_modeling(
        synthetic[0], prefix, n_topics=N_TOPICS, iterations=20, n_threads=2
    )
    calculate_model_evaluation_stats(
        sp.csr_matrix(synthetic[0]), prefix, N_TOPICS, top_topics_coh=2
    )
    filenames = LDAMalletFilenames(prefix, N_TOPICS)
    with open(filenames.model_stats_filename) as fh:
        stats = json.load(fh)
    with open(filenames.parameters_json_filename) as fh:
        parameters = json.load(fh)
    # Cell-topic counts are recovered exactly from the cell-topic probabilities.
    assert stats["loglikelihood"] == pytest.approx(
        parameters["log_likelihoods"][-1][1], rel=1e-6
    )

"""
LDA topic modeling with a parallel collapsed Gibbs sampler (numba).

The model and the sampler are the same as Mallet's (``mallet train-topics`` without
hyperparameter optimization): latent Dirichlet allocation with a symmetric Dirichlet
prior ``alpha`` on cell-topic distributions and ``eta`` (``beta`` in Mallet) on
topic-region distributions, inferred with collapsed Gibbs sampling.

Parallelization: cells and regions are both divided in ``n_partitions`` groups, which
divides the tokens (accessible regions in cells) in ``n_partitions x n_partitions``
blocks. Each iteration consists of ``n_partitions`` rounds; in each round, all
partitions sample the tokens of a different block in parallel. Blocks sampled at the
same time share no cells and no regions, so cell-topic and region-topic counts are
always up-to-date and only the total number of tokens per topic is synchronized after
each round. This is closer to sequential Gibbs sampling than Mallet's parallel
sampler, in which region-topic counts are only synchronized after each iteration.
Each block of tokens only touches a small part of the count matrices, which makes
sampling cache friendly and fast.

Results are reproducible for the same random seed and number of partitions.

"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import TYPE_CHECKING

import numba
import numpy as np
import polars as pl
import scipy.sparse

from pycisTopic.topic_modeling.mallet_models import LDAMalletFilenames

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


@numba.njit(inline="always")
def _next_random(x: np.uint64) -> tuple[np.uint64, float]:
    """Get next state and random number in [0, 1) of a xorshift64* generator."""
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    # 53 random bits fit in a signed 64-bit integer (faster conversion to float).
    return x, np.float64(
        np.int64((x * np.uint64(2685821657736338717)) >> np.uint64(11))
    ) * (1.0 / 9007199254740992.0)


@numba.njit(parallel=True, cache=True)
def _initialize_topics(
    topics: npt.NDArray[np.unsignedinteger],
    run_starts: npt.NDArray[np.int64],
    block_run_offsets: npt.NDArray[np.int64],
    n_partitions: int,
    n_topics: int,
    rng_state: npt.NDArray[np.uint64],
) -> None:
    """Assign a random topic to each token."""
    for doc_partition in numba.prange(n_partitions):
        start = run_starts[block_run_offsets[doc_partition * n_partitions]]
        end = run_starts[block_run_offsets[(doc_partition + 1) * n_partitions]]
        state = rng_state[doc_partition]
        for i in range(start, end):
            state, u = _next_random(state)
            topics[i] = min(int(u * n_topics), n_topics - 1)
        rng_state[doc_partition] = state


@numba.njit(parallel=True, cache=True)
def _copy_per_doc_partition(
    array: npt.NDArray,
    partition_starts: npt.NDArray[np.int64],
) -> npt.NDArray:
    """
    Copy array (first axis) in parallel per partition.

    With threads bound to cores, memory pages of each partition are allocated close
    to the thread that processes the partition (first touch), which reduces memory
    access times on multi-socket (NUMA) machines.

    """
    copy = np.empty_like(array)
    n_partitions = partition_starts.shape[0] - 1
    for partition in numba.prange(n_partitions):
        for i in range(partition_starts[partition], partition_starts[partition + 1]):
            copy[i] = array[i]
    return copy


@numba.njit(cache=True)
def _count_topics(
    run_words: npt.NDArray[np.int32],
    run_starts: npt.NDArray[np.int64],
    docs: npt.NDArray[np.int32],
    topics: npt.NDArray[np.unsignedinteger],
    n_docs: int,
    n_words: int,
    n_topics: int,
):
    """Count topic assignments per doc, per word and in total."""
    doc_topic_counts = np.zeros((n_docs, n_topics), dtype=np.int32)
    word_topic_counts = np.zeros((n_words, n_topics), dtype=np.int32)
    topic_counts = np.zeros(n_topics, dtype=np.int64)
    for run in range(run_words.shape[0]):
        word = run_words[run]
        for i in range(run_starts[run], run_starts[run + 1]):
            doc_topic_counts[docs[i], topics[i]] += 1
            word_topic_counts[word, topics[i]] += 1
            topic_counts[topics[i]] += 1
    return doc_topic_counts, word_topic_counts, topic_counts


@numba.njit(inline="always")
def _sample_block(
    run_words: npt.NDArray[np.int32],
    run_starts: npt.NDArray[np.int64],
    docs: npt.NDArray[np.int32],
    topics: npt.NDArray[np.unsignedinteger],
    first_run: int,
    last_run: int,
    doc_topic_counts: npt.NDArray[np.int32],
    word_topic_counts: npt.NDArray[np.int32],
    topic_counts: npt.NDArray[np.int64],
    alpha: float,
    beta: float,
    beta_sum: float,
    rng_state: np.uint64,
) -> np.uint64:
    """
    Resample topics of tokens of runs [first_run, last_run) (tokens of one word).

    ``topic_counts`` is updated in place; returns the new random number generator
    state.

    The probability of topic k for token (doc d, word w) is proportional to:
        (n_dk + alpha) * (n_wk + beta) / (n_k + beta_sum)
    with counts excluding the current token. ``1 / (n_k + beta_sum)`` is calculated
    at the start of the block (``n_k`` changes very little relative to its size
    during a block), which avoids divisions per token.

    """
    n_topics = topic_counts.shape[0]
    inverse_topic_counts = 1.0 / (topic_counts + beta_sum)
    # (n_wk + beta) / (n_k + beta_sum) for the current word.
    word_coefficients = np.empty(n_topics, dtype=np.float64)
    probabilities = np.empty(n_topics, dtype=np.float64)

    for run in range(first_run, last_run):
        word = run_words[run]
        word_counts = word_topic_counts[word]
        for k in range(n_topics):
            word_coefficients[k] = (word_counts[k] + beta) * inverse_topic_counts[k]

        for i in range(run_starts[run], run_starts[run + 1]):
            doc_counts = doc_topic_counts[docs[i]]
            old_topic = topics[i]

            # Unnormalized topic probabilities with counts including the current
            # token (separate loops so they can be vectorized) ...
            for k in range(n_topics):
                probabilities[k] = (doc_counts[k] + alpha) * word_coefficients[k]
            total = 0.0
            for k in range(n_topics):
                total += probabilities[k]
            # ... corrected for the current token (in the old topic).
            old_topic_probability = (
                (doc_counts[old_topic] - 1 + alpha)
                * (word_counts[old_topic] - 1 + beta)
                * inverse_topic_counts[old_topic]
            )
            total += old_topic_probability - probabilities[old_topic]

            rng_state, u = _next_random(rng_state)
            u *= total
            new_topic = 0
            while new_topic < n_topics - 1:
                if new_topic == old_topic:
                    u -= old_topic_probability
                else:
                    u -= probabilities[new_topic]
                if u < 0.0:
                    break
                new_topic += 1

            if new_topic != old_topic:
                # Move token from old to new topic.
                topics[i] = new_topic
                doc_counts[old_topic] -= 1
                word_counts[old_topic] -= 1
                topic_counts[old_topic] -= 1
                word_coefficients[old_topic] = (
                    word_counts[old_topic] + beta
                ) * inverse_topic_counts[old_topic]
                doc_counts[new_topic] += 1
                word_counts[new_topic] += 1
                topic_counts[new_topic] += 1
                word_coefficients[new_topic] = (
                    word_counts[new_topic] + beta
                ) * inverse_topic_counts[new_topic]

    return rng_state


@numba.njit(parallel=True, cache=True, fastmath=True)
def _sample_iteration(
    run_words: npt.NDArray[np.int32],
    run_starts: npt.NDArray[np.int64],
    block_run_offsets: npt.NDArray[np.int64],
    docs: npt.NDArray[np.int32],
    topics: npt.NDArray[np.unsignedinteger],
    n_partitions: int,
    doc_topic_counts: npt.NDArray[np.int32],
    word_topic_counts: npt.NDArray[np.int32],
    topic_counts: npt.NDArray[np.int64],
    alpha: float,
    beta: float,
    beta_sum: float,
    rng_state: npt.NDArray[np.uint64],
) -> None:
    """Run one Gibbs sampling iteration over all tokens."""
    n_topics = topic_counts.shape[0]
    # Changes in number of tokens per topic for each partition (padded to avoid
    # false sharing of cache lines between threads).
    padded_n_topics = (n_topics + 7) // 8 * 8 + 8
    topic_count_changes = np.zeros((n_partitions, padded_n_topics), dtype=np.int64)

    for round_idx in range(n_partitions):
        # Blocks in the same round have no docs and no words in common.
        for doc_partition in numba.prange(n_partitions):
            word_partition = (doc_partition + round_idx) % n_partitions
            block = doc_partition * n_partitions + word_partition
            # Thread-local copies.
            local_topic_counts = topic_counts.copy()
            rng_state[doc_partition] = _sample_block(
                run_words,
                run_starts,
                docs,
                topics,
                block_run_offsets[block],
                block_run_offsets[block + 1],
                doc_topic_counts,
                word_topic_counts,
                local_topic_counts,
                alpha,
                beta,
                beta_sum,
                rng_state[doc_partition],
            )
            topic_count_changes[doc_partition, :n_topics] = (
                local_topic_counts - topic_counts
            )

        # Synchronize total number of tokens per topic.
        for doc_partition in range(n_partitions):
            topic_counts += topic_count_changes[doc_partition, :n_topics]


@numba.njit(cache=True)
def _log_likelihood(
    doc_topic_counts: npt.NDArray[np.int32],
    word_topic_counts: npt.NDArray[np.int32],
    topic_counts: npt.NDArray[np.int64],
    alpha: float,
    beta: float,
) -> float:
    """Get log likelihood of the model (same formula as Mallet's modelLogLikelihood)."""
    n_docs, n_topics = doc_topic_counts.shape
    n_words = word_topic_counts.shape[0]
    alpha_sum = alpha * n_topics

    log_likelihood = 0.0
    lgamma_alpha = math.lgamma(alpha)
    for d in range(n_docs):
        doc_length = 0
        for k in range(n_topics):
            count = doc_topic_counts[d, k]
            if count > 0:
                log_likelihood += math.lgamma(alpha + count) - lgamma_alpha
                doc_length += count
        log_likelihood += math.lgamma(alpha_sum) - math.lgamma(alpha_sum + doc_length)

    lgamma_beta = math.lgamma(beta)
    for w in range(n_words):
        for k in range(n_topics):
            count = word_topic_counts[w, k]
            if count > 0:
                log_likelihood += math.lgamma(beta + count) - lgamma_beta
    for k in range(n_topics):
        log_likelihood -= math.lgamma(beta * n_words + topic_counts[k])
    log_likelihood += n_topics * math.lgamma(beta * n_words)

    return log_likelihood


def _partition(
    tokens_per_item: npt.NDArray[np.int64],
    n_partitions: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int32]:
    """Divide items in partitions with a similar number of tokens (random order)."""
    order = rng.permutation(tokens_per_item.shape[0])
    cumulative_tokens = np.cumsum(tokens_per_item[order])
    total_tokens = max(int(cumulative_tokens[-1]), 1) if order.shape[0] else 1
    partition_in_order = np.minimum(
        (cumulative_tokens - 1) * n_partitions // total_tokens, n_partitions - 1
    ).astype(np.int32)
    partitions = np.empty(tokens_per_item.shape[0], dtype=np.int32)
    partitions[order] = partition_in_order
    return partitions


@numba.njit(cache=True)
def _create_blocks(
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32],
    word_partition_starts: npt.NDArray[np.int64],
    doc_partition_starts: npt.NDArray[np.int64],
    n_partitions: int,
):
    """
    Group tokens of a CSR matrix (words x docs, sorted indices) in blocks.

    Words and docs are numbered so that each partition has consecutive indices.
    Returns docs per token, word and first token of each run of tokens with the same
    word (``run_starts`` has one extra element) and the first run of each block
    (``block_run_offsets``, one extra element).

    """
    n_words = indptr.shape[0] - 1
    n_tokens = indptr[n_words]

    # First token of each doc partition in each word (row): docs are sorted.
    segment_starts = np.empty((n_words, n_partitions + 1), dtype=np.int64)
    for word in range(n_words):
        i = indptr[word]
        end = indptr[word + 1]
        for doc_partition in range(n_partitions):
            while i < end and indices[i] < doc_partition_starts[doc_partition]:
                i += 1
            segment_starts[word, doc_partition] = i
        segment_starts[word, n_partitions] = end

    # Count runs per block.
    n_runs = 0
    block_run_offsets = np.zeros(n_partitions * n_partitions + 1, dtype=np.int64)
    for doc_partition in range(n_partitions):
        for word_partition in range(n_partitions):
            block = doc_partition * n_partitions + word_partition
            for word in range(
                word_partition_starts[word_partition],
                word_partition_starts[word_partition + 1],
            ):
                if (
                    segment_starts[word, doc_partition + 1]
                    > segment_starts[word, doc_partition]
                ):
                    n_runs += 1
            block_run_offsets[block + 1] = n_runs

    docs = np.empty(n_tokens, dtype=np.int32)
    run_words = np.empty(n_runs, dtype=np.int32)
    run_starts = np.empty(n_runs + 1, dtype=np.int64)
    run = 0
    position = 0
    for doc_partition in range(n_partitions):
        for word_partition in range(n_partitions):
            for word in range(
                word_partition_starts[word_partition],
                word_partition_starts[word_partition + 1],
            ):
                start = segment_starts[word, doc_partition]
                end = segment_starts[word, doc_partition + 1]
                if end > start:
                    run_words[run] = word
                    run_starts[run] = position
                    for i in range(start, end):
                        docs[position] = indices[i]
                        position += 1
                    run += 1
    run_starts[n_runs] = position

    return docs, run_words, run_starts, block_run_offsets


class LDACorpus:
    """
    Tokens of a binary region x cell accessibility matrix, ordered for sampling.

    Tokens are grouped in ``n_partitions x n_partitions`` blocks (cell partition x
    region partition) and stored as runs of tokens of the same region (sorted by
    cell) within each block.

    Cells and regions are renumbered so cells and regions of the same partition have
    consecutive indices (``doc_order`` and ``word_order`` give the original index of
    each cell and region). This keeps count matrix rows of different partitions in
    different cache lines.

    """

    def __init__(
        self,
        binary_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
        n_partitions: int,
        random_seed: int = 555,
    ):
        """
        Create corpus from a binary region x cell matrix.

        Parameters
        ----------
        binary_matrix
            Binary accessibility matrix (regions x cells). Non-zero entries are used as
            tokens (counts are ignored).
        n_partitions
            Number of cell and region partitions (number of blocks sampled in
            parallel).
        random_seed
            Random seed used to assign cells and regions to partitions.

        """
        matrix = scipy.sparse.csr_array(binary_matrix)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        self.n_words, self.n_docs = matrix.shape
        self.n_partitions = n_partitions

        rng = np.random.default_rng(random_seed)
        doc_partitions = _partition(
            np.bincount(matrix.indices, minlength=self.n_docs), n_partitions, rng
        )
        word_partitions = _partition(np.diff(matrix.indptr), n_partitions, rng)

        # Renumber cells and regions: order by partition (stable).
        self.doc_order = np.argsort(doc_partitions, kind="stable").astype(np.int32)
        self.word_order = np.argsort(word_partitions, kind="stable").astype(np.int32)
        doc_partition_starts = np.searchsorted(
            doc_partitions[self.doc_order], np.arange(n_partitions + 1)
        ).astype(np.int64)
        self.doc_partition_starts = doc_partition_starts
        word_partition_starts = np.searchsorted(
            word_partitions[self.word_order], np.arange(n_partitions + 1)
        ).astype(np.int64)

        renumbered_matrix = matrix[self.word_order][:, self.doc_order]
        del matrix
        renumbered_matrix = scipy.sparse.csr_array(renumbered_matrix)
        renumbered_matrix.sort_indices()

        self.docs, self.run_words, self.run_starts, self.block_run_offsets = (
            _create_blocks(
                renumbered_matrix.indptr.astype(np.int64),
                renumbered_matrix.indices.astype(np.int32),
                word_partition_starts,
                doc_partition_starts,
                n_partitions,
            )
        )

    @property
    def n_tokens(self) -> int:
        return self.docs.shape[0]

    @property
    def token_partition_starts(self) -> npt.NDArray[np.int64]:
        """First token of each cell partition (one extra element)."""
        return self.run_starts[self.block_run_offsets[:: self.n_partitions]].astype(
            np.int64
        )


class LDAModel:
    """LDA model trained with the parallel collapsed Gibbs sampler."""

    def __init__(
        self,
        n_topics: int,
        alpha: float,
        beta: float,
        doc_topic_counts: npt.NDArray[np.int32],
        word_topic_counts: npt.NDArray[np.int32],
        log_likelihoods: list[tuple[int, float]],
    ):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.doc_topic_counts = doc_topic_counts
        self.word_topic_counts = word_topic_counts
        self.log_likelihoods = log_likelihoods

    @property
    def cell_topic_probabilities(self) -> npt.NDArray[np.float64]:
        """
        Cell-topic probabilities (cells x topics), like Mallet ``--output-doc-topics``.

        ``(n_dk + alpha) / (n_d + n_topics * alpha)``

        """
        doc_lengths = self.doc_topic_counts.sum(axis=1, keepdims=True)
        return (self.doc_topic_counts + self.alpha) / (
            doc_lengths + self.n_topics * self.alpha
        )

    @property
    def region_topic_counts(self) -> npt.NDArray[np.int32]:
        """Region-topic counts (topics x regions), like Mallet word-topic counts."""
        return np.ascontiguousarray(self.word_topic_counts.T)


def train_lda(
    corpus: LDACorpus,
    n_topics: int,
    alpha: float = 50,
    alpha_by_topic: bool = True,
    eta: float = 0.1,
    eta_by_topic: bool = False,
    iterations: int = 150,
    random_seed: int = 555,
    n_threads: int | None = None,
    log_likelihood_interval: int = 50,
) -> LDAModel:
    """
    Train LDA model with the parallel collapsed Gibbs sampler.

    Parameters
    ----------
    corpus
        Corpus (see :class:`LDACorpus`).
    n_topics
        Number of topics.
    alpha
        Symmetric Dirichlet hyperparameter for cell-topic distributions.
        Default: 50.
    alpha_by_topic
        Divide ``alpha`` by the number of topics. Default: ``True``.
    eta
        Symmetric Dirichlet hyperparameter for topic-region distributions.
        Default: 0.1.
    eta_by_topic
        Divide ``eta`` by the number of topics. Default: ``False``.
    iterations
        Number of Gibbs sampling iterations. Default: 150.
    random_seed
        Random seed. Results are reproducible for the same random seed and number of
        corpus partitions.
    n_threads
        Number of threads (default: number of corpus partitions).
    log_likelihood_interval
        Log the log likelihood every ``log_likelihood_interval`` iterations
        (0: never).

    Returns
    -------
    Trained LDA model.

    """
    if n_topics < 1 or n_topics > np.iinfo(np.uint16).max:
        raise ValueError(f"Number of topics should be in [1, 65535], not {n_topics}.")

    alpha_per_topic = alpha / n_topics if alpha_by_topic else alpha
    beta = eta / n_topics if eta_by_topic else eta
    beta_sum = beta * corpus.n_words
    n_partitions = corpus.n_partitions

    # Independent random number generator state per partition.
    rng_state = np.random.default_rng(random_seed).integers(
        1, np.iinfo(np.uint64).max, size=n_partitions, dtype=np.uint64
    )

    previous_n_threads = numba.get_num_threads()
    numba.set_num_threads(
        max(1, min(n_threads or n_partitions, numba.config.NUMBA_NUM_THREADS))
    )
    try:
        topics = np.empty(
            corpus.n_tokens, dtype=np.uint8 if n_topics <= 256 else np.uint16
        )
        _initialize_topics(
            topics,
            corpus.run_starts,
            corpus.block_run_offsets,
            n_partitions,
            n_topics,
            rng_state,
        )
        doc_topic_counts, word_topic_counts, topic_counts = _count_topics(
            corpus.run_words,
            corpus.run_starts,
            corpus.docs,
            topics,
            corpus.n_docs,
            corpus.n_words,
            n_topics,
        )
        # Allocate memory of tokens and cell-topic counts of each cell partition
        # close to the thread that samples them.
        docs = _copy_per_doc_partition(corpus.docs, corpus.token_partition_starts)
        doc_topic_counts = _copy_per_doc_partition(
            doc_topic_counts, corpus.doc_partition_starts
        )

        log_likelihoods = []
        start_time = time.time()
        for iteration in range(1, iterations + 1):
            _sample_iteration(
                corpus.run_words,
                corpus.run_starts,
                corpus.block_run_offsets,
                docs,
                topics,
                n_partitions,
                doc_topic_counts,
                word_topic_counts,
                topic_counts,
                alpha_per_topic,
                beta,
                beta_sum,
                rng_state,
            )
            if log_likelihood_interval > 0 and (
                iteration % log_likelihood_interval == 0 or iteration == iterations
            ):
                log_likelihood = _log_likelihood(
                    doc_topic_counts,
                    word_topic_counts,
                    topic_counts,
                    alpha_per_topic,
                    beta,
                )
                log_likelihoods.append((iteration, log_likelihood))
                logger.info(
                    f"Iteration {iteration}/{iterations} "
                    f"({time.time() - start_time:.0f}s): "
                    f"log likelihood per token: {log_likelihood / corpus.n_tokens:.4f}"
                )
    finally:
        numba.set_num_threads(previous_n_threads)

    # Map renumbered cells and regions back to the original order.
    doc_topic_counts_original_order = np.empty_like(doc_topic_counts)
    doc_topic_counts_original_order[corpus.doc_order] = doc_topic_counts
    word_topic_counts_original_order = np.empty_like(word_topic_counts)
    word_topic_counts_original_order[corpus.word_order] = word_topic_counts

    return LDAModel(
        n_topics=n_topics,
        alpha=alpha_per_topic,
        beta=beta,
        doc_topic_counts=doc_topic_counts_original_order,
        word_topic_counts=word_topic_counts_original_order,
        log_likelihoods=log_likelihoods,
    )


def run_lda_topic_modeling(
    binary_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    output_prefix: str,
    n_topics: list[int] | int,
    alpha: float = 50,
    alpha_by_topic: bool = True,
    eta: float = 0.1,
    eta_by_topic: bool = False,
    iterations: int = 150,
    random_seed: int = 555,
    n_threads: int = 8,
) -> None:
    """
    Run LDA topic modeling and write output files in the same format as Mallet.

    Writes for each number of topics:

    - ``<output_prefix>.<n_topics>_topics.cell_topic_probabilities.parquet``
    - ``<output_prefix>.<n_topics>_topics.region_topic_counts.parquet``
    - ``<output_prefix>.<n_topics>_topics.parameters.json``

    which can be used by ``pycistopic topic_modeling mallet stats``, ``plot_stats``
    and ``create_anndata``.

    Parameters
    ----------
    binary_matrix
        Binary accessibility matrix (regions x cells).
    output_prefix
        Output prefix.
    n_topics
        Number(s) of topics.
    alpha
        Symmetric Dirichlet hyperparameter for cell-topic distributions.
        Default: 50.
    alpha_by_topic
        Divide ``alpha`` by the number of topics. Default: ``True``.
    eta
        Symmetric Dirichlet hyperparameter for topic-region distributions.
        Default: 0.1.
    eta_by_topic
        Divide ``eta`` by the number of topics. Default: ``False``.
    iterations
        Number of Gibbs sampling iterations. Default: 150.
    random_seed
        Random seed. Results are reproducible for the same random seed and number of
        threads.
    n_threads
        Number of threads (and number of cell and region partitions).

    """
    if isinstance(n_topics, int):
        n_topics = [n_topics]

    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    corpus = LDACorpus(binary_matrix, n_partitions=n_threads, random_seed=random_seed)
    logger.info(
        f"Created corpus with {corpus.n_tokens} tokens ({corpus.n_docs} cells, "
        f"{corpus.n_words} regions) in {time.time() - start_time:.0f}s."
    )

    for n_topics_model in n_topics:
        logger.info(f"Train LDA model with {n_topics_model} topics.")
        start_time = time.time()
        model = train_lda(
            corpus,
            n_topics=n_topics_model,
            alpha=alpha,
            alpha_by_topic=alpha_by_topic,
            eta=eta,
            eta_by_topic=eta_by_topic,
            iterations=iterations,
            random_seed=random_seed,
            n_threads=n_threads,
        )
        elapsed_time = time.time() - start_time

        lda_filenames = LDAMalletFilenames(
            output_prefix=output_prefix, n_topics=n_topics_model
        )
        pl.Series(
            "cell_topic_probabilities",
            model.cell_topic_probabilities.astype(np.float32),
        ).to_frame().write_parquet(
            lda_filenames.cell_topic_probabilities_parquet_filename
        )
        pl.Series(
            "region_topic_counts", model.region_topic_counts
        ).to_frame().write_parquet(lda_filenames.region_topic_counts_parquet_filename)
        with open(lda_filenames.parameters_json_filename, "w") as fh:
            json.dump(
                {
                    "method": "pycistopic_lda",
                    "output_prefix": output_prefix,
                    "n_topics": n_topics_model,
                    "alpha": alpha,
                    "alpha_by_topic": alpha_by_topic,
                    "eta": eta,
                    "eta_by_topic": eta_by_topic,
                    "n_threads": n_threads,
                    "iterations": iterations,
                    "random_seed": random_seed,
                    "time": elapsed_time,
                    "log_likelihoods": model.log_likelihoods,
                },
                fh,
            )
        logger.info(
            f"Trained LDA model with {n_topics_model} topics in {elapsed_time:.0f}s."
        )

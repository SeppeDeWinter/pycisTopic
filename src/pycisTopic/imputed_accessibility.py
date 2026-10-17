from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl


def _alocate_chunk_array(
    n_cells: int,
    n_regions: int,
    chunk_size: int,
    chunk_along: Literal["cell", "region"],
    log: logging.Logger | None
) -> npt.NDArray[np.float32]:
    if chunk_along == "region":
        if log is not None:
            log.info(
                f"Allocate {(chunk_size * n_cells * 4 / 1024**3):.3f} GiB of RAM for "
                f"calculating (partial) imputed accessibility per cell for ({n_cells}) cells "
                f"for chunk of {chunk_size} regions."
            )
        return np.empty(
            (chunk_size, n_cells),
            dtype=np.float32
        )
    else:
        if log is not None:
            log.info(
                f"Allocate {(chunk_size * n_regions * 4 / 1024**3):.3f} GiB of RAM for "
                f"calculating (partial) imputed accessibility per region for ({n_regions}) regions "
                f"for chunk of {chunk_size} cells."
            )
        return np.empty(
            (n_regions, chunk_size),
            dtype=np.float32
        )

def impute_accessibility_chunked(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    chunk_size: int,
    chunk_along: Literal["cell", "region"],
    log: logging.Logger | None = None,
    return_start_end: bool = False,
) -> Iterator[npt.NDArray[np.float32]] | Iterator[tuple[tuple[int, int], npt.NDArray[np.float32]]]:
    """
    Impute accessibility in chunks.

    Parameters
    ----------
    region_topic
        Region topic matrix (regions x topics).
    cell_topic
        Cell topic matrix (topic x cells).
    chunk_size
        The size of the chunks.
    chunk_along
        Whether to chunk along cells or regions.
    log
        Optional logging.Logger
    return_start_end
        Whether to return the start and end index of the chunk.

    Yields
    ------
    Numpy array with imputed accessibility of chunk.

    .. warning::
        The yielded array is a reused in-place buffer. Each call to ``next()``
        overwrites its contents. Do not hold references to previously yielded
        arrays across iterations — copy the array (e.g. ``chunk.copy()``) if
        you need to retain the data.

    """
    if region_topic.shape[1] != cell_topic.shape[0]:
        raise ValueError(
            "region- and cell-topic dimensions do not match"
            f" region_topic: {region_topic.shape[1]}"
            f" cell_topic: {cell_topic.shape[0]}."
            "Maybe you have to transpose region_topic or cell_topic"
        )

    if chunk_along not in ["cell", "region"]:
        raise ValueError(f"chunk along should be either 'cell' or 'region', not {chunk_along}!")

    if log is not None:
        log.info(f"Calculating imputed accessibility in chunks across {chunk_along} ...")

    n_regions = region_topic.shape[0]
    n_cells = cell_topic.shape[1]

    imputed_acc_chunk: npt.NDArray[np.float32] = _alocate_chunk_array(
        n_cells=n_cells,
        n_regions=n_regions,
        chunk_size=chunk_size,
        chunk_along=chunk_along,
        log=log
    )

    n_total: int = n_regions if chunk_along == "region" else n_cells

    for chunk_start in range(0, n_total, chunk_size):
        chunk_end = chunk_start + chunk_size

        # get current chunk of regions or cells
        if chunk_along == "region":
            _cell_topic = cell_topic
            _region_topic = region_topic[
                chunk_start: chunk_end
            ]
            current_chunk_size = _region_topic.shape[0]
        else:
            _cell_topic = cell_topic[:, chunk_start: chunk_end]
            _region_topic = region_topic
            current_chunk_size = _cell_topic.shape[1]

        if current_chunk_size < chunk_size:
            del imputed_acc_chunk
            imputed_acc_chunk = _alocate_chunk_array(
                n_cells=n_cells,
                n_regions=n_regions,
                chunk_size=current_chunk_size,
                chunk_along=chunk_along,
                log=log
            )
            chunk_end = chunk_start + current_chunk_size

        if log is not None:
            log.info(
                "Calculate partial imputed accessibility "
                f"{chunk_start}-{chunk_end} (out of {n_total})."
            )

        np.matmul(_region_topic, _cell_topic, out=imputed_acc_chunk)
        if return_start_end:
            yield (chunk_start, chunk_end), imputed_acc_chunk
        else:
            yield imputed_acc_chunk

def rank_imputed_accessibility(
    imputed_acc: npt.NDArray[np.float32],
    seed: int = 123,
    engine: Literal["numpy", "polars"] = "polars",
) -> npt.NDArray[np.int32]:
    """
    Generate rankings per cell based on the imputed accessibility scores per region.

    Parameters
    ----------
    imputed_acc
        Imputed accessibility matrix (regions x cells).
    seed
        Random seed to ensure reproducibility of the rankings when there are ties.
        Results are only reproducible when both the same seed and the same method
        are used.
    engine
        Engine to use to calculate rankings of imputed accessibility scores per cell.
        Options are ``numpy`` (same rankings as with older versions of pycisTopic)
        or ``polars`` (fastest).

    Returns
    -------
    Numpy array (regions x cells) containing rankings per cell based on the imputed
    accessibility scores per region (regions x cells).

    """
    if engine != "numpy" and engine != "polars":
        raise ValueError(
            f'Invalid engine ("{engine}") for ranking implementation. Use "numpy" or "polars".'
        )

    # Initialize random number generator, for handling ties in the imputed accessibility
    # scores in a reproducible way. This ensures that the same random rankings will be
    # assigned for tied scores per cell each time this function is ran with the same
    # seed and engine.
    rng = np.random.default_rng(seed=seed)

    # Function to make rankings per array.
    def rank_scores_and_assign_random_ranking_in_range_for_ties_with_numpy(
        imputed_acc_scores_for_cell_with_ties_numpy: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int32]:
        # Rank all imputed accessibility scores per cell and assign a random ranking in
        # range for regions with the same imputed accessibility score.
        #   - Create a random permutation of the indexes of the input array.
        #   - Create an empty array for the rankings.
        #   - Use argsort on the negative imputed accessibility scores to get the
        #     ranking order (highest score gets rank 0).
        #   - Use the random permutation to break ties in the scores, so that regions
        #     with the same score get a random ranking in the range of their scores
        #     instead of depending on the order in which they appear in the input array.
        random_permutations_to_break_ties_numpy = rng.permutation(
            imputed_acc_scores_for_cell_with_ties_numpy.shape[0]
        )
        imputed_acc_rankings_with_broken_ties_for_cell_numpy = np.empty(
            imputed_acc_scores_for_cell_with_ties_numpy.shape[0],
            dtype=np.int32,
        )
        imputed_acc_rankings_with_broken_ties_for_cell_numpy[
            random_permutations_to_break_ties_numpy[
                (-imputed_acc_scores_for_cell_with_ties_numpy)[
                    random_permutations_to_break_ties_numpy
                ].argsort()
            ]
        ] = np.arange(
            imputed_acc_scores_for_cell_with_ties_numpy.shape[0],
            dtype=np.int32,
        )

        return imputed_acc_rankings_with_broken_ties_for_cell_numpy

    def rank_scores_and_assign_random_ranking_in_range_for_ties_with_polars(
        imputed_acc_scores_for_cell_with_ties_numpy: npt.NDArray[np.float32],
        seed: int,
    ) -> npt.NDArray[np.int32]:
        # Rank all imputed accessibility scores per cell and assign a random ranking in
        # range for regions with the same imputed accessibility score.
        #   - Convert numpy array to Polars Series.
        #   - Replace NaN values with the minimum value of the dtype, so NaNs are
        #     ranked last.
        #   - Use the `rank` engine from Polars to assign ranks, using the "random"
        #     engine to break ties so that regions with the same score get a random
        #     ranking in the range of their scores instead of depending on the order
        #     in which they appear in the input array.
        #   - Subtract 1 from the ranks to make them zero-based.
        return (
            (
                pl.Series(imputed_acc_scores_for_cell_with_ties_numpy)
                .fill_nan(
                    float(
                        np.finfo(imputed_acc_scores_for_cell_with_ties_numpy.dtype).min
                    )
                )
                .rank(method="random", descending=True, seed=seed)
                - 1
            )
            .cast(pl.Int32)
            .to_numpy()
        )

    n_regions, n_cells = imputed_acc.shape

    # Create zeroed imputed accessibility rankings database.
    imputed_acc_rankings = np.zeros((n_regions, n_cells), dtype=np.int32)

    # Rank all imputed accessibility scores per cell and assign a random ranking in
    # range for regions with the same imputed accessibility score.
    if engine == "numpy":
        for cell_idx in range(n_cells):
            imputed_acc_rankings[:, cell_idx] = (
                rank_scores_and_assign_random_ranking_in_range_for_ties_with_numpy(
                    imputed_acc[:, cell_idx]
                )
            )
    else:
        ranking_seeds = rng.integers(np.iinfo(np.int64).max, size=n_cells)

        for cell_idx, ranking_seed in enumerate(ranking_seeds):
            imputed_acc_rankings[:, cell_idx] = (
                rank_scores_and_assign_random_ranking_in_range_for_ties_with_polars(
                    imputed_acc[:, cell_idx],
                    seed=ranking_seed,
                )
            )

    return imputed_acc_rankings

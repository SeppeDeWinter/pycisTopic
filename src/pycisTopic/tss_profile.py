from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import polars as pl

from pycisTopic.categoricals import PycisTopicCategoricals

# Cut sites and TSS positions are encoded as `chromosome_code << 32 | position` so all
# chromosomes can be searched at once in one sorted array.
_CHROM_SHIFT = np.int64(32)


def _smoothing_matrix(
    n_positions: int, smoothing_rolling_window: int
) -> npt.NDArray[np.float64]:
    """
    Get matrix to smooth a TSS profile with a rolling mean.

    ``smoothing_matrix @ profile`` gives the same result as a polars rolling mean
    (``window_size=smoothing_rolling_window``, ``min_samples=0``): the mean of the current and previous ``smoothing_rolling_window - 1`` positions
    (or less positions for the first positions).

    """
    smoothing_matrix = np.zeros((n_positions, n_positions), dtype=np.float64)
    for position in range(n_positions):
        window_start = max(0, position - smoothing_rolling_window + 1)
        smoothing_matrix[position, window_start : position + 1] = 1.0 / (
            position + 1 - window_start
        )
    return smoothing_matrix


def _tss_profile_weights(
    flank_window: int,
    smoothing_rolling_window: int,
    minimum_signal_window: int,
    tss_window: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Get weights to calculate TSS signal and background from cut site counts.

    The TSS signal (mean of the smoothed profile in ``[-tss_window, tss_window]``) and
    the background (mean of the smoothed profile in the first and last
    ``minimum_signal_window`` positions) are linear in the cut site counts per
    position, so they can be calculated as a weighted sum of cut sites.

    Returns
    -------
    (smoothing_matrix, signal_weights, background_weights)

    """
    n_positions = 2 * flank_window + 1
    smoothing_matrix = _smoothing_matrix(n_positions, smoothing_rolling_window)
    tss_rows = slice(flank_window - tss_window, flank_window + tss_window + 1)
    signal_weights = smoothing_matrix[tss_rows].mean(axis=0)
    background_weights = (
        smoothing_matrix[:minimum_signal_window].mean(axis=0)
        + smoothing_matrix[-minimum_signal_window:].mean(axis=0)
    ) / 2
    return smoothing_matrix, signal_weights, background_weights


def _encode_positions(
    chrom_codes: npt.NDArray[np.integer], positions: npt.NDArray[np.integer]
) -> npt.NDArray[np.int64]:
    return (chrom_codes.astype(np.int64) << _CHROM_SHIFT) | positions.astype(np.int64)


@numba.njit(parallel=True, cache=True)
def _cut_site_weight_sums_per_cb(
    chrom_codes: npt.NDArray[np.uint32],
    fragment_starts: npt.NDArray[np.integer],
    fragment_ends: npt.NDArray[np.integer],
    cb_codes: npt.NDArray[np.uint32],
    tss_positions: npt.NDArray[np.int64],
    tss_on_minus_strand: npt.NDArray[np.bool_],
    flank_window: int,
    signal_weights: npt.NDArray[np.float64],
    background_weights: npt.NDArray[np.float64],
    n_cbs: int,
    n_chunks: int,
):
    """
    Sum weights of cut sites in ``[-flank_window, flank_window]`` around each TSS.

    Fragment start and end - 1 are the cut sites; a cut site is counted once for each
    TSS within ``flank_window`` bp. Cut site positions are relative to the TSS taking
    the strand into account (negative: upstream, positive: downstream).

    Returns
    -------
    (signal per CB, background per CB, number of cut sites per CB,
    number of cut sites per position relative to TSS)

    """
    n_positions = 2 * flank_window + 1
    n_fragments = fragment_starts.shape[0]
    chunk_size = (n_fragments + n_chunks - 1) // n_chunks

    signal_per_chunk = np.zeros((n_chunks, n_cbs), dtype=np.float64)
    background_per_chunk = np.zeros((n_chunks, n_cbs), dtype=np.float64)
    cut_sites_per_chunk = np.zeros((n_chunks, n_cbs), dtype=np.int64)
    cut_sites_per_position_per_chunk = np.zeros((n_chunks, n_positions), dtype=np.int64)

    for chunk in numba.prange(n_chunks):
        signal = signal_per_chunk[chunk]
        background = background_per_chunk[chunk]
        cut_sites = cut_sites_per_chunk[chunk]
        cut_sites_per_position = cut_sites_per_position_per_chunk[chunk]
        for i in range(
            chunk * chunk_size, min(n_fragments, (chunk + 1) * chunk_size)
        ):
            cb = cb_codes[i]
            chrom_offset = np.int64(chrom_codes[i]) << 32
            for cut_site in (
                chrom_offset + np.int64(fragment_starts[i]),
                chrom_offset + np.int64(fragment_ends[i]) - 1,
            ):
                j = np.searchsorted(tss_positions, cut_site - flank_window)
                while (
                    j < tss_positions.shape[0]
                    and tss_positions[j] <= cut_site + flank_window
                ):
                    if tss_on_minus_strand[j]:
                        position = tss_positions[j] - cut_site + flank_window
                    else:
                        position = cut_site - tss_positions[j] + flank_window
                    signal[cb] += signal_weights[position]
                    background[cb] += background_weights[position]
                    cut_sites[cb] += 1
                    cut_sites_per_position[position] += 1
                    j += 1

    return (
        signal_per_chunk.sum(axis=0),
        background_per_chunk.sum(axis=0),
        cut_sites_per_chunk.sum(axis=0),
        cut_sites_per_position_per_chunk.sum(axis=0),
    )


@numba.njit(cache=True)
def _cut_site_counts_per_cb(
    chrom_codes: npt.NDArray[np.uint32],
    fragment_starts: npt.NDArray[np.integer],
    fragment_ends: npt.NDArray[np.integer],
    cb_rows: npt.NDArray[np.int64],
    tss_positions: npt.NDArray[np.int64],
    tss_on_minus_strand: npt.NDArray[np.bool_],
    flank_window: int,
    n_rows: int,
) -> npt.NDArray[np.int32]:
    """Count cut sites per position relative to TSS for each CB (row; -1: skip)."""
    counts = np.zeros((n_rows, 2 * flank_window + 1), dtype=np.int32)
    for i in range(fragment_starts.shape[0]):
        row = cb_rows[i]
        if row < 0:
            continue
        chrom_offset = np.int64(chrom_codes[i]) << 32
        for cut_site in (
            chrom_offset + np.int64(fragment_starts[i]),
            chrom_offset + np.int64(fragment_ends[i]) - 1,
        ):
            j = np.searchsorted(tss_positions, cut_site - flank_window)
            while (
                j < tss_positions.shape[0]
                and tss_positions[j] <= cut_site + flank_window
            ):
                if tss_on_minus_strand[j]:
                    position = tss_positions[j] - cut_site + flank_window
                else:
                    position = cut_site - tss_positions[j] + flank_window
                counts[row, position] += 1
                j += 1
    return counts


def _normalize_profiles(
    smoothed_profiles: npt.NDArray[np.float64],
    minimum_signal_window: int,
    min_norm: float,
) -> npt.NDArray[np.float64]:
    """Normalize smoothed profiles (positions x profiles) by their background."""
    background = (
        smoothed_profiles[:minimum_signal_window].mean(axis=0)
        + smoothed_profiles[-minimum_signal_window:].mean(axis=0)
    ) / 2
    return smoothed_profiles / np.maximum(background, min_norm)


def get_tss_profile(
    fragments_df_pl: pl.DataFrame,
    tss_annotation: pl.DataFrame,
    flank_window: int = 2000,
    smoothing_rolling_window: int = 10,
    minimum_signal_window: int = 100,
    tss_window: int = 50,
    min_norm: float = 0.2,
    tss_norm_matrix_per_cb_cbs: pl.Series | None = None,
    no_threads: int = 8,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """
    Get TSS profile for Polars DataFrame with fragments filtered by cell barcodes.

    Cut sites (fragment start and end - 1) within ``flank_window`` bp of a TSS are
    counted per position relative to the TSS (taking the strand into account). The
    profile is smoothed by a rolling mean and normalized by the background signal in
    the tails of the flanking window. The TSS enrichment of a cell barcode is the mean
    normalized signal in ``[-tss_window, tss_window]``.

    Parameters
    ----------
    fragments_df_pl
        Polars DataFrame with fragments (filtered by cell barcodes of interest).
        See :func:`pycisTopic.fragments.filter_fragments_by_cb`.
    tss_annotation
        TSS annotation Polars DataFrame with at least the following columns:
        ``Chromosome``, ``TSS_start`` and ``Strand``.
        The ``TSS_start`` column is 0-based like a BED file.
        See :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl` and
        :func:`pycisTopic.gene_annotation.change_chromosome_source_in_bed` for ways
        to get TSS annotation from Ensembl BioMart.
    flank_window
        Flanking window around the TSS.
        Used for intersecting fragments with TSS positions and keeping cut sites.
        Default: ``2000`` (+/- 2000 bp).
    smoothing_rolling_window
        Rolling window used to smooth the cut sites signal.
        Default: 10.
    minimum_signal_window
        Average signal in the tails of the flanking window around the TSS:

        - ``[-flank_window, -flank_window + minimum_signal_window + 1]``
        - ``[flank_window - minimum_signal_window + 1, flank_window]``

        is used to normalize the TSS enrichment.
        Default: ``100`` (average signal in ``[-2000, -1901]``, ``[1901, 2000]``
        around TSS if ``flank_window=2000``).
    tss_window
        Window around the TSS used to count fragments in the TSS when calculating
        the TSS enrichment per cell barcode.
        Default: ``50`` (+/- 50 bp).
    min_norm
        Minimum normalization score.
        If the average minimum signal value is below this value, this number is used
        to normalize the TSS signal. This approach penalizes cells with fewer reads.
        Default: ``0.2``.
    tss_norm_matrix_per_cb_cbs
        Cell barcodes for which to return the normalized TSS profile. This matrix is
        large (``2 * flank_window + 1`` rows and one column per cell barcode), so it
        is only calculated when cell barcodes are given.
        Default: ``None``.
    no_threads
        Number of threads to use.

    Returns
    -------
    tss_enrichment_per_cb, tss_norm_matrix_sample, tss_norm_matrix_per_cb
        - ``tss_enrichment_per_cb``: ``CB`` and ``tss_enrichment`` columns for cell
          barcodes with at least one cut site in the flanking window around a TSS.
        - ``tss_norm_matrix_sample``: ``position_from_tss`` and
          ``normalized_tss_enrichment`` columns with the normalized TSS profile of all
          fragments.
        - ``tss_norm_matrix_per_cb``: ``position_from_tss`` column and one column per
          cell barcode in ``tss_norm_matrix_per_cb_cbs`` with the normalized TSS
          profile (``None`` if ``tss_norm_matrix_per_cb_cbs`` is ``None``).

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb
    pycisTopic.gene_annotation.change_chromosome_source_in_bed
    pycisTopic.gene_annotation.get_tss_annotation_from_ensembl

    Examples
    --------
    Get TSS annotation for requested transcript types from Ensembl BioMart.

    >>> ensembl_tss_annotation_bed_df_pl = get_tss_annotation_from_ensembl(
    ...     biomart_name="hsapiens_gene_ensembl"
    ... )

    Get TSS profile for Polars DataFrame with fragments filtered by cell barcodes.

    >>> tss_enrichment_per_cb, tss_norm_matrix_sample, _ = get_tss_profile(
    ...     fragments_df_pl=fragments_cb_filtered_df_pl,
    ...     tss_annotation=ensembl_tss_annotation_bed_df_pl,
    ...     flank_window=2000,
    ...     smoothing_rolling_window=10,
    ...     minimum_signal_window=100,
    ...     tss_window=50,
    ...     min_norm=0.2,
    ... )

    """
    chromosome_dtype = pl.Categorical(PycisTopicCategoricals.CHROMOSOME)
    cb_dtype = pl.Categorical(PycisTopicCategoricals.CB)

    # Categorical codes are only consistent while the categories are in use, so keep
    # the casted chromosome columns of fragments and TSS annotation alive at the same
    # time.
    fragments_chroms = fragments_df_pl.get_column("Chromosome").cast(chromosome_dtype)

    # TSS positions (without TSS annotations without strand info, in case that would
    # ever happen), sorted by chromosome and position.
    tss_annotation_with_strand = tss_annotation.filter(pl.col("Strand") != ".")
    tss_df_pl = (
        pl.DataFrame(
            {
                "chrom": tss_annotation_with_strand.get_column("Chromosome")
                .cast(chromosome_dtype)
                .to_physical(),
                "tss": tss_annotation_with_strand.get_column("TSS_start").cast(pl.Int64),
                "minus_strand": tss_annotation_with_strand.get_column("Strand") == "-",
            }
        )
        # Keep encoded positions of different chromosomes separate when searching for
        # TSS positions within flank_window bp of a cut site.
        .filter(pl.col("tss") < 2**32 - flank_window)
        .sort(["chrom", "tss"])
    )
    tss_positions = _encode_positions(
        tss_df_pl["chrom"].to_numpy(), tss_df_pl["tss"].to_numpy()
    )
    tss_on_minus_strand = tss_df_pl["minus_strand"].to_numpy()

    # Cut site positions are encoded in the same way as TSS positions in the kernels.
    chrom_codes = fragments_chroms.to_physical().to_numpy()
    fragment_starts = fragments_df_pl.get_column("Start").to_numpy()
    fragment_ends = fragments_df_pl.get_column("End").to_numpy()
    # Keep the CB categories alive until CB codes are converted back to CBs.
    fragments_cbs = fragments_df_pl.get_column("CB").cast(cb_dtype)
    cb_codes = fragments_cbs.to_physical().to_numpy()
    n_cbs = int(cb_codes.max()) + 1 if cb_codes.shape[0] else 0

    smoothing_matrix, signal_weights, background_weights = _tss_profile_weights(
        flank_window=flank_window,
        smoothing_rolling_window=smoothing_rolling_window,
        minimum_signal_window=minimum_signal_window,
        tss_window=tss_window,
    )

    previous_no_threads = numba.get_num_threads()
    numba.set_num_threads(max(1, min(no_threads, numba.config.NUMBA_NUM_THREADS)))
    try:
        signal, background, cut_sites, cut_sites_per_position = (
            _cut_site_weight_sums_per_cb(
                chrom_codes,
                fragment_starts,
                fragment_ends,
                cb_codes,
                tss_positions,
                tss_on_minus_strand,
                flank_window,
                signal_weights,
                background_weights,
                n_cbs,
                # More chunks than threads for load balancing.
                max(1, min(4 * numba.get_num_threads(), fragment_starts.shape[0])),
            )
        )
    finally:
        numba.set_num_threads(previous_no_threads)

    if cut_sites_per_position.sum() == 0:
        raise ValueError(
            "No overlap found between fragments and TSS annotation. Make sure both use the same chromosome name source."
        )

    # TSS enrichment per CB: mean normalized signal in [-tss_window, tss_window].
    cbs_with_cut_sites = np.flatnonzero(cut_sites)
    tss_enrichment_per_cb = pl.DataFrame(
        {
            # Convert CB codes back to CBs.
            "CB": pl.Series(cbs_with_cut_sites, dtype=pl.UInt32).cat.to(cb_dtype),
            "tss_enrichment": signal[cbs_with_cut_sites]
            / np.maximum(background[cbs_with_cut_sites], min_norm),
        }
    )

    positions_from_tss = pl.Series(
        "position_from_tss",
        np.arange(-flank_window, flank_window + 1, dtype=np.int32),
    )

    # Normalized sample TSS profile (sum of cut sites per position over all CBs).
    tss_norm_matrix_sample = pl.DataFrame(
        [
            positions_from_tss,
            pl.Series(
                "normalized_tss_enrichment",
                _normalize_profiles(
                    smoothing_matrix @ cut_sites_per_position.astype(np.float64),
                    minimum_signal_window=minimum_signal_window,
                    min_norm=min_norm,
                ),
            ),
        ]
    )

    tss_norm_matrix_per_cb = None
    if tss_norm_matrix_per_cb_cbs is not None:
        cbs = (
            pl.Series(tss_norm_matrix_per_cb_cbs)
            .cast(pl.Utf8)
            .unique(maintain_order=True)
        )
        # Row in the per CB matrix for each CB code (-1: CB not requested).
        cb_code_to_row = np.full(n_cbs, -1, dtype=np.int64)
        cb_codes_requested = cbs.cast(cb_dtype).to_physical().to_numpy()
        in_fragments = cb_codes_requested < n_cbs
        cb_code_to_row[cb_codes_requested[in_fragments]] = np.flatnonzero(in_fragments)
        counts_per_cb = _cut_site_counts_per_cb(
            chrom_codes,
            fragment_starts,
            fragment_ends,
            cb_code_to_row[cb_codes],
            tss_positions,
            tss_on_minus_strand,
            flank_window,
            cbs.len(),
        )
        norm_matrix_per_cb = _normalize_profiles(
            smoothing_matrix @ counts_per_cb.T.astype(np.float64),
            minimum_signal_window=minimum_signal_window,
            min_norm=min_norm,
        )
        del counts_per_cb
        tss_norm_matrix_per_cb = pl.DataFrame(
            [positions_from_tss]
            + [
                pl.Series(cb, norm_matrix_per_cb[:, idx])
                for idx, cb in enumerate(cbs.to_list())
            ]
        )

    return tss_enrichment_per_cb, tss_norm_matrix_sample, tss_norm_matrix_per_cb

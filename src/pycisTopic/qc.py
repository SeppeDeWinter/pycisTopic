from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

from pycisTopic.fragments import (
    get_fragments_in_peaks,
    get_fragments_per_cb,
    get_insert_size_distribution,
)
from pycisTopic.topic_binarization import threshold_otsu
from pycisTopic.tss_profile import get_tss_profile

try:
    from jax.scipy.stats import gaussian_kde
except ImportError:
    from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    import numpy.typing as npt


def get_barcodes_passing_qc_for_sample(
    sample_id: str,
    pycistopic_qc_output_dir: str | Path,
    unique_fragments_threshold: float | None = None,
    tss_enrichment_threshold: float | None = None,
    frip_threshold: float | None = None,
    use_automatic_thresholds: bool = True,
) -> tuple[list[str], dict[str, int|float]]:
    """
    Get barcodes passing quality control (QC) for a sample.

    Parameters
    ----------
    sample_id
        Sample ID.
    pycistopic_qc_output_dir
        Directory with output from ``pycistopic qc run``.
    unique_fragments_threshold
        Threshold for number of unique fragments in peaks.
        If not defined, and use_automatic_thresholds is False,
        the threshold will be set to 0.
    tss_enrichment_threshold
        Threshold for TSS enrichment score.
        If not defined, and use_automatic_thresholds is False,
        the threshold will be set to 0.0.
    frip_threshold
        Threshold for fraction of reads in peaks (FRiP).
        If not defined the threshold will be set to 0.0.
    use_automatic_thresholds
        Use automatic thresholds for unique fragments in peaks and TSS enrichment score
        as calculated by Otsu's method. If False, the thresholds will be set to 0 if not
        defined.

    Returns
    -------
    Tuple with:
        - List with cell barcodes passing QC.
        - Dictionary with thresholds used for QC.

    Raises
    ------
    FileNotFoundError
        If the file with fragments statistics per cell barcode does not exist.

    """
    if use_automatic_thresholds:
        otsu_thresholds_tsv_filename = os.path.join(
            pycistopic_qc_output_dir,
            f"{sample_id}.otsu_thresholds.tsv",
        )

        # Check whether files exist.
        if not os.path.exists(otsu_thresholds_tsv_filename):
            Warning(f'File "{otsu_thresholds_tsv_filename}" does not exist.')
        else:
            # Read automatic thresholds.
            (
                otsu_unique_fragments_threshold,
                otsu_tss_enrichment_threshold,
            ) = pl.read_csv(
                otsu_thresholds_tsv_filename,
                separator="\t",
                columns=[
                    "unique_fragments_in_peaks_count_otsu_threshold",
                    "tss_enrichment_otsu_threshold",
                ],
            ).row(0)

            print(f"{sample_id}:")

            if unique_fragments_threshold is None:
                print(
                    f"\tUsing automatic threshold for unique fragments: {otsu_unique_fragments_threshold}"
                )
                unique_fragments_threshold = otsu_unique_fragments_threshold
            else:
                print(
                    f"\tUsing user-defined threshold for unique fragments: {unique_fragments_threshold}"
                )

            if tss_enrichment_threshold is None:
                print(
                    f"\tUsing automatic threshold for TSS enrichment: {otsu_tss_enrichment_threshold}"
                )
                tss_enrichment_threshold = otsu_tss_enrichment_threshold
            else:
                print(
                    f"\tUsing user-defined threshold for TSS enrichment: {tss_enrichment_threshold}"
                )

    if (
        unique_fragments_threshold is None
        or tss_enrichment_threshold is None
        or frip_threshold is None
    ):
        if not use_automatic_thresholds:
            print(f"{sample_id}:")

        # Set thresholds to 0 if not defined.
        if unique_fragments_threshold is None:
            print("\tNo threshold for unique fragments defined, setting to 0.")
            unique_fragments_threshold = 0

        if tss_enrichment_threshold is None:
            print("\tNo threshold for TSS enrichment defined, setting to 0.0.")
            tss_enrichment_threshold = 0.0

        if frip_threshold is None:
            print("\tNo threshold for FRiP defined, setting to 0.0.")
            frip_threshold = 0.0

    # fragments_stats_per_cb_df_pl
    fragments_stats_per_cb_filename = os.path.join(
        pycistopic_qc_output_dir,
        f"{sample_id}.fragments_stats_per_cb.parquet",
    )

    # Check whether files exist.
    if not os.path.exists(fragments_stats_per_cb_filename):
        raise FileNotFoundError(
            f'File "{fragments_stats_per_cb_filename}" does not exist.'
        )

    # Get barcodes passing filters.
    barcodes_passing_filters = (
        pl.scan_parquet(fragments_stats_per_cb_filename)
        .filter(
            (pl.col("unique_fragments_in_peaks_count") >= unique_fragments_threshold)
            & (pl.col("tss_enrichment") >= tss_enrichment_threshold)
            & (pl.col("fraction_of_fragments_in_peaks") >= frip_threshold)
        )
        .select("CB")
        .collect()
        .to_series()
        .to_list()
    )

    return barcodes_passing_filters, {
        "unique_fragments_threshold": unique_fragments_threshold,
        "tss_enrichment_threshold": tss_enrichment_threshold,
        "frip_threshold": frip_threshold,
    }


def _compute_kde_binned(
    training_data: npt.NDArray[np.float64],
    test_data: npt.NDArray[np.float64],
    grid_spacing: float = 0.1,
    max_grid_points: int = 4_000_000,
) -> npt.NDArray[np.float64]:
    """
    Compute kernel-density estimate (KDE) using Gaussian kernels on a grid.

    Approximates ``gaussian_kde(training_data)(test_data)`` (Scott's rule bandwidth):
    the data is whitened with the kernel covariance (so the kernel becomes a standard
    normal distribution), training data is linearly binned on a grid with
    ``grid_spacing`` (in kernel standard deviations), smoothed with a Gaussian filter
    and the PDF is linearly interpolated at the test data points.

    """
    from scipy.linalg import cholesky, solve_triangular
    from scipy.ndimage import gaussian_filter, map_coordinates

    n_dims, n_training = training_data.shape

    # Kernel covariance as in scipy.stats.gaussian_kde (Scott's rule).
    factor = n_training ** (-1.0 / (n_dims + 4))
    kernel_cov = np.atleast_2d(np.cov(training_data)) * factor**2
    cholesky_lower = cholesky(kernel_cov, lower=True)

    # Whiten data: kernel becomes a standard normal distribution.
    training_z = solve_triangular(cholesky_lower, training_data, lower=True)
    test_z = solve_triangular(cholesky_lower, test_data, lower=True)

    # Grid covering the training data and the kernel tails.
    padding = 5.0
    grid_min = training_z.min(axis=1) - padding
    grid_max = training_z.max(axis=1) + padding
    spacing = max(
        grid_spacing,
        (np.prod(grid_max - grid_min) / max_grid_points) ** (1.0 / n_dims),
    )
    grid_shape = tuple(np.ceil((grid_max - grid_min) / spacing).astype(int) + 2)

    # Linear binning of the training data (each point is distributed over the
    # 2 ** n_dims surrounding grid points).
    training_grid_coord = (training_z - grid_min[:, None]) / spacing
    lower_idx = np.floor(training_grid_coord).astype(np.int64)
    fraction = training_grid_coord - lower_idx
    grid = np.zeros(int(np.prod(grid_shape)), dtype=np.float64)
    for corner in np.ndindex(*(2,) * n_dims):
        corner_arr = np.array(corner)[:, None]
        weights = np.prod(np.where(corner_arr == 1, fraction, 1.0 - fraction), axis=0)
        flat_idx = np.ravel_multi_index(tuple(lower_idx + corner_arr), grid_shape)
        grid += np.bincount(flat_idx, weights=weights, minlength=grid.shape[0])
    grid = grid.reshape(grid_shape)

    # Sum of kernels at each grid point (normalized discrete Gaussian filter).
    grid = gaussian_filter(grid, sigma=1.0 / spacing, mode="constant", truncate=5.0)

    # Interpolate at test data points (0 outside the grid).
    smoothed_at_test = map_coordinates(
        grid,
        (test_z - grid_min[:, None]) / spacing,
        order=1,
        mode="constant",
        cval=0.0,
    )

    # The normalized discrete filter gives the sum of the kernels times
    # spacing ** n_dims / (2 * pi) ** (n_dims / 2).
    return np.maximum(smoothed_at_test, 0.0) / (
        spacing**n_dims * n_training * np.prod(np.diag(cholesky_lower))
    )


def _compute_kde_exact(
    training_data: npt.NDArray[np.float64],
    test_data: npt.NDArray[np.float64],
    no_threads: int = 8,
) -> npt.NDArray[np.float64]:
    """Compute ``gaussian_kde(training_data)(test_data)`` in parallel."""
    # Only calculate the KDE for unique test data points.
    test_data_unique, test_data_original_order_idx = np.unique(
        test_data, axis=1, return_inverse=True
    )
    test_data_unique_split_arrays = np.array_split(
        test_data_unique, max(1, no_threads), axis=1
    )

    kde = gaussian_kde(training_data)

    with ThreadPoolExecutor(max(1, no_threads)) as executor:
        pdf_results = list(executor.map(kde, test_data_unique_split_arrays))

    return np.concatenate(pdf_results)[test_data_original_order_idx.ravel()]


def compute_kde(
    training_data: npt.ArrayLike,
    test_data: npt.ArrayLike,
    no_threads: int = 8,
    method: Literal["auto", "exact", "binned"] = "auto",
) -> npt.NDArray[np.float64]:
    """
    Compute kernel-density estimate (KDE) using Gaussian kernels.

    The exact method gives the same result as:

    >>> from scipy.stats import gaussian_kde
    >>> gaussian_kde(training_data)(test_data)

    but its run time grows with the product of the number of training and test data
    points. The binned method approximates it on a grid (relative error < 1% where
    the density is not negligible) and is fast for any number of data points.

    Dimensions in which all training data points have the same value are ignored.

    Parameters
    ----------
    training_data
        2D numpy array with training data to train the KDE (dimensions x points).
    test_data
        2D numpy array with test data for which to evaluate the estimated probability
        density function (PDF) (dimensions x points).
    no_threads
        Number of threads to use for the exact method.
    method
        ``"exact"``, ``"binned"`` or ``"auto"`` (exact for small data:
        number of training x test data points <= 1e8).

    Returns
    -------
    1D numpy array with probability density function (PDF) values for points in
    test_data.

    """
    training_data = np.atleast_2d(np.asarray(training_data, dtype=np.float64))
    test_data = np.atleast_2d(np.asarray(test_data, dtype=np.float64))

    # Ignore dimensions with the same value for all training data points (e.g.
    # duplication ratio of 0.0 for all cell barcodes when each fragment count is 1),
    # for which the covariance matrix would be singular.
    varying_dims = np.var(training_data, axis=1) > 0.0
    if not np.any(varying_dims):
        return np.ones(test_data.shape[1], dtype=np.float64)
    training_data = training_data[varying_dims]
    test_data = test_data[varying_dims]

    if method == "auto":
        method = (
            "exact"
            if training_data.shape[1] * test_data.shape[1] <= 100_000_000
            else "binned"
        )

    if method == "exact":
        return _compute_kde_exact(training_data, test_data, no_threads=no_threads)
    if method == "binned":
        return _compute_kde_binned(training_data, test_data)
    raise ValueError(f'Invalid method "{method}": use "auto", "exact" or "binned".')


def get_otsu_threshold(
    fragments_stats_per_cb_df_pl: pl.DataFrame,
    min_otsu_fragments: int = 100,
    min_otsu_tss: float = 1.0,
):
    """
    Get Otsu thresholds for number of unique fragments in peaks and TSS enrichment score.

    Parameters
    ----------
    fragments_stats_per_cb_df_pl
        Polars DataFrame with fragments statistics per cell barcode as generated by
        :func:`pycisTopic.qc.compute_qc_stats`.
    min_otsu_fragments
        When calculating Otsu threshold for number of unique fragments in peaks per CB,
        only consider those CBs which have at least this number of fragments.
    min_otsu_tss
        When calculating Otsu threshold for TSS enrichment score per CB, only consider
        those CBs which have at least this TSS value.

    Returns
    -------
    Tuple with:
      - Otsu threshold for number of unique fragments in peaks.
      - Otsu threshold for TSS enrichment.
      - Polars DataFrame with fragments statistics per cell barcode for cell barcodes
        that passed both Otsu thresholds.

    Examples
    --------
    Only keep fragments stats for CBs that pass both Otsu thresholds.
    >>> (
    ...     unique_fragments_in_peaks_count_otsu_threshold,
    ...     tss_enrichment_otsu_threshold,
    ...     fragments_stats_per_cb_for_otsu_threshold_df_pl,
    ... ) = get_otsu_threshold(
    ...     fragments_stats_per_cb_df_pl=fragments_stats_per_cb_df_pl,
    ...     min_otsu_fragments=100,
    ...     min_otsu_tss=1.0,
    ... )

    """
    # Get number (in log10) of unique fragments in peaks per CB and TSS enrichment
    # score per CB.
    log10_unique_fragments_in_peaks_count, tss_enrichment = (
        fragments_stats_per_cb_df_pl.select(
            "log10_unique_fragments_in_peaks_count", "tss_enrichment"
        )
        .to_numpy()
        .T
    )

    # Get Otsu threshold for number of unique fragments in peaks per CB.
    unique_fragments_in_peaks_count_otsu_threshold = 10 ** threshold_otsu(
        log10_unique_fragments_in_peaks_count[
            log10_unique_fragments_in_peaks_count >= math.log10(min_otsu_fragments)
        ],
        nbins=5000,
    )

    # Get Otsu threshold for TSS enrichment score per CB.
    tss_enrichment_otsu_threshold = threshold_otsu(
        tss_enrichment[tss_enrichment >= min_otsu_tss],
        nbins=5000,
    )

    # Only keep fragments stats for CBs that pass both Otsu thresholds.
    fragments_stats_per_cb_for_otsu_threshold_df_pl = (
        fragments_stats_per_cb_df_pl.filter(
            (
                pl.col("unique_fragments_in_peaks_count")
                > unique_fragments_in_peaks_count_otsu_threshold
            )
            & (pl.col("tss_enrichment") > tss_enrichment_otsu_threshold)
        )
    )

    return (
        unique_fragments_in_peaks_count_otsu_threshold,
        tss_enrichment_otsu_threshold,
        fragments_stats_per_cb_for_otsu_threshold_df_pl,
    )


def compute_qc_stats(
    fragments_df_pl: pl.DataFrame,
    regions_df_pl: pl.DataFrame,
    tss_annotation: pl.DataFrame,
    tss_flank_window: int = 2000,
    tss_smoothing_rolling_window: int = 10,
    tss_minimum_signal_window: int = 100,
    tss_window: int = 50,
    tss_min_norm: float = 0.2,
    min_fragments_per_cb: int = 10,
    collapse_duplicates: bool = True,
    no_threads: int = 8,
    tss_norm_matrix_per_cb: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """
    Compute quality check statistics from Polars DataFrame with fragments.

    Parameters
    ----------
    fragments_df_pl
        Polars DataFrame with fragments.
        fragments_df_pl
        Polars DataFrame with fragments (filtered by cell barcodes of interest).
        See :func:`pycisTopic.fragments.filter_fragments_by_cb`.
    regions_df_pl
        Polars DataFrame with peak regions (consensus peaks or SCREEN regions).
        See :func:`pycisTopic.fragments.read_bed_to_polars_df` for a way to read a BED
        file with peak regions.
    tss_annotation
        TSS annotation Polars DataFrame with at least the following columns:
        ``["Chromosome", "Start", "Strand"]``.
        The "Start" column is 0-based like a BED file.
        See :func:`pycisTopic.gene_annotation.read_tss_annotation_from_bed`,
        :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl` and
        :func:`pycisTopic.gene_annotation.change_chromosome_source_in_bed` for ways
        to get TSS annotation from Ensembl BioMart.
    tss_flank_window
        Flanking window around the TSS.
        Used for intersecting fragments with TSS positions and keeping cut sites.
        Default: ``2000`` (+/- 2000 bp).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_smoothing_rolling_window
        Rolling window used to smooth the cut sites signal.
        Default: ``10``.
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_minimum_signal_window
        Average signal in the tails of the flanking window around the TSS:

        - ``[-flank_window, -flank_window + minimum_signal_window + 1]``
        - ``[flank_window - minimum_signal_window + 1, flank_window]``

        is used to normalize the TSS enrichment.
        Default: ``100`` (average signal in ``[-2000, -1901]``, ``[1901, 2000]``
        around TSS if ``flank_window=2000``).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_window
        Window around the TSS used to count fragments in the TSS when calculating
        the TSS enrichment per cell barcode.
        Default: ``50`` (+/- 50 bp).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_min_norm
        Minimum normalization score.
        If the average minimum signal value is below this value, this number is used
        to normalize the TSS signal. This approach penalizes cells with fewer reads.
        Default: ``0.2``
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    min_fragments_per_cb
        Minimum number of fragments needed per cell barcode to keep the fragments
        for those cell barcodes.
    collapse_duplicates
        Collapse duplicate fragments (same chromosomal positions and linked to the same
        cell barcode).
    no_threads
        Number of threads to use to calculate the TSS profile and the kernel-density
        estimates (KDE).
        Default: ``8``.
    tss_norm_matrix_per_cb
        Calculate the normalized TSS profile per cell barcode (for cell barcodes with
        more than ``min_fragments_per_cb`` fragments). This matrix is large.
        Default: ``False``.

    Returns
    -------
    Tuple with:
      - Polars DataFrame with fragments statistics per cell barcode.
      - Polars DataFrame with insert size distribution of fragments.
      - Polars DataFrame with TSS normalization matrix for the whole sample.
      - Polars DataFrame with TSS normalization matrix per cell barcode (``None`` if
        ``tss_norm_matrix_per_cb`` is ``False``).

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb
    pycisTopic.fragments.get_insert_size_distribution
    pycisTopic.fragments.get_fragments_in_peaks
    pycisTopic.fragments.read_bed_to_polars_df
    pycisTopic.fragments.read_fragments_to_polars_df
    pycisTopic.gene_annotation.read_tss_annotation_from_bed
    pycisTopic.tss_profile.get_tss_profile

    Examples
    --------
    >>> from pycisTopic.fragments import read_bed_to_polars_df
    >>> from pycisTopic.fragments import read_fragments_to_polars_df
    >>> from pycisTopic.gene_annotation import read_tss_annotation_from_bed

    1. Read gzipped fragments BED file to a Polars DataFrame.

    >>> fragments_df_pl = read_fragments_to_polars_df(
    ...     fragments_bed_filename="fragments.tsv.gz",
    ... )

    2. Read BED file with consensus peaks or SCREEN regions (get first 3 columns only)
       which will be used for counting number of fragments in peaks.

    >>> regions_df_pl = read_bed_to_polars_df(
    ...     bed_filename=screen_regions_bed_filename,
    ...     min_column_count=3,
    ... )

    3. Read TSS annotation from a file.
       See :func:`pycisTopic.gene_annotation.read_tss_annotation_from_bed` for more
       info.

    >>> tss_annotation_bed_df_pl = read_tss_annotation_from_bed(
    ...     tss_annotation_bed_filename="hg38.tss.bed",
    ... )

    4. Compute QC statistics.

    >>> (
    ...     fragments_stats_per_cb_df_pl,
    ...     insert_size_dist_df_pl,
    ...     tss_norm_matrix_sample,
    ...     tss_norm_matrix_per_cb,
    ... ) = compute_qc_stats(
    ...     fragments_df_pl=fragments_cb_filtered_df_pl,
    ...     regions_df_pl=regions_df_pl,
    ...     tss_annotation=tss_annotation_bed_df_pl,
    ...     tss_flank_window=2000,
    ...     tss_smoothing_rolling_window=10,
    ...     tss_minimum_signal_window=100,
    ...     tss_window=50,
    ...     tss_min_norm=0.2,
    ...     min_fragments_per_cb=10,
    ...     collapse_duplicates=True,
    ...     no_threads=8,
    ... )

    """
    logger = logging.getLogger(__name__)

    # Define correct column to get, based on the setting of `collapse_duplicates`.
    fragments_count_column = (
        "unique_fragments_count" if collapse_duplicates else "total_fragments_count"
    )
    fragments_in_peaks_count_column = (
        "unique_fragments_in_peaks_count"
        if collapse_duplicates
        else "total_fragments_in_peaks_count"
    )

    # Get Polars DataFrame with basic fragments statistics per cell barcode.
    logger.info("Get basic fragments statistics per cell barcode.")
    fragments_stats_per_cb_df_pl = get_fragments_per_cb(
        fragments_df_pl=fragments_df_pl,
        min_fragments_per_cb=min_fragments_per_cb,
        collapse_duplicates=collapse_duplicates,
    )

    # Get Polars DataFrame with total fragment counts and unique fragment counts
    # per region.
    logger.info("Get total fragment counts and unique fragment counts per region.")
    fragments_in_peaks_df_pl = get_fragments_in_peaks(
        fragments_df_pl=fragments_df_pl,
        regions_df_pl=regions_df_pl,
    )

    # Add fragment counts per region to fragments statistics per cell barcode.
    logger.info(
        "Add fragment counts per region to fragments statistics per cell barcode."
    )
    fragments_stats_per_cb_df_pl = (
        fragments_stats_per_cb_df_pl.lazy()
        .join(
            fragments_in_peaks_df_pl.lazy(),
            how="left",
            on="CB",
            maintain_order="left",
        )
        .with_columns(
            pl.col("total_fragments_in_peaks_count").fill_null(0),
            pl.col("unique_fragments_in_peaks_count").fill_null(0),
        )
        .with_columns(
            (
                pl.col(fragments_in_peaks_count_column) / pl.col(fragments_count_column)
            ).alias("fraction_of_fragments_in_peaks")
        )
        .select(
            pl.col("CB"),
            pl.col("barcode_rank"),
            pl.col("total_fragments_count"),
            (pl.col("total_fragments_count") + 1)
            .log10()
            .alias("log10_total_fragments_count"),
            pl.col("unique_fragments_count"),
            (pl.col("unique_fragments_count") + 1)
            .log10()
            .alias("log10_unique_fragments_count"),
            pl.col("total_fragments_in_peaks_count"),
            (pl.col("total_fragments_in_peaks_count") + 1)
            .log10()
            .alias("log10_total_fragments_in_peaks_count"),
            pl.col("unique_fragments_in_peaks_count"),
            (pl.col("unique_fragments_in_peaks_count") + 1)
            .log10()
            .alias("log10_unique_fragments_in_peaks_count"),
            pl.col("fraction_of_fragments_in_peaks"),
            pl.col("duplication_count"),
            pl.col("duplication_ratio"),
            pl.col("nucleosome_signal"),
        )
    )

    # Get insert size distribution of fragments.
    logger.info("Get insert size distribution of fragments.")
    insert_size_dist_df_pl = get_insert_size_distribution(
        fragments_df_pl=fragments_df_pl,
    )

    # Get TSS profile for fragments.
    logger.info("Get TSS profile for fragments.")
    fragments_stats_per_cb_df_pl = fragments_stats_per_cb_df_pl.collect()
    (
        tss_enrichment_per_cb,
        tss_norm_matrix_sample,
        tss_norm_matrix_per_cb_df_pl,
    ) = get_tss_profile(
        fragments_df_pl=fragments_df_pl,
        tss_annotation=tss_annotation,
        flank_window=tss_flank_window,
        smoothing_rolling_window=tss_smoothing_rolling_window,
        minimum_signal_window=tss_minimum_signal_window,
        tss_window=tss_window,
        min_norm=tss_min_norm,
        tss_norm_matrix_per_cb_cbs=(
            fragments_stats_per_cb_df_pl.get_column("CB")
            if tss_norm_matrix_per_cb
            else None
        ),
        no_threads=no_threads,
    )

    # Add TSS enrichment to fragments statistics per cell barcode.
    logger.info("Add TSS enrichment to fragments statistics per cell barcode.")
    fragments_stats_per_cb_df_pl = (
        fragments_stats_per_cb_df_pl.lazy()
        .join(
            tss_enrichment_per_cb.lazy(),
            how="left",
            on="CB",
            maintain_order="left",
        )
        .with_columns(
            pl.col("tss_enrichment").fill_null(0.0),
        )
        .collect()
    )

    # Extract certain columns as numpy arrays as they are needed for calculating KDE.
    (
        log10_unique_fragments_in_peaks_count,
        tss_enrichment,
        fraction_of_fragments_in_peaks,
        duplication_ratio,
    ) = (
        fragments_stats_per_cb_df_pl.select(
            [
                pl.col("log10_unique_fragments_in_peaks_count"),
                pl.col("tss_enrichment"),
                pl.col("fraction_of_fragments_in_peaks"),
                pl.col("duplication_ratio"),
            ]
        )
        .to_numpy()
        .T
    )

    # Construct 2D numpy matrices for usage with compute_kde.
    kde_data_for_tss_enrichment = np.vstack(
        [log10_unique_fragments_in_peaks_count, tss_enrichment]
    )
    kde_data_for_fraction_of_fragments_in_peaks = np.vstack(
        [log10_unique_fragments_in_peaks_count, fraction_of_fragments_in_peaks]
    )
    kde_data_for_duplication_ratio = np.vstack(
        [log10_unique_fragments_in_peaks_count, duplication_ratio]
    )

    # Calculate KDE for log10 unique fragments in peaks vs TSS enrichment,
    # fractions of fragments in peaks and duplication ratio.
    logger.info("Calculate KDE for log10 unique fragments in peaks vs TSS enrichment.")
    pdf_values_for_tss_enrichment = compute_kde(
        training_data=kde_data_for_tss_enrichment,
        test_data=kde_data_for_tss_enrichment,
        no_threads=no_threads,
    )
    logger.info(
        "Calculate KDE for log10 unique fragments in peaks vs fractions of fragments "
        "in peaks."
    )
    pdf_values_for_fraction_of_fragments_in_peaks = compute_kde(
        training_data=kde_data_for_fraction_of_fragments_in_peaks,
        test_data=kde_data_for_fraction_of_fragments_in_peaks,
        no_threads=no_threads,
    )
    logger.info(
        "Calculate KDE for log10 unique fragments in peaks vs duplication ratio."
    )
    pdf_values_for_duplication_ratio = compute_kde(
        training_data=kde_data_for_duplication_ratio,
        test_data=kde_data_for_duplication_ratio,
        no_threads=no_threads,
    )

    # Add probability density function (PDF) values for log10 unique fragments in peaks
    # vs TSS enrichment, fractions of fragments in peaks and duplication ratio to
    # fragments statistics per cell barcode.
    logger.info(
        "Add probability density function (PDF) values to fragments statistics per "
        "cell barcode."
    )
    fragments_stats_per_cb_df_pl = fragments_stats_per_cb_df_pl.hstack(
        pl.DataFrame(
            {
                "pdf_values_for_tss_enrichment": pdf_values_for_tss_enrichment,
                "pdf_values_for_fraction_of_fragments_in_peaks": pdf_values_for_fraction_of_fragments_in_peaks,
                "pdf_values_for_duplication_ratio": pdf_values_for_duplication_ratio,
            }
        )
    )

    return (
        fragments_stats_per_cb_df_pl,
        insert_size_dist_df_pl,
        tss_norm_matrix_sample,
        tss_norm_matrix_per_cb_df_pl,
    )

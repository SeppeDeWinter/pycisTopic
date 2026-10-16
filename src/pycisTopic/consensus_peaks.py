"""
Consensus peaks with iterative peak filtering.

Python implementation of the scripts in https://github.com/aertslab/iterative_peak_filtering
(``calculate_center_peaks_and_extend.sh``, ``iterative_peak_filtering.sh``,
``normalize_macs2_peak_scores.sh`` and
``remove_peaks_overlapping_blacklisted_regions.sh``), as used in Corces et al., 2018,
without depending on gawk or bedtools.

Typical use with MACS summits per pseudobulk::

    chrom_sizes = read_chrom_sizes("hg38.chrom.sizes")
    consensus_peaks = get_consensus_peaks(
        {name: read_peaks_bed(f"{name}_summits.bed") for name in pseudobulks},
        chrom_sizes=chrom_sizes,
        peak_half_width=250,
    )
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Mapping

import numba
import numpy as np
import polars as pl

from pycisTopic.genomic_ranges import overlap

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

PEAK_COLUMNS = ["Chromosome", "Start", "End", "Name", "Score"]


def read_chrom_sizes(chrom_sizes_filename: str | Path) -> dict[str, int]:
    """
    Read chromosome sizes file (``*.chrom.sizes`` or ``*.fa.fai``).

    Parameters
    ----------
    chrom_sizes_filename
        File with chromosome names in the first column and chromosome sizes in the
        second column.

    Returns
    -------
    Dictionary with chromosome names as keys and chromosome sizes as values (in the
    same order as the file).

    """
    chrom_sizes: dict[str, int] = {}
    with open(chrom_sizes_filename) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            chrom_sizes[columns[0]] = int(columns[1])
    return chrom_sizes


def read_peaks_bed(peaks_bed_filename: str | Path) -> pl.DataFrame:
    """
    Read peaks from a BED file (e.g. MACS ``*_summits.bed`` or ``*_peaks.narrowPeak``).

    For MACS narrowPeak and broadPeak files (10 columns), the ``-log10(pvalue)``
    column (column 8) is used as peak score, otherwise column 5.

    Parameters
    ----------
    peaks_bed_filename
        BED file with at least 5 columns.

    Returns
    -------
    Polars DataFrame with ``Chromosome``, ``Start``, ``End``, ``Name`` and ``Score``.

    """
    peaks = pl.read_csv(
        peaks_bed_filename,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        infer_schema=False,
    )
    if peaks.width < 5:
        raise ValueError(
            f'Peaks BED file "{peaks_bed_filename}" needs at least 5 columns, '
            f"found {peaks.width}."
        )
    score_column = 7 if peaks.width == 10 else 4
    return peaks.select(
        pl.col(peaks.columns[0]).alias("Chromosome"),
        pl.col(peaks.columns[1]).cast(pl.Int64).alias("Start"),
        pl.col(peaks.columns[2]).cast(pl.Int64).alias("End"),
        pl.col(peaks.columns[3]).alias("Name"),
        pl.col(peaks.columns[score_column]).cast(pl.Float64).alias("Score"),
    )


def write_peaks_bed(peaks: pl.DataFrame, peaks_bed_filename: str | Path) -> None:
    """Write peaks (Chromosome, Start, End, Name and Score) to a BED file."""
    peaks.select(PEAK_COLUMNS).write_csv(
        peaks_bed_filename, separator="\t", include_header=False
    )


def _keep_known_chromosomes(
    peaks: pl.DataFrame, chrom_sizes: Mapping[str, int]
) -> pl.DataFrame:
    unknown = peaks.filter(~pl.col("Chromosome").is_in(list(chrom_sizes)))
    if unknown.height > 0:
        warnings.warn(
            f"Removed {unknown.height} peaks on chromosomes not found in the chromosome "
            f"sizes (e.g. {unknown['Chromosome'][0]}).",
            stacklevel=3,
        )
        peaks = peaks.filter(pl.col("Chromosome").is_in(list(chrom_sizes)))
    return peaks


def sort_peaks(peaks: pl.DataFrame, chrom_sizes: Mapping[str, int]) -> pl.DataFrame:
    """
    Sort peaks by chromosome (in the order of ``chrom_sizes``), start and end.

    Same order as ``bedtools sort -g chrom_sizes``.
    """
    chrom_order = pl.DataFrame(
        {"Chromosome": list(chrom_sizes), "_chrom_idx": range(len(chrom_sizes))},
        schema={"Chromosome": pl.Utf8, "_chrom_idx": pl.Int64},
    )
    return (
        peaks.with_columns(pl.col("Chromosome").cast(pl.Utf8))
        .join(chrom_order, on="Chromosome", how="left", maintain_order="left")
        .sort(["_chrom_idx", "Start", "End"], maintain_order=True)
        .drop("_chrom_idx")
    )


def calculate_center_peaks_and_extend(
    peaks: pl.DataFrame,
    chrom_sizes: Mapping[str, int],
    peak_half_width: int = 250,
) -> pl.DataFrame:
    """
    Calculate the center of each peak and extend it in both directions.

    Peaks which would extend beyond the start or end of the chromosome are removed,
    so all returned peaks have a width of ``2 * peak_half_width + 1``.

    Parameters
    ----------
    peaks
        Peaks (e.g. MACS summits) with ``Chromosome``, ``Start``, ``End``, ``Name``
        and ``Score``.
    chrom_sizes
        Chromosome sizes.
    peak_half_width
        Number of base pairs to extend a peak in each direction from its center.

    Returns
    -------
    Extended peaks, sorted by chromosome (in the order of ``chrom_sizes``) and
    position.

    """
    peaks = _keep_known_chromosomes(peaks, chrom_sizes)
    chrom_size = pl.col("Chromosome").replace_strict(
        chrom_sizes, return_dtype=pl.Int64
    )
    # The center of peaks with an even length is slightly more to the left.
    center = pl.col("Start") + (pl.col("End") - pl.col("Start")) // 2
    extended = peaks.with_columns(
        pl.max_horizontal(center - peak_half_width, pl.lit(0)).alias("Start"),
        pl.min_horizontal(center + 1 + peak_half_width, chrom_size).alias("End"),
    ).filter(pl.col("End") - pl.col("Start") == 2 * peak_half_width + 1)
    return sort_peaks(extended, chrom_sizes)


@numba.njit
def _iterative_peak_filtering_keep(
    chrom_idx: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Get which (sorted) peaks are kept by iterative peak filtering."""
    n_peaks = starts.shape[0]
    keep = np.zeros(n_peaks, dtype=np.bool_)
    cluster_start = 0
    while cluster_start < n_peaks:
        # Find cluster of overlapping or bookended peaks (as "bedtools merge").
        cluster_end = cluster_start + 1
        cluster_max_end = ends[cluster_start]
        while (
            cluster_end < n_peaks
            and chrom_idx[cluster_end] == chrom_idx[cluster_start]
            and starts[cluster_end] <= cluster_max_end
        ):
            cluster_max_end = max(cluster_max_end, ends[cluster_end])
            cluster_end += 1
        n_cluster = cluster_end - cluster_start

        if n_cluster == 1:
            keep[cluster_start] = True
        elif n_cluster == 2:
            # Keep the peak with the highest score.
            if scores[cluster_start] >= scores[cluster_start + 1]:
                keep[cluster_start] = True
            else:
                keep[cluster_start + 1] = True
        else:
            # Process peaks from the highest to the lowest score (ties in order of
            # position). Skip peaks adjacent to or overlapping with selected peaks.
            order = np.argsort(-scores[cluster_start:cluster_end], kind="mergesort")
            ignore = np.zeros(n_cluster + 2, dtype=np.bool_)
            selected = np.empty(n_cluster, dtype=np.int64)
            n_selected = 0
            for i in order:
                if ignore[i + 1]:
                    continue
                skip = False
                current = cluster_start + i
                for k in range(n_selected):
                    previous = cluster_start + selected[k]
                    previous_length = ends[previous] - starts[previous]
                    diff_end_prev_start_cur = ends[previous] - starts[current]
                    diff_end_cur_start_prev = ends[current] - starts[previous]
                    if 0 <= diff_end_prev_start_cur <= previous_length:
                        skip = True
                    if 0 <= diff_end_cur_start_prev <= previous_length:
                        skip = True
                if not skip:
                    keep[current] = True
                    selected[n_selected] = i
                    n_selected += 1
                    ignore[i] = True
                    ignore[i + 1] = True
                    ignore[i + 2] = True
        cluster_start = cluster_end
    return keep


def iterative_peak_filtering(
    peaks: pl.DataFrame,
    chrom_sizes: Mapping[str, int],
) -> pl.DataFrame:
    """
    Filter overlapping peaks iteratively.

    Overlapping (or bookended) peaks are grouped. Within each group, the most
    significant peak is kept and peaks overlapping with it are removed. This is
    repeated for the next most significant peak (that is not removed already) until
    all peaks are processed.

    Parameters
    ----------
    peaks
        Peaks with ``Chromosome``, ``Start``, ``End``, ``Name`` and ``Score``.
    chrom_sizes
        Chromosome sizes (used for sorting the peaks).

    Returns
    -------
    Filtered peaks, sorted by chromosome (in the order of ``chrom_sizes``) and
    position.

    """
    peaks = sort_peaks(_keep_known_chromosomes(peaks, chrom_sizes), chrom_sizes)
    chrom_idx = (
        peaks.get_column("Chromosome")
        .replace_strict(
            list(chrom_sizes), list(range(len(chrom_sizes))), return_dtype=pl.Int64
        )
        .to_numpy()
    )
    keep = _iterative_peak_filtering_keep(
        chrom_idx,
        peaks.get_column("Start").to_numpy().astype(np.int64),
        peaks.get_column("End").to_numpy().astype(np.int64),
        peaks.get_column("Score").to_numpy().astype(np.float64),
    )
    return peaks.filter(pl.Series(keep))


def normalize_peak_scores(peaks: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize peak scores to a "score per million".

    Each peak score is divided by the sum of all peak scores and multiplied by one
    million. Like the original gawk script, normalized scores are rounded to 6
    significant digits.

    Parameters
    ----------
    peaks
        Peaks with a ``Score`` column (e.g. MACS ``-log10(pvalue)``).

    Returns
    -------
    Peaks with normalized scores.

    """
    normalized = peaks.get_column("Score").to_numpy().astype(np.float64)
    normalized = normalized / normalized.sum() * 1_000_000
    rounded = np.array([float(f"{x:.6g}") for x in normalized], dtype=np.float64)
    return peaks.with_columns(pl.Series("Score", rounded))


def remove_peaks_overlapping_blacklisted_regions(
    peaks: pl.DataFrame,
    blacklist: pl.DataFrame,
    chrom_sizes: Mapping[str, int],
) -> pl.DataFrame:
    """
    Remove peaks which (partially) overlap with blacklisted regions.

    Parameters
    ----------
    peaks
        Peaks with ``Chromosome``, ``Start`` and ``End``.
    blacklist
        Blacklisted regions with ``Chromosome``, ``Start`` and ``End``.
    chrom_sizes
        Chromosome sizes (used for sorting the peaks).

    Returns
    -------
    Peaks which do not overlap with blacklisted regions, sorted by chromosome and
    position.

    """
    blacklist = blacklist.select(
        pl.col("Chromosome").cast(pl.Utf8),
        pl.col("Start").cast(pl.Int64),
        pl.col("End").cast(pl.Int64),
    )
    return sort_peaks(
        overlap(
            peaks.with_columns(pl.col("Chromosome").cast(pl.Utf8)),
            blacklist,
            invert=True,
        ),
        chrom_sizes,
    )


def get_consensus_peaks(
    peaks_per_sample: Mapping[str, pl.DataFrame],
    chrom_sizes: Mapping[str, int],
    peak_half_width: int = 250,
    blacklist: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Get consensus peaks from peaks (MACS summits) called per sample or pseudobulk.

    For each sample:
      - the peaks are centered and extended by ``peak_half_width`` in both
        directions (:func:`calculate_center_peaks_and_extend`),
      - overlapping peaks are filtered iteratively (:func:`iterative_peak_filtering`),
      - peak scores are normalized to a score per million
        (:func:`normalize_peak_scores`).

    Then the peaks of all samples are combined and overlapping peaks are filtered
    iteratively again. Optionally, peaks overlapping with blacklisted regions are
    removed. Finally, the peak ends are decreased by one, so consensus peaks have a
    width of ``2 * peak_half_width``.

    Parameters
    ----------
    peaks_per_sample
        Dictionary with sample names as keys and peaks (see :func:`read_peaks_bed`) as
        values, e.g. MACS ``*_summits.bed`` files.
    chrom_sizes
        Chromosome sizes (see :func:`read_chrom_sizes`).
    peak_half_width
        Number of base pairs to extend a peak in each direction from its center.
    blacklist
        Optional blacklisted regions (``Chromosome``, ``Start``, ``End``).

    Returns
    -------
    Consensus peaks (``Chromosome``, ``Start``, ``End``, ``Name``, ``Score``), sorted by
    chromosome (in the order of ``chrom_sizes``) and position.

    """
    normalized_peaks = [
        normalize_peak_scores(
            iterative_peak_filtering(
                calculate_center_peaks_and_extend(
                    peaks.select(PEAK_COLUMNS), chrom_sizes, peak_half_width
                ),
                chrom_sizes,
            )
        )
        for peaks in peaks_per_sample.values()
    ]
    consensus_peaks = iterative_peak_filtering(
        pl.concat(normalized_peaks), chrom_sizes
    )
    if blacklist is not None:
        consensus_peaks = remove_peaks_overlapping_blacklisted_regions(
            consensus_peaks, blacklist, chrom_sizes
        )
    return consensus_peaks.with_columns(pl.col("End") - 1)

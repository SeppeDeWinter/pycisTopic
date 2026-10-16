from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import numpy.typing as npt

# Intersection/overlap code is based on:
#   https://github.com/biocore-ntnu/pyranges/blob/master/pyranges/methods/intersection.py


def _get_start_end_and_indexes_for_chrom(
    regions_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    chrom: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Get start, end and index positions from per chromosome Polars dataframe.

    Parameters
    ----------
    regions_per_chrom_dfs_pl
        Dictionary of region dataframes partitioned by chromosome.
    chrom
        Chromosome name.

    Returns
    -------
    (starts, ends, indexes)
        Tuple of numpy arrays with starts, ends and index positions
        for the requested chromosome.

    """
    starts, ends, indexes = list(
        regions_per_chrom_dfs_pl[chrom]
        .with_row_index()
        .select(
            [
                pl.col("Start").cast(pl.Int64),
                pl.col("End").cast(pl.Int64),
                pl.col("index").cast(pl.Int64),
            ]
        )
        .to_numpy()
        .T
    )

    return starts, ends, indexes


def _intersect_per_chrom_ncls(
    regions1_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    regions2_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    chrom: str,
    how: Literal["all", "containment", "first", "last"] | str | None = None,
) -> tuple[pl.Series, pl.Series]:
    """
    Get intersection between two region sets per chromosome.

    Get intersection between regions from first set and second set of regions for a
    chromosome and return index positions for those overlaps in the first and
    second set of regions.

    Parameters
    ----------
    regions1_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for first set
        of regions.
    regions2_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for second set
        of regions.
    chrom
        Chromosome name.
    how
        What intervals to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.
          - ``"last"``: last overlap with second set of regions.
          - ``"outer"``: all regions for first and all regions of second (outer join).
            If no overlap was found for a region, the other region set will contain
            ``None`` for that entry.
          - ``"left"``: all first set of regions and overlap with second set of regions
            (left join).
            If no overlap was found for a region in the first set, the second region
            set will contain None for that entry.
          - ``"right"``: all second set of regions and overlap with first set of regions
            (right join).
            If no overlap was found for a region in the second set, the first region
            set will contain ``None`` for that entry.

    Returns
    -------
    (regions1_indexes, regions2_indexes)
        Tuple of indexes for regions from Polars Dataframe 1 and indexes for regions
        from Polars Dataframe 2 that have an overlap.

    """
    from ncls import NCLS

    starts2, ends2, indexes2 = _get_start_end_and_indexes_for_chrom(
        regions2_per_chrom_dfs_pl,
        chrom=chrom,
    )

    oncls = NCLS(starts2, ends2, indexes2)

    indexes2_length = len(indexes2)

    del starts2, ends2, indexes2

    starts1, ends1, indexes1 = _get_start_end_and_indexes_for_chrom(
        regions1_per_chrom_dfs_pl,
        chrom=chrom,
    )

    if not how or how is None or how == "all":
        regions1_indexes, regions2_indexes = oncls.all_overlaps_both(
            starts1, ends1, indexes1
        )
    elif how == "containment":
        regions1_indexes, regions2_indexes = oncls.all_containments_both(
            starts1, ends1, indexes1
        )
    elif how == "first":
        regions1_indexes, regions2_indexes = oncls.first_overlap_both(
            starts1, ends1, indexes1
        )
    elif how == "last":
        regions1_indexes, regions2_indexes = oncls.last_overlap_both(
            starts1, ends1, indexes1
        )
    elif how in {"outer", "left", "right"}:
        regions1_indexes, regions2_indexes = oncls.all_overlaps_both(
            starts1, ends1, indexes1
        )

        indexes1_length = len(indexes1)

        del starts1, ends1, indexes1, oncls

        regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())
        regions2_indexes = pl.Series("idx", regions2_indexes, dtype=pl.get_index_type())

        regions1_all_indexes = pl.arange(
            0, indexes1_length, dtype=pl.get_index_type(), eager=True
        ).alias("idx")
        regions2_all_indexes = pl.arange(
            0, indexes2_length, dtype=pl.get_index_type(), eager=True
        ).alias("idx")

        regions1_missing_indexes = (
            regions1_all_indexes.to_frame()
            .join(
                regions1_indexes.to_frame(),
                on="idx",
                how="anti",
            )
            .to_series()
        )

        regions2_missing_indexes = (
            regions2_all_indexes.to_frame()
            .join(
                regions2_indexes.to_frame(),
                on="idx",
                how="anti",
            )
            .to_series()
        )

        regions1_none_indexes = pl.select(
            pl.repeat(
                None,
                n=regions2_missing_indexes.len(),
                dtype=pl.get_index_type(),
            ).alias("idx")
        ).to_series()
        regions2_none_indexes = pl.select(
            pl.repeat(
                None,
                n=regions1_missing_indexes.len(),
                dtype=pl.get_index_type(),
            ).alias("idx")
        ).to_series()

        if how == "outer":
            regions1_indexes = pl.concat(
                [
                    regions1_indexes,
                    regions1_missing_indexes,
                    regions1_none_indexes,
                ]
            )
            regions2_indexes = pl.concat(
                [
                    regions2_indexes,
                    regions2_none_indexes,
                    regions2_missing_indexes,
                ]
            )
        elif how == "left":
            regions1_indexes = pl.concat([regions1_indexes, regions1_missing_indexes])
            regions2_indexes = pl.concat([regions2_indexes, regions2_none_indexes])
        elif how == "right":
            regions1_indexes = pl.concat([regions1_indexes, regions1_none_indexes])
            regions2_indexes = pl.concat([regions2_indexes, regions2_missing_indexes])

        return regions1_indexes, regions2_indexes

    del starts1, ends1, indexes1, oncls

    regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())
    regions2_indexes = pl.Series("idx", regions2_indexes, dtype=pl.get_index_type())

    return regions1_indexes, regions2_indexes


def _overlap_per_chrom_ncls(
    regions1_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    regions2_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    chrom: str,
    how: Literal["all", "containment", "first"] | str | None = "first",
) -> npt.NDArray[np.int64]:
    """
    Get overlap between two region sets per chromosome.

    Get overlap between regions from first set and second set of regions for a
    chromosome and return index positions for those overlaps in the first set
    of regions.

    Parameters
    ----------
    regions1_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for first set
        of regions.
    regions2_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for second set
        of regions.
    chrom
        Chromosome name.
    how
        What intervals to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.

    Returns
    -------
    regions1_indexes
        Indexes for regions from Polars Dataframe 1 that had an overlap.

    """
    from ncls import NCLS

    starts2, ends2, indexes2 = _get_start_end_and_indexes_for_chrom(
        regions2_per_chrom_dfs_pl,
        chrom=chrom,
    )

    oncls = NCLS(starts2, ends2, indexes2)

    del starts2, ends2, indexes2

    starts1, ends1, indexes1 = _get_start_end_and_indexes_for_chrom(
        regions1_per_chrom_dfs_pl,
        chrom=chrom,
    )

    if not how or how is None or how == "all":
        regions1_indexes = oncls.all_overlaps_self(starts1, ends1, indexes1)
    elif how == "containment":
        regions1_indexes, _ = oncls.all_containments_both(starts1, ends1, indexes1)
    elif how == "first":
        regions1_indexes = oncls.has_overlaps(starts1, ends1, indexes1)

    del starts1, ends1, indexes1

    return regions1_indexes


def _intersect_per_chrom_ruranges(
    regions1_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    regions2_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    chrom: str,
    how: Literal["all", "containment", "first", "last"] | str | None = None,
) -> tuple[pl.Series, pl.Series]:
    """
    Get intersection between two region sets per chromosome.

    Get intersection between regions from first set and second set of regions for a
    chromosome and return index positions for those overlaps in the first and
    second set of regions.

    Parameters
    ----------
    regions1_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for first set
        of regions.
    regions2_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for second set
        of regions.
    chrom
        Chromosome name.
    how
        What intervals to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.
          - ``"last"``: last overlap with second set of regions.
          - ``"outer"``: all regions for first and all regions of second (outer join).
            If no overlap was found for a region, the other region set will contain
            ``None`` for that entry.
          - ``"left"``: all first set of regions and overlap with second set of regions
            (left join).
            If no overlap was found for a region in the first set, the second region
            set will contain None for that entry.
          - ``"right"``: all second set of regions and overlap with first set of regions
            (right join).
            If no overlap was found for a region in the second set, the first region
            set will contain ``None`` for that entry.

    Returns
    -------
    (regions1_indexes, regions2_indexes)
        Tuple of indexes for regions from Polars Dataframe 1 and indexes for regions
        from Polars Dataframe 2 that have an overlap.

    """
    import ruranges

    starts1, ends1, indexes1 = _get_start_end_and_indexes_for_chrom(
        regions1_per_chrom_dfs_pl,
        chrom=chrom,
    )
    starts2, ends2, indexes2 = _get_start_end_and_indexes_for_chrom(
        regions2_per_chrom_dfs_pl,
        chrom=chrom,
    )

    contained = False

    if not how or how is None or how == "all":
        multiple = "all"
    elif how == "containment":
        multiple = "all"
        contained = True
    elif how == "first":
        multiple = "first"
    elif how == "last":
        multiple = "last"
    elif how in {"outer", "left", "right"}:
        multiple = "all"
    else:
        raise ValueError(
            f"Invalid value for `how`: {how}. "
            "Valid values are: 'all', 'containment', 'first', 'last', "
            "'outer', 'left', 'right'."
        )

    regions1_indexes, regions2_indexes = ruranges.overlaps_simple(
        starts=starts1,
        ends=ends1,
        starts2=starts2,
        ends2=ends2,
        groups=None,
        groups2=None,
        multiple=multiple,
        contained=contained,
        sort_output=False,
        slack=0,
    )

    indexes1_length = len(indexes1)
    indexes2_length = len(indexes2)
    del starts1, ends1, indexes1, starts2, ends2, indexes2

    if how in {"outer", "left", "right"}:
        regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())
        regions2_indexes = pl.Series("idx", regions2_indexes, dtype=pl.get_index_type())

        regions1_all_indexes = pl.arange(
            0, indexes1_length, dtype=pl.get_index_type(), eager=True
        ).alias("idx")
        regions2_all_indexes = pl.arange(
            0, indexes2_length, dtype=pl.get_index_type(), eager=True
        ).alias("idx")

        regions1_missing_indexes = (
            regions1_all_indexes.to_frame()
            .join(
                regions1_indexes.to_frame(),
                on="idx",
                how="anti",
            )
            .to_series()
        )

        regions2_missing_indexes = (
            regions2_all_indexes.to_frame()
            .join(
                regions2_indexes.to_frame(),
                on="idx",
                how="anti",
            )
            .to_series()
        )

        regions1_none_indexes = pl.select(
            pl.repeat(
                None,
                n=regions2_missing_indexes.len(),
                dtype=pl.get_index_type(),
            ).alias("idx")
        ).to_series()
        regions2_none_indexes = pl.select(
            pl.repeat(
                None,
                n=regions1_missing_indexes.len(),
                dtype=pl.get_index_type(),
            ).alias("idx")
        ).to_series()

        if how == "outer":
            regions1_indexes = pl.concat(
                [
                    regions1_indexes,
                    regions1_missing_indexes,
                    regions1_none_indexes,
                ]
            )
            regions2_indexes = pl.concat(
                [
                    regions2_indexes,
                    regions2_none_indexes,
                    regions2_missing_indexes,
                ]
            )
        elif how == "left":
            regions1_indexes = pl.concat([regions1_indexes, regions1_missing_indexes])
            regions2_indexes = pl.concat([regions2_indexes, regions2_none_indexes])
        elif how == "right":
            regions1_indexes = pl.concat([regions1_indexes, regions1_none_indexes])
            regions2_indexes = pl.concat([regions2_indexes, regions2_missing_indexes])

        return regions1_indexes, regions2_indexes

    regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())
    regions2_indexes = pl.Series("idx", regions2_indexes, dtype=pl.get_index_type())

    return regions1_indexes, regions2_indexes


def _overlap_per_chrom_ruranges(
    regions1_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    regions2_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    chrom: str,
    how: Literal["all", "containment", "first", "last"] | str | None = "first",
) -> pl.Series:
    """
    Get overlap between two region sets per chromosome.

    Get overlap between regions from first set and second set of regions for a
    chromosome and return index positions for those overlaps in the first set
    of regions.

    Parameters
    ----------
    regions1_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for first set
        of regions.
    regions2_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for second set
        of regions.
    chrom
        Chromosome name.
    how
        What intervals to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.
          - ``"last"``: last overlap with second set of regions.

    Returns
    -------
    regions1_indexes
        Indexes for regions from Polars Dataframe 1 that had an overlap.

    """
    import ruranges

    starts1, ends1, indexes1 = _get_start_end_and_indexes_for_chrom(
        regions1_per_chrom_dfs_pl,
        chrom=chrom,
    )
    starts2, ends2, indexes2 = _get_start_end_and_indexes_for_chrom(
        regions2_per_chrom_dfs_pl,
        chrom=chrom,
    )

    contained = False

    if not how or how is None or how == "all":
        multiple = "all"
    elif how == "containment":
        multiple = "all"
        contained = True
    elif how == "first":
        multiple = "first"
    elif how == "last":
        multiple = "last"
    else:
        raise ValueError(
            f"Invalid value for `how`: {how}. "
            "Valid values are: 'all', 'containment', 'first', 'last'."
        )

    regions1_indexes, regions2_indexes = ruranges.overlaps_simple(
        starts=starts1,
        ends=ends1,
        starts2=starts2,
        ends2=ends2,
        groups=None,
        groups2=None,
        multiple=multiple,
        contained=contained,
        sort_output=False,
        slack=0,
    )

    del starts1, ends1, indexes1, starts2, ends2, indexes2

    regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())

    return regions1_indexes


@overload
def _filter_intersection_output_columns(
    df: pl.DataFrame,
    regions1_info: bool,
    regions2_info: bool,
    regions1_coord: bool,
    regions2_coord: bool,
    regions1_suffix: str,
    regions2_suffix: str,
    add_overlap_size: bool,
) -> pl.DataFrame: ...


@overload
def _filter_intersection_output_columns(
    df: pl.LazyFrame,
    regions1_info: bool,
    regions2_info: bool,
    regions1_coord: bool,
    regions2_coord: bool,
    regions1_suffix: str,
    regions2_suffix: str,
    add_overlap_size: bool,
) -> pl.LazyFrame: ...


def _filter_intersection_output_columns(
    df: pl.DataFrame | pl.LazyFrame,
    regions1_info: bool,
    regions2_info: bool,
    regions1_coord: bool,
    regions2_coord: bool,
    regions1_suffix: str,
    regions2_suffix: str,
    add_overlap_size: bool,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Filter intersection output columns.

    Parameters
    ----------
    df
        Polars DataFrame or LazyFrame with intersection results.
    regions1_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions2_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions1_coord
        Add coordinates from first set of regions to output of intersection.
    regions2_coord
        Add coordinates from second set of regions to output of intersection.
    regions1_suffix
        Suffix added to coordinate columns of first set of regions.
    regions2_suffix
        Suffix added to coordinate and info columns of second set of regions.
    add_overlap_size
        Add columns with absolute and relative overlap size to output of intersection.

    Returns
    -------
    Polars LazyFrame with intersection results with only the requested columns.

    """
    # Get coordinate column names for first set of regions.
    regions1_coord_columns = [
        f"Chromosome{regions1_suffix}",
        f"Start{regions1_suffix}",
        f"End{regions1_suffix}",
    ]

    # Get coordinate column names for second set of regions.
    regions2_coord_columns = [
        f"Chromosome{regions2_suffix}",
        f"Start{regions2_suffix}",
        f"End{regions2_suffix}",
    ]

    # Get info column names for first set of regions
    # (all columns except coordinate columns).
    regions1_suffix_length = len(regions1_suffix)
    regions1_info_columns = [
        # Remove region1 suffix from column names.
        pl.col(column_name).alias(column_name[:-regions1_suffix_length])
        for column_name in df.collect_schema().names()
        if (
            column_name.endswith(regions1_suffix)
            and column_name not in regions1_coord_columns
        )
    ]

    # Get info column names for second set of regions
    # (all columns except coordinate columns).
    regions2_suffix_length = len(regions2_suffix)
    regions2_info_columns = [
        # Remove region2 suffix from column names if no region1 info will be displayed.
        pl.col(column_name)
        if regions1_info
        else pl.col(column_name).alias(column_name[:-regions2_suffix_length])
        for column_name in df.collect_schema().names()
        if (
            column_name.endswith(regions2_suffix)
            and column_name not in regions2_coord_columns
        )
    ]

    select_columns = [pl.col(["Chromosome", "Start", "End"])]
    if regions1_coord:
        select_columns.append(pl.col(regions1_coord_columns))
    if regions2_coord:
        select_columns.append(pl.col(regions2_coord_columns))

    if regions1_info:
        select_columns.extend(regions1_info_columns)
    if regions2_info:
        select_columns.extend(regions2_info_columns)

    if add_overlap_size:
        select_columns.extend([pl.col(["nb_inter", "fr1_inter", "fr2_inter"])])

    return df.select(select_columns)


@overload
def _get_intersection_size(
    intersect: pl.LazyFrame,
    regions1_suffix: str,
    regions2_suffix: str,
) -> pl.LazyFrame: ...


@overload
def _get_intersection_size(
    intersect: pl.DataFrame,
    regions1_suffix: str,
    regions2_suffix: str,
) -> pl.DataFrame: ...


def _get_intersection_size(
    intersect: pl.LazyFrame | pl.DataFrame,
    regions1_suffix: str = "@1",
    regions2_suffix: str = "@2",
) -> pl.LazyFrame | pl.DataFrame:
    """
    Calculate intersect size both in absolute terms and as a fraction of regions1 and regions2.

    Parameters
    ----------
    intersect
        output from intersect function.
    regions1_suffix
        Suffix added to coordinate columns of first set of regions.
    regions2_suffix
        Suffix added to coordinate and info columns of second set of regions.

    Returns
    -------
    Polars dataframe with:
     - absolute intersect (`nb_inter`)
     - intersect as fraction of regions1 (`fr1_inter`)
     - intersect as fraction of regions2 (`fr2_inter`)

    Examples
    --------
    >>> intersect
        shape: (3, 10)
    ┌────────────┬───────┬─────┬──────────────┬─────────┬───────┬──────────────┬─────────┬───────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ Chromosome@1 ┆ Start@1 ┆ End@1 ┆ Chromosome@2 ┆ Start@2 ┆ End@2 ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ ---          ┆ ---     ┆ ---   ┆ ---          ┆ ---     ┆ ---   ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str          ┆ i64     ┆ i64   ┆ str          ┆ i64     ┆ i64   ┆ str │
    ╞════════════╪═══════╪═════╪══════════════╪═════════╪═══════╪══════════════╪═════════╪═══════╪═════╡
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 1       ┆ 3     ┆ chr1         ┆ 2       ┆ 9     ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 1       ┆ 3     ┆ chr1         ┆ 2       ┆ 3     ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ chr1         ┆ 4       ┆ 9     ┆ chr1         ┆ 2       ┆ 9     ┆ b   │
    └────────────┴───────┴─────┴──────────────┴─────────┴───────┴──────────────┴─────────┴───────┴─────┘

    >>> _get_intersection_size(intersect)
        shape: (3, 13)
    ┌────────────┬───────┬─────┬───────────┬───┬─────┬──────────┬───────────┬───────────┐
    │ Chromosome ┆ Start ┆ End ┆ Chromosom ┆ … ┆ ID  ┆ nb_inter ┆ fr1_inter ┆ fr2_inter │
    │ ---        ┆ ---   ┆ --- ┆ e@1       ┆   ┆ --- ┆ ---      ┆ ---       ┆ ---       │
    │ str        ┆ i64   ┆ i64 ┆ ---       ┆   ┆ str ┆ i64      ┆ f64       ┆ f64       │
    │            ┆       ┆     ┆ str       ┆   ┆     ┆          ┆           ┆           │
    ╞════════════╪═══════╪═════╪═══════════╪═══╪═════╪══════════╪═══════════╪═══════════╡
    │ chr1       ┆ 2     ┆ 3   ┆ chr1      ┆ … ┆ a   ┆ 1        ┆ 0.5       ┆ 0.142857  │
    │ chr1       ┆ 2     ┆ 3   ┆ chr1      ┆ … ┆ a   ┆ 1        ┆ 0.5       ┆ 1.0       │
    │ chr1       ┆ 4     ┆ 9   ┆ chr1      ┆ … ┆ b   ┆ 5        ┆ 1.0       ┆ 0.714286  │
    └────────────┴───────┴─────┴───────────┴───┴─────┴──────────┴───────────┴───────────┘

    """
    intersect = intersect.with_columns(
        (pl.col("End") - pl.col("Start")).alias("nb_inter"),
    )

    return intersect.with_columns(
        (
            pl.col("nb_inter")
            / (pl.col(f"End{regions1_suffix}") - pl.col(f"Start{regions1_suffix}"))
        ).alias("fr1_inter"),
        (
            pl.col("nb_inter")
            / (pl.col(f"End{regions2_suffix}") - pl.col(f"Start{regions2_suffix}"))
        ).alias("fr2_inter"),
    )


def intersection(
    regions1_df_pl: pl.DataFrame,
    regions2_df_pl: pl.DataFrame,
    how: Literal["all", "containment", "first", "last"] | str | None = None,
    regions1_info: bool = True,
    regions2_info: bool = False,
    regions1_coord: bool = False,
    regions2_coord: bool = False,
    regions1_suffix: str = "@1",
    regions2_suffix: str = "@2",
    engine: Literal["ncls", "ruranges"] = "ncls",
    add_overlap_size: bool = False,
) -> pl.DataFrame:
    """
    Get overlapping subintervals between first set and second set of regions.

    Parameters
    ----------
    regions1_df_pl
        Polars DataFrame containing BED entries for first set of regions.
    regions2_df_pl
        Polars DataFrame containing BED entries for second set of regions.
    how
        What intervals to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.
          - ``"last"``: last overlap with second set of regions.
          - ``"outer"``: all regions for first and all regions of second (outer join).
            If no overlap was found for a region, the other region set will contain
            ``None`` for that entry.
          - ``"left"``: all first set of regions and overlap with second set of regions
            (left join).
            If no overlap was found for a region in the first set, the second region
            set will contain None for that entry.
          - ``"right"``: all second set of regions and overlap with first set of regions
            (right join).
            If no overlap was found for a region in the second set, the first region
            set will contain ``None`` for that entry.
    regions1_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions2_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions1_coord
        Add coordinates from first set of regions to output of intersection.
    regions2_coord
        Add coordinates from second set of regions to output of intersection.
    regions1_suffix
        Suffix added to coordinate columns of first set of regions.
    regions2_suffix
        Suffix added to coordinate and info columns of second set of regions.
    engine
        Engine to use for intersection: ``"ncls"`` or ``"ruranges"``.
    add_overlap_size
        Wether to add overlap size as the absolute number of basepairs and
        as a fraction of region1 and region2.
    strandedness
        Note: Not implemented yet.
        {``None``, ``"same"``, ``"opposite"``, ``False``}, default ``None``, i.e. auto
        Whether to compare PyRanges on the same strand, the opposite or ignore strand
        information. The default, ``None``, means use ``"same"`` if both PyRanges are
        stranded, otherwise ignore the strand information.

    Returns
    -------
    intersection_df_pl
        Polars Dataframe containing BED entries with the intersection.

    Examples
    --------
    >>> regions1_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 3,
    ...         "Start": [1, 4, 10],
    ...         "End": [3, 9, 11],
    ...         "ID": ["a", "b", "c"],
    ...     }
    ... )
    >>> regions1_df_pl
    shape: (3, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 10    ┆ 11  ┆ c   │
    └────────────┴───────┴─────┴─────┘

    >>> regions2_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 3,
    ...         "Start": [2, 2, 9],
    ...         "End": [3, 9, 10],
    ...         "Name": ["reg1", "reg2", "reg3"]
    ...     }
    ... )
    >>> regions2_df_pl
    shape: (3, 4)
    ┌────────────┬───────┬─────┬──────┐
    │ Chromosome ┆ Start ┆ End ┆ Name │
    │ ---        ┆ ---   ┆ --- ┆ ---  │
    │ str        ┆ i64   ┆ i64 ┆ str  │
    ╞════════════╪═══════╪═════╪══════╡
    │ chr1       ┆ 2     ┆ 3   ┆ reg1 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 9   ┆ reg2 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 9     ┆ 10  ┆ reg3 │
    └────────────┴───────┴─────┴──────┘

    >>> intersection(regions1_df_pl, regions2_df_pl)
    shape: (3, 3)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 2     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> intersection(regions1_df_pl, regions2_df_pl, how="first")
    shape: (2, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 2     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> intersection(
    ...     regions1_df_pl,
    ...     regions2_df_pl,
    ...     how="containment",
    ...     regions1_info=False,
    ...     regions2_info=True,
    ... )
    shape: (1, 4)
    ┌────────────┬───────┬─────┬──────┐
    │ Chromosome ┆ Start ┆ End ┆ Name │
    │ ---        ┆ ---   ┆ --- ┆ ---  │
    │ str        ┆ i64   ┆ i64 ┆ str  │
    ╞════════════╪═══════╪═════╪══════╡
    │ chr1       ┆ 4     ┆ 9   ┆ reg2 │
    └────────────┴───────┴─────┴──────┘

    >>> intersection(
    ...     regions1_df_pl,
    ...     regions2_df_pl,
    ...     regions1_coord=True,
    ...     regions2_coord=True,
    ... )
    shape: (3, 10)
    ┌────────────┬───────┬─────┬──────────────┬─────────┬───────┬──────────────┬─────────┬───────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ Chromosome@1 ┆ Start@1 ┆ End@1 ┆ Chromosome@2 ┆ Start@2 ┆ End@2 ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ ---          ┆ ---     ┆ ---   ┆ ---          ┆ ---     ┆ ---   ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str          ┆ i64     ┆ i64   ┆ str          ┆ i64     ┆ i64   ┆ str │
    ╞════════════╪═══════╪═════╪══════════════╪═════════╪═══════╪══════════════╪═════════╪═══════╪═════╡
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 1       ┆ 3     ┆ chr1         ┆ 2       ┆ 9     ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 1       ┆ 3     ┆ chr1         ┆ 2       ┆ 3     ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ chr1         ┆ 4       ┆ 9     ┆ chr1         ┆ 2       ┆ 9     ┆ b   │
    └────────────┴───────┴─────┴──────────────┴─────────┴───────┴──────────────┴─────────┴───────┴─────┘

    >>> intersection(
    ...     regions1_df_pl,
    ...     regions2_df_pl,
    ...     regions1_info=False,
    ...     regions_info=True,
    ...     regions2_coord=True,
    ... )
    shape: (3, 7)
    ┌────────────┬───────┬─────┬──────────────┬─────────┬───────┬──────┐
    │ Chromosome ┆ Start ┆ End ┆ Chromosome@2 ┆ Start@2 ┆ End@2 ┆ Name │
    │ ---        ┆ ---   ┆ --- ┆ ---          ┆ ---     ┆ ---   ┆ ---  │
    │ str        ┆ i64   ┆ i64 ┆ str          ┆ i64     ┆ i64   ┆ str  │
    ╞════════════╪═══════╪═════╪══════════════╪═════════╪═══════╪══════╡
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 2       ┆ 9     ┆ reg2 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 2       ┆ 3     ┆ reg1 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ chr1         ┆ 2       ┆ 9     ┆ reg2 │
    └────────────┴───────┴─────┴──────────────┴─────────┴───────┴──────┘

    """
    # TODO: chrom, stranded partitioning
    regions1_per_chrom_dfs_pl = {
        str(chrom): regions1_chrom_df_pl
        for (chrom,), regions1_chrom_df_pl in regions1_df_pl.partition_by(
            ["Chromosome"],
            as_dict=True,
            maintain_order=True,
        ).items()
    }

    regions2_per_chrom_dfs_pl = {
        str(chrom): regions2_chrom_df_pl
        for (chrom,), regions2_chrom_df_pl in regions2_df_pl.partition_by(
            ["Chromosome"],
            as_dict=True,
            maintain_order=True,
        ).items()
    }

    intersection_chrom_dfs_pl = {}

    for chrom in list(regions1_per_chrom_dfs_pl.keys()):
        if chrom in list(regions2_per_chrom_dfs_pl.keys()):
            # Find intersection between regions from first and second per chromosome
            # dataframe and return index positions in both dataframes for those
            # intersections.
            regions1_indexes, regions2_indexes = (
                _intersect_per_chrom_ruranges(
                    regions1_per_chrom_dfs_pl=regions1_per_chrom_dfs_pl,
                    regions2_per_chrom_dfs_pl=regions2_per_chrom_dfs_pl,
                    chrom=chrom,
                    how=how,
                )
                if engine == "ruranges"
                else _intersect_per_chrom_ncls(
                    regions1_per_chrom_dfs_pl=regions1_per_chrom_dfs_pl,
                    regions2_per_chrom_dfs_pl=regions2_per_chrom_dfs_pl,
                    chrom=chrom,
                    how=how,
                )
            )

            # Skip empty intersections.
            if regions1_indexes.shape[0] == 0:
                continue

            # Get all regions from first and second per chromosome dataframe for the
            # index positions calculated above.
            intersection_chrom_df_pl = (
                regions1_per_chrom_dfs_pl.pop(chrom)[regions1_indexes]
                .select(pl.all().name.suffix(regions1_suffix))
                .hstack(
                    (regions2_per_chrom_dfs_pl.pop(chrom)[regions2_indexes]).select(
                        pl.all().name.suffix(regions2_suffix)
                    )
                )
            )

            # Calculate intersection start and end coordinates and return the columns
            # of interest.
            intersection_chrom_ldf_pl = (
                intersection_chrom_df_pl.lazy()
                .with_columns(
                    [
                        # Chromosome name for intersection.
                        pl.coalesce(
                            pl.col(f"Chromosome{regions1_suffix}"),
                            pl.col(f"Chromosome{regions2_suffix}"),
                        ).alias("Chromosome"),
                        # Calculate start coordinate for intersection.
                        pl.when(
                            pl.col(f"Start{regions1_suffix}")
                            > pl.col(f"Start{regions2_suffix}")
                        )
                        .then(pl.col(f"Start{regions1_suffix}"))
                        .otherwise(
                            pl.coalesce(
                                pl.col(f"Start{regions2_suffix}"),
                                pl.col(f"Start{regions1_suffix}"),
                            )
                        )
                        .alias("Start"),
                        # Calculate end coordinate for intersection.
                        pl.when(
                            pl.col(f"End{regions1_suffix}")
                            < pl.col(f"End{regions2_suffix}")
                        )
                        .then(pl.col(f"End{regions1_suffix}"))
                        .otherwise(
                            pl.coalesce(
                                pl.col(f"End{regions2_suffix}"),
                                pl.col(f"End{regions1_suffix}"),
                            )
                        )
                        .alias("End"),
                    ]
                )
                .pipe(
                    function=_get_intersection_size,
                    regions1_suffix=regions1_suffix,
                    regions2_suffix=regions2_suffix,
                )
                .pipe(
                    function=_filter_intersection_output_columns,
                    regions1_info=regions1_info,
                    regions2_info=regions2_info,
                    regions1_coord=regions1_coord,
                    regions2_coord=regions2_coord,
                    regions1_suffix=regions1_suffix,
                    regions2_suffix=regions2_suffix,
                    add_overlap_size=add_overlap_size,
                )
            )

            intersection_chrom_dfs_pl[chrom] = intersection_chrom_ldf_pl.collect()

    if len(list(intersection_chrom_dfs_pl.values())) == 0:
        # Return empty dataframe if there was no shared chromosome.
        return (
            regions1_df_pl.select(pl.all().name.suffix(regions1_suffix))
            .head(0)
            .hstack(
                regions2_df_pl.select(pl.all().name.suffix(regions2_suffix)).head(0)
            )
            .lazy()
            .with_columns(
                [
                    # Chromosome name for intersection.
                    pl.coalesce(
                        pl.col(f"Chromosome{regions1_suffix}"),
                        pl.col(f"Chromosome{regions2_suffix}"),
                    ).alias("Chromosome"),
                    # Calculate start coordinate for intersection.
                    pl.when(
                        pl.col(f"Start{regions1_suffix}")
                        > pl.col(f"Start{regions2_suffix}")
                    )
                    .then(pl.col(f"Start{regions1_suffix}"))
                    .otherwise(
                        pl.coalesce(
                            pl.col(f"Start{regions2_suffix}"),
                            pl.col(f"Start{regions1_suffix}"),
                        )
                    )
                    .alias("Start"),
                    # Calculate end coordinate for intersection.
                    pl.when(
                        pl.col(f"End{regions1_suffix}")
                        < pl.col(f"End{regions2_suffix}")
                    )
                    .then(pl.col(f"End{regions1_suffix}"))
                    .otherwise(
                        pl.coalesce(
                            pl.col(f"End{regions2_suffix}"),
                            pl.col(f"End{regions1_suffix}"),
                        )
                    )
                    .alias("End"),
                ]
            )
            .pipe(
                function=_get_intersection_size,
                regions1_suffix=regions1_suffix,
                regions2_suffix=regions2_suffix,
            )
            .pipe(
                function=_filter_intersection_output_columns,
                regions1_info=regions1_info,
                regions2_info=regions2_info,
                regions1_coord=regions1_coord,
                regions2_coord=regions2_coord,
                regions1_suffix=regions1_suffix,
                regions2_suffix=regions2_suffix,
                add_overlap_size=add_overlap_size,
            )
            .collect()
        )

    # Combine per chromosome dataframes to a full one.
    intersection_df_pl = pl.concat(
        list(intersection_chrom_dfs_pl.values()),
        rechunk=False,
    )

    return intersection_df_pl


def overlap(
    regions1_df_pl: pl.DataFrame,
    regions2_df_pl: pl.DataFrame,
    how: Literal["all", "containment", "first"] | str | None = "first",
    invert: bool = False,
    engine: Literal["ncls", "ruranges"] = "ncls",
) -> pl.DataFrame:
    """
    Get overlap between two region sets.

    Get overlap between first set and second set of regions and return interval of
    first set of regions.

    Parameters
    ----------
    regions1_df_pl
        Polars DataFrame containing BED entries for first set of regions.
    regions2_df_pl
        Polars DataFrame containing BED entries for second set of regions.
    how
        What overlaps to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.
    invert
        Whether to return the intervals without overlaps.
    engine
        Engine to use for intersection: ``"ncls"`` or ``"ruranges"``.

    strandedness
        Note: Not implemented yet.
        {``None``, ``"same"``, ``"opposite"``, ``False``}, default ``None``, i.e. auto
        Whether to compare PyRanges on the same strand, the opposite or ignore strand
        information. The default, ``None``, means use ``"same"`` if both PyRanges are
        stranded, otherwise ignore the strand information.

    Returns
    -------
    overlap_df_pl
        Polars Dataframe containing BED entries with the overlap.

    Examples
    --------
    >>> regions1_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 3,
    ...         "Start": [1, 4, 10],
    ...         "End": [3, 9, 11],
    ...         "ID": ["a", "b", "c"],
    ...     }
    ... )
    >>> regions1_df_pl
    shape: (3, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 10    ┆ 11  ┆ c   │
    └────────────┴───────┴─────┴─────┘

    >>> regions2_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 3,
    ...         "Start": [2, 2, 9],
    ...         "End": [3, 9, 10],
    ...         "Name": ["reg1", "reg2", "reg3"]
    ...     }
    ... )
    >>> regions2_df_pl
    shape: (3, 4)
    ┌────────────┬───────┬─────┬──────┐
    │ Chromosome ┆ Start ┆ End ┆ Name │
    │ ---        ┆ ---   ┆ --- ┆ ---  │
    │ str        ┆ i64   ┆ i64 ┆ str  │
    ╞════════════╪═══════╪═════╪══════╡
    │ chr1       ┆ 2     ┆ 3   ┆ reg1 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 9   ┆ reg2 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 9     ┆ 10  ┆ reg3 │
    └────────────┴───────┴─────┴──────┘

    >>> overlap(regions1_df_pl, regions2_df_pl, how="first")
    shape: (2, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> overlap(regions1_df_pl, regions2_df_pl, how="all")
    shape: (3, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> overlap(regions1_df_pl, regions2_df_pl, how="containment")
    shape: (1, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> overlap(regions1_df_pl, regions2_df_pl, how="containment", invert=True)
    shape: (2, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 10    ┆ 11  ┆ c   │
    └────────────┴───────┴─────┴─────┘

    """
    # TODO: chrom, stranded partitioning
    regions1_per_chrom_dfs_pl = {
        str(chrom): regions1_chrom_df_pl
        for (chrom,), regions1_chrom_df_pl in regions1_df_pl.partition_by(
            ["Chromosome"],
            as_dict=True,
            maintain_order=True,
        ).items()
    }

    regions2_per_chrom_dfs_pl = {
        str(chrom): regions2_chrom_df_pl
        for (chrom,), regions2_chrom_df_pl in regions2_df_pl.partition_by(
            ["Chromosome"],
            as_dict=True,
            maintain_order=True,
        ).items()
    }

    overlap_chrom_dfs_pl = {}

    for chrom, regions1_chrom_df_pl in regions1_per_chrom_dfs_pl.items():
        if chrom in regions2_per_chrom_dfs_pl:
            # Find overlap between regions from first and second per chromosome
            # dataframe and return index positions in first dataframe for those
            # overlaps.
            regions1_indexes = (
                _overlap_per_chrom_ruranges(
                    regions1_per_chrom_dfs_pl=regions1_per_chrom_dfs_pl,
                    regions2_per_chrom_dfs_pl=regions2_per_chrom_dfs_pl,
                    chrom=chrom,
                    how=how,
                )
                if engine == "ruranges"
                else _overlap_per_chrom_ncls(
                    regions1_per_chrom_dfs_pl=regions1_per_chrom_dfs_pl,
                    regions2_per_chrom_dfs_pl=regions2_per_chrom_dfs_pl,
                    chrom=chrom,
                    how=how,
                )
            )
        else:
            # No regions on this chromosome in the second set, so nothing overlaps.
            regions1_indexes = np.empty((0,), dtype=np.int64)

        if invert:
            # Get inverse selection of regions from first dataframe for the overlap.
            # Regions on chromosomes without any overlap are all kept.
            keep_mask = np.ones(regions1_chrom_df_pl.height, dtype=bool)
            keep_mask[regions1_indexes] = False
            overlap_chrom_df_pl = regions1_chrom_df_pl.filter(pl.Series(keep_mask))
        else:
            # Get selection of regions from first dataframe for the overlap.
            overlap_chrom_df_pl = regions1_chrom_df_pl[regions1_indexes]

        # Skip empty overlaps.
        if overlap_chrom_df_pl.height == 0:
            continue

        overlap_chrom_dfs_pl[chrom] = overlap_chrom_df_pl

    if len(list(overlap_chrom_dfs_pl.values())) == 0:
        # Return empty dataframe if there was no shared chromosome.
        return regions1_df_pl.head(0)

    # Combine per chromosome dataframes to a full one.
    overlap_df_pl = pl.concat(
        list(overlap_chrom_dfs_pl.values()),
        rechunk=False,
    )

    return overlap_df_pl


def extend(
    regions_df_pl: pl.DataFrame,
    upstream: int | str = 0,
    downstream: int | str = 0,
) -> pl.DataFrame:
    """
    Extend regions upstream and downstream, taking the strand into account.

    For regions on the ``"-"`` strand, upstream is towards higher coordinates.
    All other regions (``"+"``, ``"."``) are extended as if on the ``"+"`` strand.

    Parameters
    ----------
    regions_df_pl
        Polars DataFrame containing BED entries with a ``Strand`` column.
    upstream
        Number of basepairs to extend upstream, or name of a column with the number
        of basepairs to extend upstream per region.
    downstream
        Number of basepairs to extend downstream, or name of a column with the
        number of basepairs to extend downstream per region.

    Returns
    -------
    Polars DataFrame with extended regions. Start and End keep their original
    data type.

    Examples
    --------
    >>> regions_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 2,
    ...         "Start": [100, 100],
    ...         "End": [200, 200],
    ...         "Strand": ["+", "-"],
    ...     }
    ... )
    >>> extend(regions_df_pl, upstream=10, downstream=5)
    shape: (2, 4)
    ┌────────────┬───────┬─────┬────────┐
    │ Chromosome ┆ Start ┆ End ┆ Strand │
    │ ---        ┆ ---   ┆ --- ┆ ---    │
    │ str        ┆ i64   ┆ i64 ┆ str    │
    ╞════════════╪═══════╪═════╪════════╡
    │ chr1       ┆ 90    ┆ 205 ┆ +      │
    │ chr1       ┆ 95    ┆ 210 ┆ -      │
    └────────────┴───────┴─────┴────────┘

    """
    upstream_expr = pl.col(upstream) if isinstance(upstream, str) else pl.lit(upstream)
    downstream_expr = (
        pl.col(downstream) if isinstance(downstream, str) else pl.lit(downstream)
    )
    is_minus_strand = pl.col("Strand") == "-"

    return regions_df_pl.with_columns(
        (
            pl.col("Start")
            - pl.when(is_minus_strand).then(downstream_expr).otherwise(upstream_expr)
        ).cast(regions_df_pl.schema["Start"]),
        (
            pl.col("End")
            + pl.when(is_minus_strand).then(upstream_expr).otherwise(downstream_expr)
        ).cast(regions_df_pl.schema["End"]),
    )


def nearest(
    regions1_df_pl: pl.DataFrame,
    regions2_df_pl: pl.DataFrame,
    how: Literal["upstream", "downstream", "previous", "next"],
    regions2_suffix: str = "@2",
) -> pl.DataFrame:
    """
    Find the nearest non-overlapping region of the second set for each first region.

    Only regions of the second set that lie completely before (``"previous"``) or
    after (``"next"``) a region of the first set are considered. For ``"upstream"``
    and ``"downstream"``, the direction is relative to the strand of the region of
    the first set (``"-"`` strand: upstream is towards higher coordinates); the
    strand of the second set of regions is ignored.

    The distance is the gap between both regions plus one (so bookended regions
    have a distance of 1), which is the same convention as ``pyranges.nearest``.

    Parameters
    ----------
    regions1_df_pl
        Polars DataFrame containing BED entries for first set of regions. Needs a
        ``Strand`` column when ``how`` is ``"upstream"`` or ``"downstream"``.
    regions2_df_pl
        Polars DataFrame containing BED entries for second set of regions.
    how
        Direction in which to look for the nearest region:
        ``"upstream"``, ``"downstream"``, ``"previous"`` or ``"next"``.
    regions2_suffix
        Suffix added to the coordinate columns of the nearest second region.

    Returns
    -------
    First set of regions (same order) with ``Start{regions2_suffix}``,
    ``End{regions2_suffix}`` of the nearest region of the second set and the
    ``Distance`` to it. Those columns are null when no such region exists.

    Examples
    --------
    >>> genes_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 2,
    ...         "Start": [100, 100],
    ...         "End": [200, 200],
    ...         "Strand": ["+", "-"],
    ...     }
    ... )
    >>> boundaries_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 2,
    ...         "Start": [10, 250],
    ...         "End": [50, 300],
    ...     }
    ... )
    >>> nearest(genes_df_pl, boundaries_df_pl, how="upstream")
    shape: (2, 7)
    ┌────────────┬───────┬─────┬────────┬─────────┬───────┬──────────┐
    │ Chromosome ┆ Start ┆ End ┆ Strand ┆ Start@2 ┆ End@2 ┆ Distance │
    │ ---        ┆ ---   ┆ --- ┆ ---    ┆ ---     ┆ ---   ┆ ---      │
    │ str        ┆ i64   ┆ i64 ┆ str    ┆ i64     ┆ i64   ┆ i64      │
    ╞════════════╪═══════╪═════╪════════╪═════════╪═══════╪══════════╡
    │ chr1       ┆ 100   ┆ 200 ┆ +      ┆ 10      ┆ 50    ┆ 51       │
    │ chr1       ┆ 100   ┆ 200 ┆ -      ┆ 250     ┆ 300   ┆ 51       │
    └────────────┴───────┴─────┴────────┴─────────┴───────┴──────────┘

    """
    if how not in ("upstream", "downstream", "previous", "next"):
        raise ValueError(
            f'Invalid value for how ("{how}"). Use "upstream", "downstream", '
            '"previous" or "next".'
        )

    start2_col = f"Start{regions2_suffix}"
    end2_col = f"End{regions2_suffix}"

    regions2_per_chrom_dfs_pl = {
        str(chrom): regions2_chrom_df_pl
        for (chrom,), regions2_chrom_df_pl in regions2_df_pl.partition_by(
            ["Chromosome"],
            as_dict=True,
            maintain_order=True,
        ).items()
    }

    nearest_chrom_dfs_pl = []

    for (chrom,), regions1_chrom_df_pl in (
        regions1_df_pl.with_row_index("_row_idx")
        .partition_by(["Chromosome"], as_dict=True, maintain_order=True)
        .items()
    ):
        n_regions1 = regions1_chrom_df_pl.height
        nearest_start2 = np.zeros(n_regions1, dtype=np.int64)
        nearest_end2 = np.zeros(n_regions1, dtype=np.int64)
        distance = np.zeros(n_regions1, dtype=np.int64)
        found = np.zeros(n_regions1, dtype=bool)

        regions2_chrom_df_pl = regions2_per_chrom_dfs_pl.get(str(chrom))

        if regions2_chrom_df_pl is not None and regions2_chrom_df_pl.height > 0:
            start1 = regions1_chrom_df_pl.get_column("Start").to_numpy().astype(np.int64)
            end1 = regions1_chrom_df_pl.get_column("End").to_numpy().astype(np.int64)
            start2 = regions2_chrom_df_pl.get_column("Start").to_numpy().astype(np.int64)
            end2 = regions2_chrom_df_pl.get_column("End").to_numpy().astype(np.int64)

            if how in ("previous", "next"):
                look_previous = np.full(n_regions1, how == "previous")
            else:
                is_minus_strand = (
                    regions1_chrom_df_pl.get_column("Strand").to_numpy() == "-"
                )
                look_previous = (
                    ~is_minus_strand if how == "upstream" else is_minus_strand
                )

            # Previous: region of second set with the largest end <= start1.
            order_by_end2 = np.argsort(end2, kind="stable")
            end2_sorted = end2[order_by_end2]
            previous_idx = np.searchsorted(end2_sorted, start1, side="right") - 1
            has_previous = previous_idx >= 0
            previous_idx = order_by_end2[np.maximum(previous_idx, 0)]

            # Next: region of second set with the smallest start >= end1.
            order_by_start2 = np.argsort(start2, kind="stable")
            start2_sorted = start2[order_by_start2]
            next_idx = np.searchsorted(start2_sorted, end1, side="left")
            has_next = next_idx < start2.shape[0]
            next_idx = order_by_start2[np.minimum(next_idx, start2.shape[0] - 1)]

            nearest_idx = np.where(look_previous, previous_idx, next_idx)
            found = np.where(look_previous, has_previous, has_next)
            nearest_start2 = start2[nearest_idx]
            nearest_end2 = end2[nearest_idx]
            distance = np.where(
                look_previous,
                start1 - nearest_end2 + 1,
                nearest_start2 - end1 + 1,
            )

        found_series = pl.Series(found)
        nearest_chrom_dfs_pl.append(
            regions1_chrom_df_pl.with_columns(
                pl.when(found_series)
                .then(pl.Series(nearest_start2))
                .alias(start2_col),
                pl.when(found_series).then(pl.Series(nearest_end2)).alias(end2_col),
                pl.when(found_series).then(pl.Series(distance)).alias("Distance"),
            )
        )

    if len(nearest_chrom_dfs_pl) == 0:
        return regions1_df_pl.with_columns(
            pl.lit(None, dtype=pl.Int64).alias(start2_col),
            pl.lit(None, dtype=pl.Int64).alias(end2_col),
            pl.lit(None, dtype=pl.Int64).alias("Distance"),
        )

    return pl.concat(nearest_chrom_dfs_pl).sort("_row_idx").drop("_row_idx")

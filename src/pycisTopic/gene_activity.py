from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import polars as pl
import scipy.sparse as sparse

from pycisTopic.genomic_ranges import extend, intersection, nearest, overlap
from pycisTopic.imputed_accessibility import impute_accessibility_chunked

if TYPE_CHECKING:
    import logging

    import numpy.typing as npt


def _region_names_to_bed_df_pl(region_names: Sequence[str]) -> pl.DataFrame:
    """Convert region names ("chrom:start-end") to a BED Polars DataFrame."""
    return (
        pl.DataFrame({"Name": list(region_names)})
        .with_columns(
            pl.col("Name")
            # Replace first ":" with "-".
            .str.replace(":", "-")
            # Split on "-" to generate 3 parts: "Chromosome", "Start" and "End".
            .str.split_exact("-", 2)
            .struct.rename_fields(["Chromosome", "Start", "End"])
            .alias("RegionFields")
        )
        .unnest("RegionFields")
        .select(
            pl.col("Chromosome"),
            pl.col("Start").cast(pl.Int64),
            pl.col("End").cast(pl.Int64),
            pl.col("Name"),
        )
    )


def _gini_per_row(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Calculate the Gini coefficient for each row of a 2D numpy array."""
    # Based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    x = np.sort(x.astype(np.float64), axis=1)
    # Values cannot be negative.
    row_min = x[:, :1]
    x = np.where(row_min < 0, x - row_min, x)
    n = x.shape[1]
    index = np.arange(1, n + 1, dtype=np.float64)
    return ((2 * index - n - 1) * x).sum(axis=1) / (n * x.sum(axis=1))


def get_region_gene_weights(
    regions: pl.DataFrame,
    gene_annotation: pl.DataFrame,
    chrom_sizes: pl.DataFrame,
    predefined_boundaries: pl.DataFrame | None = None,
    use_gene_boundaries: bool = True,
    upstream: tuple[int, int] = (1000, 100000),
    downstream: tuple[int, int] = (1000, 100000),
    distance_weight: bool = True,
    decay_rate: float = 1.0,
    extend_gene_body_upstream: int = 5000,
    extend_gene_body_downstream: int = 0,
    gene_size_weight: bool = False,
    gene_size_scale_factor: Literal["median"] | float = "median",
    remove_promoters: bool = False,
    extend_tss: tuple[int, int] = (10, 10),
    log: logging.Logger | None = None,
) -> pl.DataFrame:
    """
    Link regions to genes and calculate gene size and distance weights.

    Parameters
    ----------
    regions
        Polars DataFrame with ``Chromosome``, ``Start``, ``End`` and ``Name``
        columns.
    gene_annotation
        Polars DataFrame with gene annotation as returned by
        :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl`
        (``Chromosome``, ``Gene_start``, ``Gene_end``, ``Gene``, ``Strand`` and
        ``TSS_start`` columns are used).
    chrom_sizes
        Polars DataFrame with ``Chromosome``, ``Start`` (0) and ``End`` (chromosome
        size) columns.
    predefined_boundaries
        Polars DataFrame (``Chromosome``, ``Start``, ``End``) with predefined genomic
        domain boundaries (e.g. TAD boundaries) to use as boundaries. If given,
        ``use_gene_boundaries`` will be ignored.
    use_gene_boundaries
        Whether to stop the search space when encountering the promoter of another
        gene (on either strand).
    upstream
        Search space upstream. The minimum (first position) means that even if there
        is a gene right next to it these bp will be taken. The second position
        indicates the maximum distance.
    downstream
        Search space downstream. The minimum (first position) means that even if
        there is a gene right next to it these bp will be taken. The second position
        indicates the maximum distance.
    distance_weight
        Whether to add a distance weight (an exponential function, the weight will
        decrease with distance).
    decay_rate
        Exponent for the distance exponential function (the higher the faster will
        be the decrease).
    extend_gene_body_upstream
        Number of bp upstream immune to the distance weight (their value will be
        maximum for this weight).
    extend_gene_body_downstream
        Number of bp downstream immune to the distance weight (their value will be
        maximum for this weight).
    gene_size_weight
        Whether to add a weight based on the length of the gene.
    gene_size_scale_factor
        Dividend to calculate the gene size weight. ``"median"`` uses the median
        gene length of all rows in ``gene_annotation``.
    remove_promoters
        Whether to ignore regions overlapping promoters of other genes.
    extend_tss
        Space (upstream, downstream) around the TSS considered as promoter.
    log
        Optional logger.

    Returns
    -------
    Polars DataFrame with one row per region-gene link: ``Name``, ``Gene``,
    ``Distance`` (positive: upstream of the gene; 0: in the gene body; negative:
    downstream of the gene), ``Distance_upstream`` and ``Distance_downstream`` (the
    search space of the gene; only when gene or predefined boundaries are used),
    ``Gene_size_weight`` and ``Distance_weight``.

    """
    use_boundaries = use_gene_boundaries or predefined_boundaries is not None

    annot = gene_annotation.select(
        pl.col("Chromosome"),
        pl.col("Gene_start").cast(pl.Int64).alias("Start"),
        pl.col("Gene_end").cast(pl.Int64).alias("End"),
        pl.col("Strand"),
        pl.col("Gene").fill_null("na"),
        pl.col("TSS_start").cast(pl.Int64),
    ).with_columns((pl.col("End") - pl.col("Start")).abs().alias("Gene_width"))

    if gene_size_weight:
        if log is not None:
            log.info("Calculating gene size weights.")
        if gene_size_scale_factor == "median":
            gene_size_scale_factor = annot.get_column("Gene_width").median()
        annot = annot.with_columns(
            (gene_size_scale_factor / pl.col("Gene_width")).alias("Gene_size_weight")
        )
    else:
        annot = annot.with_columns(pl.lit(1.0).alias("Gene_size_weight"))

    # Promoters: TSS extended by extend_tss.
    promoters = extend(
        annot.select(
            pl.col("Chromosome"),
            pl.col("TSS_start").alias("Start"),
            pl.col("TSS_start").alias("End"),
            pl.col("Strand"),
            pl.col("Gene"),
        ),
        upstream=extend_tss[0],
        downstream=extend_tss[1],
    )

    # One entry per gene.
    annot_nodup = (
        annot.drop("TSS_start")
        .unique(
            subset=["Chromosome", "Start", "End", "Strand"],
            keep="first",
            maintain_order=True,
        )
        .unique(subset=["Gene"], keep="first", maintain_order=True)
    )

    if use_boundaries:
        if predefined_boundaries is not None:
            if log is not None:
                log.info("Using predefined domains.")
            space = predefined_boundaries
        else:
            if log is not None:
                log.info("Calculating gene boundaries.")
            space = promoters

        chrom_sizes = chrom_sizes.select(
            pl.col("Chromosome"),
            pl.col("Start").cast(pl.Int64),
            pl.col("End").cast(pl.Int64),
        )
        boundaries = pl.concat(
            [
                space.select(
                    pl.col("Chromosome"),
                    pl.col("Start").cast(pl.Int64),
                    pl.col("End").cast(pl.Int64),
                ),
                # Chromosome limits.
                chrom_sizes.with_columns(pl.lit(1, dtype=pl.Int64).alias("End")),
                chrom_sizes.with_columns((pl.col("End") - 1).alias("Start")),
            ]
        )

        # Distance to nearest boundary (non-overlapping) upstream and downstream of
        # each gene, limited to the requested minimum and maximum search space.
        annot_nodup = (
            annot_nodup.with_columns(
                nearest(annot_nodup, boundaries, how="upstream")
                .get_column("Distance")
                .alias("Distance_upstream"),
                nearest(annot_nodup, boundaries, how="downstream")
                .get_column("Distance")
                .alias("Distance_downstream"),
            )
            .drop_nulls(["Distance_upstream", "Distance_downstream"])
            .with_columns(
                pl.col("Distance_upstream").clip(upstream[0], upstream[1]),
                pl.col("Distance_downstream").clip(downstream[0], downstream[1]),
            )
        )
        search_space = extend(
            annot_nodup.with_columns(
                pl.col("Start").alias("Gene_start"), pl.col("End").alias("Gene_end")
            ),
            upstream="Distance_upstream",
            downstream="Distance_downstream",
        )
    else:
        search_space = extend(
            annot_nodup.with_columns(
                pl.col("Start").alias("Gene_start"), pl.col("End").alias("Gene_end")
            ),
            upstream=upstream[1],
            downstream=downstream[1],
        )

    search_space = search_space.unique(
        subset=["Chromosome", "Start", "End", "Strand"],
        keep="first",
        maintain_order=True,
    )

    # Link regions to the genes in whose search space they are located.
    regions_per_gene = (
        intersection(
            regions.select(["Chromosome", "Start", "End", "Name"]),
            search_space,
            regions1_info=True,
            regions2_info=True,
            regions1_coord=True,
            regions1_suffix="@1",
            regions2_suffix="@2",
        )
        .select(
            pl.col("Chromosome"),
            pl.col("Start@1").alias("Start"),
            pl.col("End@1").alias("End"),
            pl.col("Name"),
            pl.all().exclude(
                ["Chromosome", "Start", "End", "Chromosome@1", "Start@1", "End@1", "Name"]
            ).name.map(lambda c: c.removesuffix("@2")),
        )
    )

    # Represent each region by its midpoint (rounded half to even).
    midpoint = np.round(
        regions_per_gene.get_column("Start").to_numpy()
        + (
            regions_per_gene.get_column("End").to_numpy()
            - regions_per_gene.get_column("Start").to_numpy()
        )
        / 2
    ).astype(np.int64)

    if log is not None:
        log.info("Calculating distances.")

    # Distance of the region midpoint to the gene body:
    #   positive: upstream of the gene, 0: in the gene, negative: downstream.
    regions_per_gene = (
        regions_per_gene.with_columns(
            pl.Series("Start", midpoint),
            pl.Series("End", midpoint + 1),
        )
        .with_columns(
            (pl.col("Gene_start") - pl.col("Start")).alias("_start_dist"),
            (pl.col("Gene_end") - pl.col("Start")).alias("_end_dist"),
        )
        .with_columns(
            (
                pl.when((pl.col("_start_dist") > 0) & (pl.col("_end_dist") > 0))
                .then(1)
                .when((pl.col("_start_dist") < 0) & (pl.col("_end_dist") < 0))
                .then(-1)
                .otherwise(0)
                * pl.when(pl.col("Strand") == "-").then(-1).otherwise(1)
                * pl.min_horizontal(
                    pl.col("_start_dist").abs(), pl.col("_end_dist").abs()
                )
            ).alias("Distance")
        )
        .drop(["_start_dist", "_end_dist"])
    )

    if distance_weight:
        if log is not None:
            log.info("Calculating distance weights.")
        is_gene_body = (pl.col("Distance") <= extend_gene_body_upstream) & (
            pl.col("Distance") >= extend_gene_body_downstream
        )
        is_upstream = pl.col("Distance") > extend_gene_body_upstream
        is_downstream = pl.col("Distance") < extend_gene_body_downstream
        exp_decay = (
            (-decay_rate * pl.col("Distance").abs() / 5000).exp() + np.exp(-1)
        )
        regions_per_gene = regions_per_gene.filter(
            is_gene_body | is_upstream | is_downstream
        ).with_columns(
            pl.when(is_gene_body)
            .then(1 + np.exp(-1))
            .otherwise(exp_decay)
            .alias("Distance_weight")
        )
    else:
        regions_per_gene = regions_per_gene.with_columns(
            pl.lit(1.0).alias("Distance_weight")
        )

    if remove_promoters:
        if log is not None:
            log.info("Removing distal regions overlapping promoters.")
        regions_per_gene = pl.concat(
            [
                regions_per_gene.filter(pl.col("Distance") == 0),
                overlap(
                    regions_per_gene.filter(pl.col("Distance") != 0),
                    promoters.select(["Chromosome", "Start", "End"]),
                    invert=True,
                ),
            ]
        )

    return regions_per_gene.select(
        pl.col("Name"),
        pl.col("Gene"),
        pl.col("Distance"),
        *(
            [pl.col("Distance_upstream"), pl.col("Distance_downstream")]
            if use_boundaries
            else []
        ),
        pl.col("Gene_size_weight"),
        pl.col("Distance_weight"),
    )


def get_gene_activity(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    region_names: Sequence[str],
    gene_annotation: pl.DataFrame,
    chrom_sizes: pl.DataFrame,
    predefined_boundaries: pl.DataFrame | None = None,
    use_gene_boundaries: bool = True,
    upstream: tuple[int, int] = (1000, 100000),
    downstream: tuple[int, int] = (1000, 100000),
    distance_weight: bool = True,
    decay_rate: float = 1.0,
    extend_gene_body_upstream: int = 5000,
    extend_gene_body_downstream: int = 0,
    gene_size_weight: bool = False,
    gene_size_scale_factor: Literal["median"] | float = "median",
    remove_promoters: bool = False,
    extend_tss: tuple[int, int] = (10, 10),
    gini_weight: bool = True,
    average_scores: bool = True,
    scale_factor: float = 1,
    chunk_size: int = 20000,
    log: logging.Logger | None = None,
) -> tuple[npt.NDArray[np.float32], list[str], pl.DataFrame]:
    """
    Infer gene activity from the topic model.

    Imputed accessibility (``region_topic @ cell_topic``) of the regions linked to a
    gene is aggregated using region weights. The imputed accessibility matrix is
    never fully materialized.

    Parameters
    ----------
    region_topic
        Region topic matrix (regions x topics).
    cell_topic
        Cell topic matrix (topics x cells).
    region_names
        Region names ("chrom:start-end") corresponding to the rows of
        ``region_topic``.
    gene_annotation
        Polars DataFrame with gene annotation as returned by
        :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl`.
    chrom_sizes
        Polars DataFrame with ``Chromosome``, ``Start`` (0) and ``End`` (chromosome
        size) columns.
    predefined_boundaries
        See :func:`get_region_gene_weights`.
    use_gene_boundaries
        See :func:`get_region_gene_weights`.
    upstream
        See :func:`get_region_gene_weights`.
    downstream
        See :func:`get_region_gene_weights`.
    distance_weight
        See :func:`get_region_gene_weights`.
    decay_rate
        See :func:`get_region_gene_weights`.
    extend_gene_body_upstream
        See :func:`get_region_gene_weights`.
    extend_gene_body_downstream
        See :func:`get_region_gene_weights`.
    gene_size_weight
        See :func:`get_region_gene_weights`.
    gene_size_scale_factor
        See :func:`get_region_gene_weights`.
    remove_promoters
        See :func:`get_region_gene_weights`.
    extend_tss
        See :func:`get_region_gene_weights`.
    gini_weight
        Whether to add a Gini index weight: the more cell specific the imputed
        accessibility of a region is, the higher this weight will be.
    average_scores
        Whether to divide the gene activity by the number of regions linked to the
        gene.
    scale_factor
        Value to multiply the gene activity with. If different from 1, the scaled
        gene activity is rounded and genes without any activity are removed.
    chunk_size
        Number of regions for which imputed accessibility is calculated at once when
        calculating Gini weights.
    log
        Optional logger.

    Returns
    -------
    Tuple with gene activity (cells x genes), gene names and the region weights
    (Polars DataFrame with one row per region-gene link, see
    :func:`get_region_gene_weights`, extended with ``Gini_weight`` and ``Weight``).

    """
    if region_topic.shape[0] != len(region_names):
        raise ValueError(
            f"Number of region names ({len(region_names)}) does not match the number "
            f"of rows of region_topic ({region_topic.shape[0]})."
        )
    if region_topic.shape[1] != cell_topic.shape[0]:
        raise ValueError(
            "region_topic (regions x topics) and cell_topic (topics x cells) "
            f"dimensions do not match: {region_topic.shape} and {cell_topic.shape}."
        )

    region_topic = region_topic.astype(np.float32, copy=False)
    cell_topic = cell_topic.astype(np.float32, copy=False)

    # Only keep regions with imputed accessibility in at least one cell.
    has_imputed_acc = (region_topic @ cell_topic.sum(axis=1)) != 0
    regions = _region_names_to_bed_df_pl(region_names).filter(
        pl.Series(has_imputed_acc)
    )

    region_weights = get_region_gene_weights(
        regions=regions,
        gene_annotation=gene_annotation,
        chrom_sizes=chrom_sizes,
        predefined_boundaries=predefined_boundaries,
        use_gene_boundaries=use_gene_boundaries,
        upstream=upstream,
        downstream=downstream,
        distance_weight=distance_weight,
        decay_rate=decay_rate,
        extend_gene_body_upstream=extend_gene_body_upstream,
        extend_gene_body_downstream=extend_gene_body_downstream,
        gene_size_weight=gene_size_weight,
        gene_size_scale_factor=gene_size_scale_factor,
        remove_promoters=remove_promoters,
        extend_tss=extend_tss,
        log=log,
    )

    region_name_to_idx = pl.DataFrame(
        {
            "Name": list(region_names),
            "Region_idx": np.arange(len(region_names), dtype=np.int64),
        }
    ).unique(subset=["Name"], keep="first", maintain_order=True)
    region_weights = region_weights.join(region_name_to_idx, on="Name", how="left")

    if gini_weight:
        if log is not None:
            log.info("Calculating Gini weights.")
        linked_region_idx = np.unique(
            region_weights.get_column("Region_idx").to_numpy()
        )
        gini = np.empty(linked_region_idx.shape[0], dtype=np.float64)
        for (chunk_start, chunk_end), imputed_acc_chunk in impute_accessibility_chunked(
            region_topic=region_topic[linked_region_idx],
            cell_topic=cell_topic,
            chunk_size=chunk_size,
            chunk_along="region",
            log=log,
            return_start_end=True,
        ):
            gini[chunk_start:chunk_end] = _gini_per_row(imputed_acc_chunk)
        region_weights = region_weights.join(
            pl.DataFrame(
                {
                    "Region_idx": linked_region_idx,
                    "Gini_weight": np.exp(1 - gini) + np.exp(-1),
                }
            ),
            on="Region_idx",
            how="left",
        )
    else:
        region_weights = region_weights.with_columns(pl.lit(1.0).alias("Gini_weight"))

    region_weights = region_weights.with_columns(
        (
            pl.col("Gene_size_weight") * pl.col("Distance_weight") * pl.col("Gini_weight")
        ).alias("Weight")
    ).filter(pl.col("Weight") > 0)

    if log is not None:
        log.info("Getting gene activity scores.")

    gene_names = sorted(region_weights.get_column("Gene").unique().to_list())
    gene_idx = (
        region_weights.get_column("Gene")
        .replace_strict(gene_names, list(range(len(gene_names))), return_dtype=pl.Int64)
        .to_numpy()
    )
    weights = region_weights.get_column("Weight").to_numpy()
    if average_scores:
        weights = weights / np.bincount(gene_idx, minlength=len(gene_names))[gene_idx]

    # Gene activity = W @ (region_topic @ cell_topic) = (W @ region_topic) @ cell_topic
    region_gene_weights = sparse.csr_matrix(
        (weights, (gene_idx, region_weights.get_column("Region_idx").to_numpy())),
        shape=(len(gene_names), region_topic.shape[0]),
    )
    gene_topic = region_gene_weights @ region_topic.astype(np.float64)
    gene_activity = (gene_topic @ cell_topic.astype(np.float64)).T.astype(np.float32)

    if scale_factor != 1:
        if log is not None:
            log.info("Scaling gene activity.")
        gene_activity = np.round(gene_activity * scale_factor)
        has_activity = np.any(gene_activity != 0, axis=0)
        gene_activity = gene_activity[:, has_activity]
        gene_names = [g for g, keep in zip(gene_names, has_activity) if keep]

    return gene_activity, gene_names, region_weights.drop("Region_idx")

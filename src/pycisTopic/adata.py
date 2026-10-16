"""
pycisTopic AnnData container.

A topic model is stored in a single AnnData object (cells x regions), in the same way
as scanpy stores a PCA:

- ``X``: binary accessibility matrix (cells x regions, sparse) or ``None``.
- ``obs``: cell metadata and QC statistics.
- ``var``: region coordinates (``Chromosome``, ``Start``, ``End``).
- ``obsm["X_topic"]``: cell-topic probabilities (cells x topics).
- ``varm["topic"]``: region-topic probabilities (regions x topics).
- ``uns["pycistopic"]``: ``version``, topic model parameters (``model``) and per topic
  statistics (``topics``, a DataFrame indexed by topic name).

Use ``sc.pp.neighbors(adata, use_rep="X_topic")`` to run scanpy on the cell-topic
probabilities.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    import anndata
    import numpy.typing as npt
    import scipy.sparse

OBSM_TOPIC_KEY = "X_topic"
VARM_TOPIC_KEY = "topic"
UNS_KEY = "pycistopic"


def _pycistopic_version() -> str:
    try:
        return version("pycisTopic")
    except PackageNotFoundError:
        return "unknown"


def _h5ad_compatible(value: Any) -> Any:
    """Make (nested) model parameters storable in h5ad (no None values)."""
    if isinstance(value, dict):
        return {
            str(k): _h5ad_compatible(v) for k, v in value.items() if v is not None
        }
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return value


def topic_names(n_topics: int) -> list[str]:
    """Topic names used by pycisTopic: ``Topic1``, ``Topic2``, ..."""
    return [f"Topic{i}" for i in range(1, n_topics + 1)]


def region_ids_to_var(region_ids: Sequence[str]) -> pd.DataFrame:
    """
    Create a ``var`` DataFrame with region coordinates from region IDs.

    Parameters
    ----------
    region_ids
        Region IDs in ``chrom:start-end`` format.

    Returns
    -------
    Pandas DataFrame indexed by region ID with ``Chromosome``, ``Start`` and ``End``.

    """
    coordinates = (
        pl.DataFrame({"region_id": list(region_ids)}, schema={"region_id": pl.Utf8})
        .with_columns(
            pl.col("region_id")
            .str.extract_groups(r"^(?<Chromosome>.+):(?<Start>[0-9]+)-(?<End>[0-9]+)$")
            .alias("coordinates")
        )
        .unnest("coordinates")
    )
    invalid = coordinates.filter(pl.col("Chromosome").is_null())["region_id"]
    if invalid.len() > 0:
        raise ValueError(
            f'Region IDs should be in "chrom:start-end" format, got: "{invalid[0]}".'
        )
    var = coordinates.select(
        pl.col("Chromosome"),
        pl.col("Start").cast(pl.Int64),
        pl.col("End").cast(pl.Int64),
    ).to_pandas()
    var.index = pd.Index(list(region_ids), name=None)
    return var


def create_anndata(
    cell_topic: npt.ArrayLike,
    region_topic: npt.ArrayLike,
    cell_barcodes: Sequence[str],
    region_ids: Sequence[str],
    binary_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    model_parameters: dict[str, Any] | None = None,
    topic_stats: pd.DataFrame | None = None,
    obs: pd.DataFrame | None = None,
) -> anndata.AnnData:
    """
    Create a pycisTopic AnnData object from a topic model.

    Parameters
    ----------
    cell_topic
        Cell-topic probabilities (cells x topics).
    region_topic
        Region-topic probabilities (regions x topics).
    cell_barcodes
        Cell barcodes (same order as the rows of ``cell_topic``).
    region_ids
        Region IDs in ``chrom:start-end`` format (same order as the rows of
        ``region_topic``).
    binary_matrix
        Optional binary accessibility matrix (regions x cells), as written by
        ``pycistopic count_matrix create``. Stored transposed in ``X``.
        If ``None``, ``X`` is not stored, which keeps the file small.
    model_parameters
        Topic model parameters, stored in ``uns["pycistopic"]["model"]``.
    topic_stats
        Per topic statistics (topics x statistics), stored in
        ``uns["pycistopic"]["topics"]``.
    obs
        Optional cell metadata, indexed by cell barcode.

    Returns
    -------
    AnnData object (cells x regions).

    """
    import anndata
    import scipy.sparse

    cell_topic = np.asarray(cell_topic, dtype=np.float32)
    region_topic = np.asarray(region_topic, dtype=np.float32)
    n_cells, n_topics = cell_topic.shape

    if len(cell_barcodes) != n_cells:
        raise ValueError(
            f"Number of cell barcodes ({len(cell_barcodes)}) does not match the number "
            f"of rows of cell_topic ({n_cells})."
        )
    if len(region_ids) != region_topic.shape[0]:
        raise ValueError(
            f"Number of region IDs ({len(region_ids)}) does not match the number of "
            f"rows of region_topic ({region_topic.shape[0]})."
        )
    if region_topic.shape[1] != n_topics:
        raise ValueError(
            f"cell_topic ({n_topics} topics) and region_topic "
            f"({region_topic.shape[1]} topics) have a different number of topics."
        )

    X = None
    if binary_matrix is not None:
        if binary_matrix.shape != (len(region_ids), n_cells):
            raise ValueError(
                f"binary_matrix should have shape (regions, cells) = "
                f"{(len(region_ids), n_cells)}, got {binary_matrix.shape}."
            )
        X = scipy.sparse.csr_matrix(binary_matrix.T, dtype=np.uint8)

    obs_df = pd.DataFrame(index=pd.Index(list(cell_barcodes)))
    if obs is not None:
        obs_df = obs_df.join(obs)

    topics = (
        topic_stats.copy()
        if topic_stats is not None
        else pd.DataFrame(index=topic_names(n_topics))
    )
    if len(topics) != n_topics:
        raise ValueError(
            f"topic_stats has {len(topics)} rows, but the model has {n_topics} topics."
        )
    topics.index = topic_names(n_topics)

    adata = anndata.AnnData(
        X=X,
        obs=obs_df,
        var=region_ids_to_var(region_ids),
        obsm={OBSM_TOPIC_KEY: cell_topic},
        varm={VARM_TOPIC_KEY: region_topic},
        uns={
            UNS_KEY: {
                "version": _pycistopic_version(),
                "model": _h5ad_compatible(model_parameters or {}),
                "topics": topics,
            }
        },
    )
    return adata


def check_anndata(adata: anndata.AnnData) -> None:
    """Check that an AnnData object contains a pycisTopic topic model."""
    if OBSM_TOPIC_KEY not in adata.obsm or VARM_TOPIC_KEY not in adata.varm:
        raise ValueError(
            f'AnnData object does not contain a topic model (.obsm["{OBSM_TOPIC_KEY}"] '
            f'and .varm["{VARM_TOPIC_KEY}"]). Create it with '
            "`pycistopic topic_modeling mallet create_anndata`."
        )


def get_topic_names(adata: anndata.AnnData) -> list[str]:
    """Get topic names of the topic model in an AnnData object."""
    check_anndata(adata)
    topics = adata.uns.get(UNS_KEY, {}).get("topics")
    if topics is not None:
        return list(topics.index)
    return topic_names(adata.obsm[OBSM_TOPIC_KEY].shape[1])


def get_topics(adata: anndata.AnnData) -> pd.DataFrame:
    """Get per topic statistics (``uns["pycistopic"]["topics"]``)."""
    check_anndata(adata)
    uns = adata.uns.setdefault(UNS_KEY, {})
    if "topics" not in uns:
        uns["topics"] = pd.DataFrame(index=get_topic_names(adata))
    return uns["topics"]


def get_cell_topic(adata: anndata.AnnData) -> npt.NDArray[np.float32]:
    """Get cell-topic probabilities as topics x cells (as used by pycisTopic)."""
    check_anndata(adata)
    return np.asarray(adata.obsm[OBSM_TOPIC_KEY], dtype=np.float32).T


def get_region_topic(adata: anndata.AnnData) -> npt.NDArray[np.float32]:
    """Get region-topic probabilities as regions x topics."""
    check_anndata(adata)
    return np.asarray(adata.varm[VARM_TOPIC_KEY], dtype=np.float32)


def get_regions(adata: anndata.AnnData) -> pl.DataFrame:
    """Get region coordinates and names (region IDs) as a Polars DataFrame."""
    return pl.DataFrame(
        {
            "Chromosome": adata.var["Chromosome"].astype(str).to_numpy(),
            "Start": adata.var["Start"].to_numpy().astype(np.int64),
            "End": adata.var["End"].to_numpy().astype(np.int64),
            "Name": adata.var_names.to_numpy().astype(str),
        }
    )


def add_qc_stats(
    adata: anndata.AnnData,
    fragments_stats_per_cb: pl.DataFrame | pd.DataFrame,
    cb_column: str = "CB",
) -> anndata.AnnData:
    """
    Add QC statistics per cell barcode (as written by ``pycistopic qc run``) to ``obs``.

    Parameters
    ----------
    adata
        pycisTopic AnnData object.
    fragments_stats_per_cb
        QC statistics per cell barcode, e.g. read from
        ``{sample_id}.fragments_stats_per_cb.parquet`` files. Cell barcodes should be
        in the same format as ``adata.obs_names`` (``CB___sample_id``).
    cb_column
        Column with cell barcodes.

    Returns
    -------
    The AnnData object, with QC statistics added to ``obs``. Columns that already
    exist in ``obs`` are overwritten.

    """
    if isinstance(fragments_stats_per_cb, pl.DataFrame):
        stats = fragments_stats_per_cb.with_columns(
            pl.col(cb_column).cast(pl.Utf8)
        ).to_pandas()
    else:
        stats = fragments_stats_per_cb.copy()
    stats = stats.set_index(cb_column)
    if stats.index.has_duplicates:
        raise ValueError("Cell barcodes in the QC statistics are not unique.")
    adata.obs = adata.obs.drop(
        columns=[c for c in stats.columns if c in adata.obs.columns]
    ).join(stats)
    return adata


def read_qc_stats(
    pycistopic_qc_output_dir: str,
    sample_ids: Sequence[str],
    cb_end_to_remove: str | None = None,
    cb_sample_separator: str = "___",
) -> pl.DataFrame:
    """
    Read QC statistics per cell barcode of ``pycistopic qc run`` for multiple samples.

    Cell barcodes are converted to pycisTopic cell barcodes (``CB___sample_id``) in
    the same way as by ``pycistopic count_matrix create``.

    Parameters
    ----------
    pycistopic_qc_output_dir
        Directory with ``{sample_id}.fragments_stats_per_cb.parquet`` files.
    sample_ids
        Sample IDs. Samples without QC statistics file are skipped with a warning.
    cb_end_to_remove
        Suffix to remove from the cell barcodes (e.g. ``"-1"``), as used when creating
        the count matrix.
    cb_sample_separator
        Separator between cell barcode and sample ID.

    Returns
    -------
    Polars DataFrame with QC statistics per cell barcode (``CB`` column) and a
    ``sample_id`` column.

    """
    import os
    import warnings

    stats_per_sample = []
    for sample_id in sample_ids:
        filename = os.path.join(
            pycistopic_qc_output_dir, f"{sample_id}.fragments_stats_per_cb.parquet"
        )
        if not os.path.exists(filename):
            warnings.warn(f'QC statistics file "{filename}" not found.', stacklevel=2)
            continue
        cb = pl.col("CB").cast(pl.Utf8)
        if cb_end_to_remove:
            cb = cb.str.strip_suffix(cb_end_to_remove)
        stats_per_sample.append(
            pl.read_parquet(filename).with_columns(
                (cb + pl.lit(f"{cb_sample_separator}{sample_id}")).alias("CB"),
                pl.lit(sample_id).alias("sample_id"),
            )
        )
    if not stats_per_sample:
        raise FileNotFoundError(
            f'No QC statistics files found in "{pycistopic_qc_output_dir}".'
        )
    return pl.concat(stats_per_sample, how="diagonal_relaxed")

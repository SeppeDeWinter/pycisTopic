from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pycisTopic.adata import create_anndata, topic_names
from pycisTopic.topic_modeling.mallet_models import LDAMallet, LDAMalletFilenames

if TYPE_CHECKING:
    import anndata
    import scipy.sparse


def _read_model_stats(
    lda_mallet_filenames: LDAMalletFilenames,
) -> tuple[pd.DataFrame | None, dict]:
    """Read per topic and per model statistics written by ``mallet stats``."""
    if not os.path.exists(lda_mallet_filenames.model_stats_filename):
        return None, {}
    with open(lda_mallet_filenames.model_stats_filename) as fh:
        stats = json.load(fh)
    n_topics = lda_mallet_filenames.n_topics
    topic_stats = pd.DataFrame(
        {
            "Assignments": np.asarray(stats["assignments"], dtype=np.float64),
            "Coherence": np.asarray(stats["coherence"], dtype=np.float64),
            "Marginal_topic_dist": np.asarray(stats["marg_topic"], dtype=np.float64),
        },
        index=topic_names(n_topics),
    )
    model_metrics = {
        key: float(stats[key])
        for key in ("Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood")
        if key in stats
    }
    return topic_stats, model_metrics


def create_anndata_from_mallet(
    output_prefix: str,
    n_topics: int,
    cell_barcodes: list[str],
    region_ids: list[str],
    binary_matrix: scipy.sparse.spmatrix | None = None,
    write: bool = True,
) -> anndata.AnnData:
    """
    Create a pycisTopic AnnData object from Mallet topic modeling results.

    See :mod:`pycisTopic.adata` for the layout of the AnnData object.

    Parameters
    ----------
    output_prefix
        Output prefix used for running topic modeling with Mallet.
    n_topics
        Number of topics used in the topic model created by Mallet.
        In combination with output_prefix, this allows to load the correct region
        topic counts and cell topic probabilities parquet files.
    cell_barcodes
        List containing cell names as ordered in the binary matrix columns.
    region_ids
        List containing region names as ordered in the binary matrix rows.
    binary_matrix
        Optional binary accessibility matrix (regions x cells) to store in ``X``.
    write
        Write the AnnData object to ``{output_prefix}.{n_topics}_topics.h5ad``.

    Returns
    -------
    AnnData object (cells x regions).

    """
    lda_mallet_filenames = LDAMalletFilenames(
        output_prefix=output_prefix, n_topics=n_topics
    )
    print("Reading Mallet results ...")
    topic_region = LDAMallet.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
        mallet_region_topic_counts_parquet_filename=lda_mallet_filenames.region_topic_counts_parquet_filename
    )
    cell_topic = LDAMallet.read_cell_topic_probabilities_parquet_file(
        mallet_cell_topic_probabilities_parquet_filename=lda_mallet_filenames.cell_topic_probabilities_parquet_filename
    )

    model_parameters: dict = {"n_topics": n_topics}
    if os.path.exists(lda_mallet_filenames.parameters_json_filename):
        model_parameters.update(
            LDAMallet.read_parameters_json_filename(
                lda_mallet_filenames.parameters_json_filename
            )
        )
    topic_stats, model_metrics = _read_model_stats(lda_mallet_filenames)
    if model_metrics:
        model_parameters["metrics"] = model_metrics

    adata = create_anndata(
        cell_topic=cell_topic,
        region_topic=topic_region.T,
        cell_barcodes=cell_barcodes,
        region_ids=region_ids,
        binary_matrix=binary_matrix,
        model_parameters=model_parameters,
        topic_stats=topic_stats,
    )
    print(f"Created AnnData object with shape: {adata.shape}")

    if write:
        print(f"Writing to: {lda_mallet_filenames.anndata_filename}")
        adata.write_h5ad(lda_mallet_filenames.anndata_filename)

    return adata

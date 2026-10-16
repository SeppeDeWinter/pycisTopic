"""
Convert CistopicObject pickles from pycisTopic < 3 to the pycisTopic AnnData format.

Old pickles are read without the old pycisTopic classes: every pickled
``pycisTopic.*`` object is restored as a plain object with the same attributes.
"""

from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING, Any, BinaryIO, ClassVar

import numpy as np
import pandas as pd

from pycisTopic.adata import create_anndata, topic_names

if TYPE_CHECKING:
    from pathlib import Path

    import anndata


class LegacyObject:
    """Attribute container for pickled objects of pycisTopic < 3."""

    legacy_class = "unknown"

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"<legacy {self.legacy_class} with {sorted(self.__dict__)}>"


try:
    # Unpickler used by `pandas.read_pickle` to read pickles of older pandas versions.
    from pandas.compat.pickle_compat import Unpickler as _BaseUnpickler
except ImportError:  # pragma: no cover
    _BaseUnpickler = pickle.Unpickler


class _LegacyUnpickler(_BaseUnpickler):
    _classes: ClassVar[dict[str, type[LegacyObject]]] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module == "pycisTopic" or module.startswith("pycisTopic."):
            full_name = f"{module}.{name}"
            if full_name not in self._classes:
                self._classes[full_name] = type(
                    name, (LegacyObject,), {"legacy_class": full_name}
                )
            return self._classes[full_name]
        return super().find_class(module, name)


def read_legacy_pickle(pickle_file: str | Path | BinaryIO) -> Any:
    """
    Read a pickle created with pycisTopic < 3 (e.g. a CistopicObject).

    Only read pickles from trusted sources: unpickling can execute arbitrary code.

    Parameters
    ----------
    pickle_file
        Filename or binary file object.

    Returns
    -------
    Unpickled object, with pycisTopic classes replaced by :class:`LegacyObject`.

    """
    if hasattr(pickle_file, "read"):
        return _LegacyUnpickler(pickle_file).load()
    with open(pickle_file, "rb") as fh:
        return _LegacyUnpickler(fh).load()


def _obs_or_var_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    """Make object columns (possibly with mixed types and NaNs) storable in h5ad."""
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == object:
            values = df[column]
            df[column] = values.where(values.isna(), values.astype(str)).astype(
                "category"
            )
    df.columns = [str(column) for column in df.columns]
    return df


def _topic_stats(model: Any, n_topics: int) -> pd.DataFrame:
    """Per topic statistics of a legacy CistopicLDAModel."""
    names = topic_names(n_topics)
    topic_qc_metrics = getattr(model, "topic_qc_metrics", None)
    if isinstance(topic_qc_metrics, pd.DataFrame) and len(topic_qc_metrics) == n_topics:
        stats = topic_qc_metrics.copy()
        stats.index = names
        return stats

    stats = pd.DataFrame(index=names)
    topic_ass = getattr(model, "topic_ass", None)
    if isinstance(topic_ass, pd.DataFrame) and "Assignments" in topic_ass:
        stats["Assignments"] = topic_ass["Assignments"].to_numpy(dtype=np.float64)
    coherence = getattr(model, "coherence", None)
    if isinstance(coherence, pd.DataFrame) and "Mimno_2011" in coherence:
        stats["Coherence"] = coherence["Mimno_2011"].to_numpy(dtype=np.float64)
    marg_topic = getattr(model, "marg_topic", None)
    if isinstance(marg_topic, pd.DataFrame) and "Marg_Topic" in marg_topic:
        stats["Marginal_topic_dist"] = marg_topic["Marg_Topic"].to_numpy(
            dtype=np.float64
        )
    return stats


def _model_parameters(model: Any) -> dict[str, Any]:
    """Model parameters and metrics of a legacy CistopicLDAModel."""
    parameters: dict[str, Any] = {"converted_from": "pycisTopic < 3"}
    legacy_parameters = getattr(model, "parameters", None)
    if isinstance(legacy_parameters, pd.DataFrame) and legacy_parameters.shape[1] > 0:
        for key, value in legacy_parameters.iloc[:, 0].items():
            parameters[str(key)] = str(value)
    metrics = getattr(model, "metrics", None)
    if isinstance(metrics, pd.DataFrame) and len(metrics) > 0:
        parameters["metrics"] = {
            str(key): float(value)
            for key, value in metrics.iloc[0].items()
            if np.isscalar(value) and pd.notna(value)
        }
    return parameters


def cistopic_object_to_anndata(
    cistopic_obj: Any,
    store_matrix: bool = False,
) -> anndata.AnnData:
    """
    Convert a CistopicObject of pycisTopic < 3 to a pycisTopic AnnData object.

    Parameters
    ----------
    cistopic_obj
        CistopicObject read with :func:`read_legacy_pickle`.
    store_matrix
        Store the binary accessibility matrix in ``X``.

    Returns
    -------
    pycisTopic AnnData object (see :mod:`pycisTopic.adata`) with cell data in ``obs``,
    region data in ``var``, the selected topic model in ``obsm["X_topic"]`` and
    ``varm["topic"]`` and cell projections (e.g. UMAP) in ``obsm["X_<name>"]``.

    """
    model = getattr(cistopic_obj, "selected_model", None)
    if model is None or isinstance(model, list):
        raise ValueError(
            "CistopicObject does not contain a selected topic model "
            "(cistopic_obj.selected_model)."
        )

    cell_names = [str(cell) for cell in cistopic_obj.cell_names]
    region_names = [str(region) for region in cistopic_obj.region_names]

    # Old models: cell_topic is topics x cells, topic_region is regions x topics.
    cell_topic = model.cell_topic.loc[:, cistopic_obj.cell_names].to_numpy().T
    region_topic = model.topic_region.loc[cistopic_obj.region_names].to_numpy()
    n_topics = cell_topic.shape[1]

    obs = None
    cell_data = getattr(cistopic_obj, "cell_data", None)
    if isinstance(cell_data, pd.DataFrame):
        obs = _obs_or_var_for_h5ad(
            cell_data[~cell_data.index.duplicated(keep="first")]
        )
        obs.index = obs.index.astype(str)

    adata = create_anndata(
        cell_topic=cell_topic,
        region_topic=region_topic,
        cell_barcodes=cell_names,
        region_ids=region_names,
        binary_matrix=cistopic_obj.binary_matrix if store_matrix else None,
        model_parameters=_model_parameters(model),
        topic_stats=_topic_stats(model, n_topics),
        obs=obs,
    )

    region_data = getattr(cistopic_obj, "region_data", None)
    if isinstance(region_data, pd.DataFrame):
        extra_columns = [
            c for c in region_data.columns if c not in ("Chromosome", "Start", "End")
        ]
        region_data = region_data[extra_columns]
        n_duplicated = int(region_data.index.duplicated().sum())
        if n_duplicated > 0:
            warnings.warn(
                f"CistopicObject.region_data contains {n_duplicated} duplicated region "
                "names; only the first entry of each region is kept.",
                stacklevel=2,
            )
            region_data = region_data[~region_data.index.duplicated(keep="first")]
        region_data = _obs_or_var_for_h5ad(region_data)
        region_data.index = region_data.index.astype(str)
        adata.var = adata.var.join(region_data)

    projections = getattr(cistopic_obj, "projections", {}) or {}
    for name, projection in (projections.get("cell") or {}).items():
        if isinstance(projection, pd.DataFrame):
            adata.obsm[f"X_{str(name).lower()}"] = (
                projection.reindex(cistopic_obj.cell_names).to_numpy(dtype=np.float32)
            )

    adata.uns["pycistopic"]["project"] = str(getattr(cistopic_obj, "project", ""))
    return adata

from __future__ import annotations

import pickle
import subprocess
import sys
import types

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from pycisTopic.legacy import (
    LegacyObject,
    cistopic_object_to_anndata,
    read_legacy_pickle,
)

CELLS = [f"AAA{i}-1-sample" for i in range(6)]
REGIONS = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(5)]
TOPICS = ["Topic1", "Topic2"]


def _legacy_classes(monkeypatch):
    """Register fake pycisTopic < 3 modules so pickles refer to the old class paths."""
    cistopic_class = types.ModuleType("pycisTopic.cistopic_class")
    lda_models = types.ModuleType("pycisTopic.lda_models")

    class CistopicObject:
        pass

    class CistopicLDAModel:
        pass

    for module, cls in ((cistopic_class, CistopicObject), (lda_models, CistopicLDAModel)):
        cls.__module__ = module.__name__
        cls.__qualname__ = cls.__name__
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return CistopicObject, CistopicLDAModel


@pytest.fixture
def legacy_pickle(tmp_path, monkeypatch, rng):
    CistopicObject, CistopicLDAModel = _legacy_classes(monkeypatch)

    model = CistopicLDAModel()
    # Cells in a different order than cistopic_obj.cell_names.
    model.cell_topic = pd.DataFrame(
        rng.dirichlet(np.ones(2), size=len(CELLS)).T,
        index=TOPICS,
        columns=CELLS[::-1],
    )
    model.topic_region = pd.DataFrame(
        rng.random((len(REGIONS), 2)), index=REGIONS, columns=TOPICS
    )
    model.topic_ass = pd.DataFrame({"Topic": [1, 2], "Assignments": [100, 1000]})
    model.coherence = pd.DataFrame({"Mimno_2011": [-1.0, -2.0]})
    model.marg_topic = pd.DataFrame({"Marg_Topic": [0.4, 0.6]})
    model.parameters = pd.DataFrame(
        {"Parameter": ["Mallet", 2, 50]}, index=["package", "n_topics", "alpha"]
    )
    model.metrics = pd.DataFrame(
        {"Arun_2010": [1.5], "loglikelihood": [-100.0]}, index=["Metric"]
    )

    obj = CistopicObject()
    obj.cell_names = CELLS
    obj.region_names = REGIONS
    obj.binary_matrix = scipy.sparse.csr_matrix(
        (rng.random((len(REGIONS), len(CELLS))) < 0.5).astype(np.int32)
    )
    obj.fragment_matrix = obj.binary_matrix * 2
    obj.cell_data = pd.DataFrame(
        {
            "celltype": ["a", "b", np.nan, "a", "b", 3],  # mixed types and NaN
            "tss": np.arange(len(CELLS), dtype=float),
        },
        index=CELLS,
    )
    obj.region_data = pd.DataFrame(
        {
            "Chromosome": ["chr1"] * 6,
            "Start": [0, 1000, 2000, 3000, 4000, 0],
            "End": [500, 1500, 2500, 3500, 4500, 500],
            "cisTopic_nr_acc": [1, 2, 3, 4, 5, 99],
        },
        index=[*REGIONS, REGIONS[0]],  # duplicated region
    )
    obj.projections = {
        "cell": {"UMAP": pd.DataFrame(rng.random((len(CELLS), 2)), index=CELLS[::-1])},
        "region": {},
    }
    obj.project = "test_project"
    obj.selected_model = model
    obj.path_to_fragments = {"sample": "fragments.tsv.gz"}

    path = tmp_path / "cistopic_obj.pkl"
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return path, obj


def test_read_legacy_pickle(legacy_pickle, monkeypatch):
    path, original = legacy_pickle
    # The old classes are not importable when reading.
    monkeypatch.delitem(sys.modules, "pycisTopic.cistopic_class")
    monkeypatch.delitem(sys.modules, "pycisTopic.lda_models")
    obj = read_legacy_pickle(path)
    assert isinstance(obj, LegacyObject)
    assert isinstance(obj.selected_model, LegacyObject)
    assert obj.legacy_class == "pycisTopic.cistopic_class.CistopicObject"
    pd.testing.assert_frame_equal(obj.cell_data, original.cell_data)
    with open(path, "rb") as fh:
        assert read_legacy_pickle(fh).cell_names == CELLS


class TestCistopicObjectToAnnData:
    @pytest.fixture
    def obj(self, legacy_pickle):
        return read_legacy_pickle(legacy_pickle[0])

    def test_conversion(self, obj, tmp_path):
        with pytest.warns(UserWarning, match="1 duplicated region names"):
            adata = cistopic_object_to_anndata(obj, store_matrix=True)
        assert list(adata.obs_names) == CELLS
        assert list(adata.var_names) == REGIONS
        np.testing.assert_allclose(
            adata.obsm["X_topic"], obj.selected_model.cell_topic[CELLS].to_numpy().T
        )
        np.testing.assert_allclose(
            adata.varm["topic"], obj.selected_model.topic_region.to_numpy()
        )
        np.testing.assert_array_equal(adata.X.toarray(), obj.binary_matrix.T.toarray())
        np.testing.assert_allclose(
            adata.obsm["X_umap"],
            obj.projections["cell"]["UMAP"].loc[CELLS].to_numpy(),
            rtol=1e-6,
        )
        assert adata.var["cisTopic_nr_acc"].tolist() == [1, 2, 3, 4, 5]
        assert adata.obs["tss"].tolist() == list(range(6))
        uns = adata.uns["pycistopic"]
        assert uns["project"] == "test_project"
        assert uns["model"]["alpha"] == "50"
        assert uns["model"]["metrics"] == {"Arun_2010": 1.5, "loglikelihood": -100.0}
        assert uns["topics"]["Assignments"].tolist() == [100, 1000]
        assert uns["topics"]["Coherence"].tolist() == [-1.0, -2.0]
        assert uns["topics"]["Marginal_topic_dist"].tolist() == [0.4, 0.6]

        # Mixed type columns with NaNs can be written to h5ad.
        adata.write_h5ad(tmp_path / "converted.h5ad")
        result = anndata.read_h5ad(tmp_path / "converted.h5ad")
        celltype = result.obs["celltype"]
        assert celltype.isna().tolist() == [False, False, True, False, False, False]
        assert celltype.dropna().tolist() == ["a", "b", "a", "b", "3"]

    def test_without_matrix(self, obj):
        with pytest.warns(UserWarning, match="duplicated region names"):
            assert cistopic_object_to_anndata(obj).X is None

    def test_without_selected_model(self, obj):
        obj.selected_model = []
        with pytest.raises(ValueError, match="selected topic model"):
            cistopic_object_to_anndata(obj)


def test_cli_convert(legacy_pickle, tmp_path):
    output = tmp_path / "converted.h5ad"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pycisTopic.cli.pycistopic",
            "convert",
            "-i",
            str(legacy_pickle[0]),
            "-o",
            str(output),
            "--store_matrix",
        ],
        check=True,
        capture_output=True,
    )
    adata = anndata.read_h5ad(output)
    assert adata.shape == (len(CELLS), len(REGIONS))
    assert adata.X is not None

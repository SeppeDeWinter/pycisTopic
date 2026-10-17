from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse

from pycisTopic.adata import (
    add_qc_stats,
    check_anndata,
    create_anndata,
    get_cell_topic,
    get_region_topic,
    get_regions,
    get_topic_names,
    get_topics,
    region_ids_to_var,
)

N_CELLS, N_REGIONS, N_TOPICS = 6, 5, 3
CELLS = [f"cell{i}___s1" for i in range(N_CELLS)]
REGIONS = [f"chr{i % 2 + 1}:{i * 100}-{i * 100 + 50}" for i in range(N_REGIONS)]


@pytest.fixture
def model(rng):
    cell_topic = rng.dirichlet(np.ones(N_TOPICS), size=N_CELLS)
    region_topic = rng.random((N_REGIONS, N_TOPICS))
    binary_matrix = scipy.sparse.csr_matrix(
        (rng.random((N_REGIONS, N_CELLS)) < 0.5).astype(np.int32)
    )
    return cell_topic, region_topic, binary_matrix


def test_region_ids_to_var():
    var = region_ids_to_var(["chr1:10-20", "HLA-A:1-5", "chrUn_KI270302v1:3-4"])
    assert var.to_dict("list") == {
        "Chromosome": ["chr1", "HLA-A", "chrUn_KI270302v1"],
        "Start": [10, 1, 3],
        "End": [20, 5, 4],
    }
    with pytest.raises(ValueError, match="chrom:start-end"):
        region_ids_to_var(["chr1_10_20"])


class TestCreateAnnData:
    def test_layout(self, model):
        cell_topic, region_topic, binary_matrix = model
        adata = create_anndata(
            cell_topic,
            region_topic,
            CELLS,
            REGIONS,
            binary_matrix,
            model_parameters={"alpha": 50, "skipped": None, "cmd": ["a", 1]},
        )
        assert adata.shape == (N_CELLS, N_REGIONS)
        assert list(adata.obs_names) == CELLS
        assert list(adata.var_names) == REGIONS
        np.testing.assert_allclose(get_cell_topic(adata), cell_topic.T, rtol=1e-6)
        np.testing.assert_allclose(get_region_topic(adata), region_topic, rtol=1e-6)
        np.testing.assert_array_equal(adata.X.toarray(), binary_matrix.T.toarray())
        assert get_topic_names(adata) == ["Topic1", "Topic2", "Topic3"]
        assert adata.uns["pycistopic"]["model"] == {"alpha": 50, "cmd": ["a", "1"]}
        assert "version" in adata.uns["pycistopic"]

    def test_read_only_input(self, model):
        cell_topic, region_topic, _ = model
        cell_topic = cell_topic.astype(np.float32)
        region_topic = region_topic.astype(np.float32)
        cell_topic.flags.writeable = False
        region_topic.flags.writeable = False
        adata = create_anndata(cell_topic, region_topic, CELLS, REGIONS)
        assert adata.obsm["X_topic"].flags.writeable
        assert adata.varm["topic"].flags.writeable

    def test_h5ad_roundtrip(self, model, tmp_path):
        cell_topic, region_topic, binary_matrix = model
        topic_stats = pd.DataFrame({"Coherence": [-1.0, -2.0, -3.0]})
        for matrix in (binary_matrix, None):
            adata = create_anndata(
                cell_topic,
                region_topic,
                CELLS,
                REGIONS,
                matrix,
                model_parameters={"n_topics": 3, "metrics": {"Mimno_2011": -1.5}},
                topic_stats=topic_stats,
            )
            adata.write_h5ad(tmp_path / "adata.h5ad")
            result = anndata.read_h5ad(tmp_path / "adata.h5ad")
            assert (result.X is None) == (matrix is None)
            np.testing.assert_allclose(get_cell_topic(result), get_cell_topic(adata))
            assert result.uns["pycistopic"]["model"]["metrics"] == {"Mimno_2011": -1.5}
            pd.testing.assert_frame_equal(
                get_topics(result), get_topics(adata), check_index_type=False
            )

    def test_obs(self, model):
        cell_topic, region_topic, _ = model
        obs = pd.DataFrame({"celltype": ["a", "b"]}, index=CELLS[:2])
        adata = create_anndata(cell_topic, region_topic, CELLS, REGIONS, obs=obs)
        assert adata.obs["celltype"].tolist()[:2] == ["a", "b"]
        assert adata.obs["celltype"].isna().sum() == N_CELLS - 2

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"cell_barcodes": CELLS[:-1]}, "cell barcodes"),
            ({"region_ids": REGIONS[:-1]}, "region IDs"),
            ({"binary_matrix": scipy.sparse.csr_matrix((N_CELLS, N_REGIONS))}, "shape"),
            ({"topic_stats": pd.DataFrame({"x": [1, 2]})}, "topic_stats"),
        ],
    )
    def test_validation(self, model, kwargs, message):
        cell_topic, region_topic, _ = model
        arguments = {
            "cell_topic": cell_topic,
            "region_topic": region_topic,
            "cell_barcodes": CELLS,
            "region_ids": REGIONS,
            **kwargs,
        }
        with pytest.raises(ValueError, match=message):
            create_anndata(**arguments)

    def test_topic_mismatch(self, model):
        cell_topic, region_topic, _ = model
        with pytest.raises(ValueError, match="different number of topics"):
            create_anndata(cell_topic, region_topic[:, :2], CELLS, REGIONS)


def test_check_anndata():
    with pytest.raises(ValueError, match="does not contain a topic model"):
        check_anndata(anndata.AnnData(np.zeros((2, 2))))


def test_get_regions(model):
    adata = create_anndata(*model[:2], CELLS, REGIONS)
    regions = get_regions(adata)
    assert regions.columns == ["Chromosome", "Start", "End", "Name"]
    assert regions.row(1) == ("chr2", 100, 150, REGIONS[1])


def test_get_topics_creates_table(model):
    adata = create_anndata(*model[:2], CELLS, REGIONS)
    del adata.uns["pycistopic"]["topics"]
    get_topics(adata)["x"] = [1, 2, 3]
    assert adata.uns["pycistopic"]["topics"]["x"].tolist() == [1, 2, 3]


class TestAddQcStats:
    def test_polars(self, model):
        adata = create_anndata(*model[:2], CELLS, REGIONS)
        adata.obs["tss_enrichment"] = 0.0
        stats = pl.DataFrame(
            {
                "CB": pl.Series(CELLS[:3] + ["other"], dtype=pl.Categorical),
                "tss_enrichment": [5.0, 6.0, 7.0, 8.0],
                "unique_fragments_count": [1, 2, 3, 4],
            }
        )
        add_qc_stats(adata, stats)
        assert adata.obs["tss_enrichment"].tolist()[:3] == [5.0, 6.0, 7.0]
        assert adata.obs["tss_enrichment"].isna().sum() == N_CELLS - 3
        assert list(adata.obs_names) == CELLS

    def test_duplicates(self, model):
        adata = create_anndata(*model[:2], CELLS, REGIONS)
        stats = pd.DataFrame({"CB": [CELLS[0], CELLS[0]], "x": [1, 2]})
        with pytest.raises(ValueError, match="not unique"):
            add_qc_stats(adata, stats)


class TestReadQcStats:
    @pytest.fixture
    def qc_dir(self, tmp_path):
        for sample, cbs in (("s1", ["AAA-1", "BBB-1"]), ("s2", ["AAA-1"])):
            pl.DataFrame(
                {
                    "CB": pl.Series(cbs, dtype=pl.Categorical),
                    "tss_enrichment": [float(len(sample))] * len(cbs),
                }
            ).write_parquet(tmp_path / f"{sample}.fragments_stats_per_cb.parquet")
        return tmp_path

    def test_read_qc_stats(self, qc_dir):
        from pycisTopic.adata import read_qc_stats

        stats = read_qc_stats(str(qc_dir), ["s1", "s2"], cb_end_to_remove="-1")
        assert stats["CB"].to_list() == ["AAA___s1", "BBB___s1", "AAA___s2"]
        assert stats["sample_id"].to_list() == ["s1", "s1", "s2"]

    def test_missing_samples(self, qc_dir):
        from pycisTopic.adata import read_qc_stats

        with pytest.warns(UserWarning, match="s3.fragments_stats_per_cb.parquet"):
            stats = read_qc_stats(str(qc_dir), ["s1", "s3"])
        assert stats["CB"].to_list() == ["AAA-1___s1", "BBB-1___s1"]
        with pytest.warns(UserWarning, match="not found"), pytest.raises(FileNotFoundError):
            read_qc_stats(str(qc_dir), ["s3"])

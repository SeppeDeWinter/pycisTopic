from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pycisTopic.topic_qc import _proportion_ztest_larger, gini_coefficient


@pytest.mark.parametrize(
    ("count", "nobs", "value"), [(30, 100, 0.2), (5, 100, 0.2), (50, 60, 0.5)]
)
def test_proportion_ztest_larger(count, nobs, value):
    # statsmodels.stats.proportion.proportions_ztest(alternative="larger") uses the
    # observed proportion to estimate the variance.
    prop = count / nobs
    z = (prop - value) / np.sqrt(prop * (1 - prop) / nobs)
    assert _proportion_ztest_larger(count, nobs, value) == pytest.approx(norm.sf(z))


@pytest.mark.parametrize(("count", "expected"), [(0, 1.0), (10, 0.0)])
def test_proportion_ztest_larger_zero_variance(count, expected):
    assert _proportion_ztest_larger(count, 10, 0.5) == expected


def test_gini_coefficient():
    assert gini_coefficient(np.ones(10)) == pytest.approx(0.0)
    x = np.zeros(100)
    x[0] = 1
    assert gini_coefficient(x) == pytest.approx(0.99)


class TestTopicQcOnAnnData:
    @pytest.fixture
    def adata(self, rng):
        import pandas as pd

        from pycisTopic.adata import create_anndata

        n_cells = 300
        cells = [f"c{i}" for i in range(n_cells)]
        cell_topic = rng.dirichlet(np.full(3, 0.2), size=n_cells)
        # Topic1 is specific for group A, Topic2 for group B.
        cell_topic[:100, 0] += 2
        cell_topic[100:200, 1] += 2
        cell_topic /= cell_topic.sum(axis=1, keepdims=True)
        return create_anndata(
            cell_topic,
            rng.random((4, 3)),
            cells,
            [f"chr1:{i}0-{i}5" for i in range(1, 5)],
            topic_stats=pd.DataFrame(
                {
                    "Assignments": [10.0, 100.0, 1000.0],
                    "Coherence": [-1.0, -2.0, -3.0],
                    "Marginal_topic_dist": [0.2, 0.3, 0.5],
                }
            ),
            obs=pd.DataFrame({"group": np.repeat(["A", "B", "C"], 100)}, index=cells),
        )

    def test_compute_topic_metrics(self, adata):
        from pycisTopic.topic_qc import compute_topic_metrics

        metrics = compute_topic_metrics(adata)
        assert metrics["Log10_Assignments"].tolist() == pytest.approx([1, 2, 3])
        cell_topic = adata.obsm["X_topic"]
        assert metrics["Gini_index"].tolist() == pytest.approx(
            [gini_coefficient(cell_topic[:, i]) for i in range(3)]
        )
        assert "Gini_index" in adata.uns["pycistopic"]["topics"]

    def test_compute_topic_metrics_without_stats(self, adata):
        from pycisTopic.topic_qc import compute_topic_metrics

        del adata.uns["pycistopic"]["topics"]
        with pytest.warns(UserWarning, match="mallet stats"):
            metrics = compute_topic_metrics(adata)
        assert list(metrics.columns) == ["Gini_index"]

    def test_topic_annotation(self, adata):
        from pycisTopic.topic_qc import topic_annotation

        annotation = topic_annotation(adata, "group")
        assert annotation.loc["Topic1", "group"] == "A"
        assert annotation.loc["Topic2", "group"] == "B"
        assert annotation["Ratio_group_in_population"].tolist()[:2] == pytest.approx(
            [1 / 3, 1 / 3]
        )
        assert not annotation["is_general"].any()
        assert "is_general" in adata.uns["pycistopic"]["topics"]

    def test_topic_annotation_general_topic(self, adata):
        from pycisTopic.topic_qc import topic_annotation

        cells = adata.obs_names.tolist()
        binarized = {
            "Topic1": cells[:100],  # all cells of group A
            "Topic2": cells,  # all cells: no group enriched
            "Topic3": cells[:100] + cells[150:300],  # group A + most other cells
        }
        annotation = topic_annotation(
            adata, "group", binarized_cell_topic=binarized, general_topic_thr=0.2
        )
        assert annotation["group"].tolist() == ["A", "", "A, C"]
        assert annotation["is_general"].tolist() == [False, True, False]
        assert annotation.loc["Topic3", "Ratio_cells_in_topic"] == pytest.approx(250 / 300)


def test_plot_topic_qc(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("adjustText")
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from pycisTopic.topic_qc import plot_topic_qc

    topics = pd.DataFrame(
        {
            "Coherence": [-1.0, -0.5, -2.0],
            "Log10_Assignments": [5.0, 6.0, 4.0],
            "Gini_index": [0.2, 0.5, 0.8],
        },
        index=["Topic1", "Topic2", "Topic3"],
    )
    selected = plot_topic_qc(
        topics,
        var_x="Coherence",
        var_y="Log10_Assignments",
        min_x=-1.5,
        var_color="Gini_index",
        plot=False,
        save=str(tmp_path / "topic_qc.png"),
        return_topics=True,
    )
    assert selected == ["Topic1", "Topic2"]
    assert (tmp_path / "topic_qc.png").exists()

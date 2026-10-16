from __future__ import annotations

import warnings
from itertools import compress
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pycisTopic.adata import OBSM_TOPIC_KEY, get_topic_names, get_topics

if TYPE_CHECKING:
    import anndata


def compute_topic_metrics(adata: anndata.AnnData) -> pd.DataFrame:
    """
    Compute topic quality control metrics.

    Parameters
    ----------
    adata
        pycisTopic AnnData object (see :mod:`pycisTopic.adata`). The number of
        assignments, coherence and marginal topic distribution per topic are taken
        from ``uns["pycistopic"]["topics"]``, which is filled by
        ``pycistopic topic_modeling mallet create_anndata`` when the model statistics
        (``pycistopic topic_modeling mallet stats``) were calculated first.

    Returns
    -------
    Data frame (topics x metrics) with the number of assignments (and its log10), the
    topic coherence (Mimno et al., 2011), the marginal topic distribution (which
    indicates how much each topic contributes to the model) and the Gini index of the
    cell-topic distribution (which indicates the specificity of topics).
    The metrics are also stored in ``uns["pycistopic"]["topics"]``.

    References
    ----------
    Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011).
    Optimizing semantic coherence in topic models.
    In Proceedings of the 2011 Conference on Empirical Methods in Natural Language
    Processing (pp. 262-272).

    """
    topics = get_topics(adata)
    missing = [
        column
        for column in ("Assignments", "Coherence", "Marginal_topic_dist")
        if column not in topics.columns
    ]
    if missing:
        warnings.warn(
            f"Topic statistics {missing} are not available. Run `pycistopic "
            "topic_modeling mallet stats` before `create_anndata` to include them.",
            stacklevel=2,
        )
    if "Assignments" in topics.columns:
        topics["Log10_Assignments"] = np.log10(topics["Assignments"])
    cell_topic = np.asarray(adata.obsm[OBSM_TOPIC_KEY])
    topics["Gini_index"] = [
        gini_coefficient(cell_topic[:, topic_idx])
        for topic_idx in range(cell_topic.shape[1])
    ]
    return topics.copy()


def plot_topic_qc(
    topic_qc_metrics: pd.DataFrame,
    var_x: str,
    var_y: str,
    min_x: int | None = None,
    max_x: int | None = None,
    min_y: int | None = None,
    max_y: int | None = None,
    var_color: str | None = None,
    cmap: str = "viridis",
    dot_size: int = 10,
    text_size: int = 10,
    plot: bool = False,
    save: str | None = None,
    return_topics: bool = False,
    return_fig: bool = False,
):
    """
    Plotting topic qc metrics and filtering.

    Parameters
    ----------
    topic_qc_metrics
        A topic metrics dataframe.
    var_x
        Metric to plot.
    var_y
        A second metric to plot in combination with `var_x`.
    min_x
        Minimum value on `var_x` to keep the barcode/cell. Default: None.
    max_x
        Maximum value on `var_x` to keep the barcode/cell. Default: None.
    min_y
        Minimum value on `var_y` to keep the barcode/cell. Default: None.
    max_y
        Maximum value on `var_y` to keep the barcode/cell. Default: None.
    var_color
        Metric to color plot by. Default: None
    cmap
        Color map to color 2D dot plots by density. Default: None.
    dot_size
        Dot size in the plot. Default: 10
    text_size
        Size of the labels in the plot. Default: 10
    plot
        Whether the plots should be returned to the console. Default: True.
    save
        Path to save plots as a file. Default: None.
    return_topics
        Whether to return selected topics based on user-given thresholds.
        Default: True.
    return_fig
        Whether to return the plot figure; if several samples it will return a
        dictionary with the figures per sample. Default: False.

    Returns
    -------
    A list with the selected topics.

    """
    import matplotlib.cm as cm
    import matplotlib.patheffects as PathEffects
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    # Plot xy.
    fig = plt.figure()
    if var_color is not None:
        plt.scatter(
            topic_qc_metrics[var_x],
            topic_qc_metrics[var_y],
            c=topic_qc_metrics[var_color],
            cmap=cmap,
            s=dot_size,
        )
    else:
        plt.scatter(topic_qc_metrics[var_x], topic_qc_metrics[var_y], s=dot_size)

    # Topics.
    n = topic_qc_metrics.index.tolist()

    plt.xlabel(var_x, fontsize=10)
    plt.ylabel(var_y, fontsize=10)
    # Add topic number.
    texts = []
    for i, _txt in enumerate(n):
        texts.append(
            plt.text(
                topic_qc_metrics[var_x][i],
                topic_qc_metrics[var_y][i],
                i + 1,
                horizontalalignment="center",
                verticalalignment="center",
                size=text_size,
                weight="bold",
                path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")],
            )
        )
    adjust_text(texts, arrowprops={"arrowstyle": "-", "color": "gray", "alpha": 0.5})

    # Add limits.
    x = topic_qc_metrics[var_x]
    y = topic_qc_metrics[var_y]
    if min_x is not None:
        plt.axvline(x=min_x, color="skyblue", linestyle="--")
        n = list(compress(n, x > min_x))
        y = y[list(x > min_x)]
        x = x[list(x > min_x)]
    if max_x is not None:
        plt.axvline(x=max_x, color="tomato", linestyle="--")
        n = list(compress(n, x < max_x))
        y = y[list(x < max_x)]
        x = x[list(x < max_x)]
    if min_y is not None:
        plt.axhline(y=min_y, color="skyblue", linestyle="--")
        n = list(compress(n, y > min_y))
        x = x[list(y > min_y)]
        y = y[list(y > min_y)]
    if max_y is not None:
        plt.axhline(y=max_y, color="tomato", linestyle="--")
        n = list(compress(n, y < max_y))
        x = x[list(y < max_y)]
        y = y[list(y < max_y)]

    # Setup the colorbar.
    if var_color is not None:
        scalarmappaple = cm.ScalarMappable(cmap=cmap)
        scalarmappaple.set_array(topic_qc_metrics[var_color])
        cbar = plt.colorbar(scalarmappaple)
        cbar.set_label(var_color, rotation=270, labelpad=15)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    if plot is not False:
        plt.show()
    else:
        plt.close(fig)

    if return_topics is True:
        if return_fig is True:
            return fig, n
        else:
            return n
    else:
        if return_fig is True:
            return fig


def topic_annotation(
    adata: anndata.AnnData,
    annot_var: str,
    binarized_cell_topic: dict[str, list[str]] | None = None,
    general_topic_thr: float = 0.2,
    **kwargs,
) -> pd.DataFrame:
    """
    Automatic annotation of topics.

    For each group of cells (``adata.obs[annot_var]``), the fraction of cells of the
    group in a binarized topic is compared to the fraction of all cells in that topic
    (one-sided proportion z-test, p < 0.05).

    Parameters
    ----------
    adata
        pycisTopic AnnData object (see :mod:`pycisTopic.adata`).
    annot_var
        Name of the column in ``adata.obs`` to use for annotation.
    binarized_cell_topic
        Dictionary with topic names as keys and the cell barcodes in the binarized
        topic as values. If not provided, the cell-topic distributions are binarized
        with :func:`pycisTopic.topic_binarization.binarize_topics`.
    general_topic_thr
        Threshold for considering a topic as general. After assigning topics to
        annotations, the ratio of cells in the binarized topic in the whole
        population is compared with the ratio of the total number of cells in the
        assigned groups versus the whole population.
        If the difference is above this threshold, the topic is considered general.
    **kwargs
        Arguments to pass to :func:`pycisTopic.topic_binarization.binarize_topics`.

    Returns
    -------
    Data frame (topics x annotation) containing a column with the annotations
    (separated by ``", "``), the ratio of cells in the binarized topic, the ratio of
    cells assigned to a topic based on the annotated groups and whether the topic is
    general. The columns are also stored in ``uns["pycistopic"]["topics"]``.

    """
    names = get_topic_names(adata)
    cell_names = adata.obs_names.tolist()
    n_cells = len(cell_names)
    if binarized_cell_topic is None:
        from pycisTopic.topic_binarization import binarize_topics

        binarized_cells_per_topic, _, _ = binarize_topics(
            np.asarray(adata.obsm[OBSM_TOPIC_KEY]), cell_names, **kwargs
        )
        binarized_cell_topic = dict(zip(names, binarized_cells_per_topic))

    annot = adata.obs[annot_var]
    groups = sorted(annot.dropna().astype(str).unique())
    group_cells = {
        group: set(annot.index[annot.astype(str) == group]) for group in groups
    }
    binarized_cells = {topic: set(binarized_cell_topic[topic]) for topic in names}

    annotations, ratio_cells_in_topic, ratio_group_in_population = [], [], []
    for topic in names:
        value = len(binarized_cells[topic]) / n_cells
        assigned_groups = []
        assigned_group_size = 0
        for group in groups:
            count = len(group_cells[group] & binarized_cells[topic])
            nobs = len(group_cells[group])
            if _proportion_ztest_larger(count, nobs, value) < 0.05:
                assigned_groups.append(group)
                assigned_group_size += nobs
        annotations.append(", ".join(assigned_groups))
        ratio_cells_in_topic.append(value)
        ratio_group_in_population.append(assigned_group_size / n_cells)

    topic_annot = pd.DataFrame(
        {
            annot_var: annotations,
            "Ratio_cells_in_topic": ratio_cells_in_topic,
            "Ratio_group_in_population": ratio_group_in_population,
        },
        index=names,
    )
    topic_annot["is_general"] = (
        topic_annot["Ratio_cells_in_topic"] - topic_annot["Ratio_group_in_population"]
    ) > general_topic_thr

    topics = get_topics(adata)
    for column in topic_annot.columns:
        topics[column] = topic_annot[column]
    return topic_annot


def _proportion_ztest_larger(count: int, nobs: int, value: float) -> float:
    """
    One-sample z-test for a proportion with alternative "larger".

    Same as ``statsmodels.stats.proportion.proportions_ztest(count, nobs, value=value,
    alternative="larger")``: the variance is estimated from the observed proportion.
    """
    from scipy.stats import norm

    prop = count / nobs
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = np.float64(prop - value) / np.sqrt(prop * (1 - prop) / nobs)
    return float(norm.sf(zstat))


def gini_coefficient(x):
    """Compute Gini coefficient of array of values."""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))

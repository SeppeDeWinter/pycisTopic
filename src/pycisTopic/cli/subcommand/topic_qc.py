from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_topic_qc(args):
    import anndata

    from pycisTopic.topic_qc import compute_topic_metrics, topic_annotation

    adata = anndata.read_h5ad(args.adata)

    print("Computing topic QC metrics ...")
    topics = compute_topic_metrics(adata)

    if args.annotation is not None:
        if args.annotation not in adata.obs.columns:
            raise ValueError(f'Column "{args.annotation}" not found in .obs.')
        print(f'Annotating topics with "{args.annotation}" ...')
        topic_annotation(
            adata,
            annot_var=args.annotation,
            general_topic_thr=args.general_topic_thr,
            method=args.binarize_method,
            smooth_topics=args.smooth_topics,
            nbins=args.nbins,
        )
        topics = adata.uns["pycistopic"]["topics"]

    if args.table is not None:
        print(f'Writing topic QC table to "{args.table}".')
        topics.to_csv(args.table, sep="\t", index_label="Topic")

    if args.plot is not None:
        from pycisTopic.topic_qc import plot_topic_qc

        missing = [c for c in (args.var_x, args.var_y) if c not in topics.columns]
        if missing:
            raise ValueError(
                f"Topic QC metrics {missing} are not available (run `pycistopic "
                "topic_modeling lda stats` (or `mallet stats`) before `create_anndata`)."
            )
        print(f'Writing topic QC plot to "{args.plot}".')
        plot_topic_qc(
            topics,
            var_x=args.var_x,
            var_y=args.var_y,
            var_color=args.var_color if args.var_color in topics.columns else None,
            save=args.plot,
        )

    print(f'Writing AnnData with topic QC metrics to "{args.output}".')
    adata.write_h5ad(args.output)


def add_parser_topic_qc(subparsers: _SubParsersAction[ArgumentParser]):
    parser = subparsers.add_parser(
        "topic_qc",
        help="Compute topic QC metrics and annotate topics.",
        description="Compute topic QC metrics (number of assignments, coherence, "
        "marginal topic distribution, Gini index) and optionally annotate topics with "
        "a cell annotation in .obs. Results are stored in .uns['pycistopic']['topics'] "
        "of the AnnData file.",
    )
    parser.set_defaults(func=run_topic_qc)
    parser.add_argument(
        "-a",
        "--adata",
        dest="adata",
        required=True,
        help="pycisTopic AnnData (.h5ad) file with the topic model.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        help="Output AnnData (.h5ad) file (can be the same as --adata).",
    )
    parser.add_argument(
        "--annotation",
        dest="annotation",
        default=None,
        help="Column in .obs (e.g. cell type) to annotate topics with.",
    )
    parser.add_argument(
        "--general_topic_thr",
        dest="general_topic_thr",
        type=float,
        default=0.2,
        help="Threshold for considering a topic as general. Default: 0.2.",
    )
    parser.add_argument(
        "-m",
        "--binarize_method",
        dest="binarize_method",
        choices=("otsu", "yen", "li", "aucell"),
        default="otsu",
        help="Method to binarize cell-topic probabilities for topic annotation. "
        "Default: otsu.",
    )
    parser.add_argument(
        "--no_smooth",
        dest="smooth_topics",
        action="store_false",
        help="Do not smooth cell-topic probabilities before binarization.",
    )
    parser.add_argument(
        "-b",
        "--nbins",
        dest="nbins",
        type=int,
        default=100,
        help="Number of bins used for otsu, yen and li thresholding. Default: 100.",
    )
    parser.add_argument(
        "--table",
        dest="table",
        default=None,
        help="Write topic QC metrics (and annotation) to this TSV file.",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        default=None,
        help="Write a topic QC plot to this file (requires matplotlib).",
    )
    parser.add_argument(
        "--var_x",
        dest="var_x",
        default="Coherence",
        help="Metric on the x-axis of the plot. Default: Coherence.",
    )
    parser.add_argument(
        "--var_y",
        dest="var_y",
        default="Log10_Assignments",
        help="Metric on the y-axis of the plot. Default: Log10_Assignments.",
    )
    parser.add_argument(
        "--var_color",
        dest="var_color",
        default="Gini_index",
        help="Metric to color topics by in the plot. Default: Gini_index.",
    )

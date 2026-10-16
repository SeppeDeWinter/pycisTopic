from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_gene_activity(args):
    import logging
    import sys

    import anndata
    import pandas as pd
    import polars as pl

    from pycisTopic.adata import get_cell_topic, get_region_topic
    from pycisTopic.consensus_peaks import read_chrom_sizes
    from pycisTopic.gene_activity import get_gene_activity
    from pycisTopic.gene_annotation import read_tss_annotation_from_bed

    log = None
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
        log = logging.getLogger("pycisTopic")

    adata = anndata.read_h5ad(args.adata, backed="r")
    chrom_sizes = read_chrom_sizes(args.chrom_sizes)
    predefined_boundaries = None
    if args.boundaries is not None:
        from pycisTopic.fragments import read_bed_to_polars_df

        predefined_boundaries = read_bed_to_polars_df(args.boundaries)

    gene_activity, gene_names, region_weights = get_gene_activity(
        region_topic=get_region_topic(adata),
        cell_topic=get_cell_topic(adata),
        region_names=adata.var_names.tolist(),
        gene_annotation=read_tss_annotation_from_bed(args.tss_annotation),
        chrom_sizes=pl.DataFrame(
            {
                "Chromosome": list(chrom_sizes),
                "Start": [0] * len(chrom_sizes),
                "End": list(chrom_sizes.values()),
            }
        ),
        predefined_boundaries=predefined_boundaries,
        use_gene_boundaries=args.use_gene_boundaries,
        upstream=tuple(args.upstream),
        downstream=tuple(args.downstream),
        distance_weight=args.distance_weight,
        decay_rate=args.decay_rate,
        extend_gene_body_upstream=args.extend_gene_body_upstream,
        extend_gene_body_downstream=args.extend_gene_body_downstream,
        gene_size_weight=args.gene_size_weight,
        remove_promoters=args.remove_promoters,
        extend_tss=tuple(args.extend_tss),
        gini_weight=args.gini_weight,
        average_scores=args.average_scores,
        scale_factor=args.scale_factor,
        chunk_size=args.chunk_size,
        log=log,
    )

    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in ("func", "adata", "output", "weights", "verbose")
        and value is not None
    }
    gene_activity_adata = anndata.AnnData(
        X=gene_activity,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(gene_names)),
        obsm=dict(adata.obsm.items()),
        uns={"pycistopic": {"gene_activity": parameters}},
    )
    print(
        f"Writing gene activity ({gene_activity_adata.n_obs} cells x "
        f'{gene_activity_adata.n_vars} genes) to "{args.output}".'
    )
    gene_activity_adata.write_h5ad(args.output)

    if args.weights is not None:
        print(f'Writing region weights to "{args.weights}".')
        region_weights.write_csv(args.weights, separator="\t")


def add_parser_gene_activity(subparsers: _SubParsersAction[ArgumentParser]):
    parser = subparsers.add_parser(
        "gene_activity",
        help="Infer gene activity from the topic model.",
        description="Infer gene activity by aggregating imputed accessibility of the "
        "regions in the search space of each gene, weighted by distance to the gene, "
        "region specificity (Gini index) and optionally gene size. The result is "
        "written as AnnData (cells x genes) with .obs and .obsm of the topic model.",
    )
    parser.set_defaults(func=run_gene_activity)
    parser.add_argument(
        "-a", "--adata", dest="adata", required=True,
        help="pycisTopic AnnData (.h5ad) file with the topic model.",
    )  # fmt: skip
    parser.add_argument(
        "-t", "--tss_annotation", dest="tss_annotation", required=True,
        help="TSS annotation BED file (output of `pycistopic tss get_tss`), using the "
        "same chromosome names as the regions.",
    )  # fmt: skip
    parser.add_argument(
        "-c", "--chrom_sizes", dest="chrom_sizes", required=True,
        help="Chromosome sizes file (*.chrom.sizes or *.fa.fai).",
    )  # fmt: skip
    parser.add_argument(
        "-o", "--output", dest="output", required=True,
        help="Output AnnData (.h5ad) file with gene activity (cells x genes).",
    )  # fmt: skip
    parser.add_argument(
        "--weights", dest="weights", default=None,
        help="Write region-gene links and weights to this TSV file.",
    )  # fmt: skip
    parser.add_argument(
        "--boundaries", dest="boundaries", default=None,
        help="BED file with predefined domain boundaries (e.g. TAD boundaries) "
        "limiting the search space of genes (instead of promoters of other genes).",
    )  # fmt: skip
    parser.add_argument(
        "--no_gene_boundaries", dest="use_gene_boundaries", action="store_false",
        help="Do not stop the search space at promoters of other genes.",
    )  # fmt: skip
    parser.add_argument(
        "--upstream", dest="upstream", type=int, nargs=2, default=[1000, 100000],
        metavar=("MIN", "MAX"),
        help="Minimum and maximum search space upstream of genes. Default: 1000 100000.",
    )  # fmt: skip
    parser.add_argument(
        "--downstream", dest="downstream", type=int, nargs=2, default=[1000, 100000],
        metavar=("MIN", "MAX"),
        help="Minimum and maximum search space downstream of genes. "
        "Default: 1000 100000.",
    )  # fmt: skip
    parser.add_argument(
        "--no_distance_weight", dest="distance_weight", action="store_false",
        help="Do not weight regions by their distance to the gene.",
    )  # fmt: skip
    parser.add_argument(
        "--decay_rate", dest="decay_rate", type=float, default=1.0,
        help="Decay rate of the distance weight. Default: 1.0.",
    )  # fmt: skip
    parser.add_argument(
        "--extend_gene_body_upstream", dest="extend_gene_body_upstream", type=int,
        default=5000,
        help="Base pairs upstream of the gene with maximal distance weight. "
        "Default: 5000.",
    )  # fmt: skip
    parser.add_argument(
        "--extend_gene_body_downstream", dest="extend_gene_body_downstream",
        type=int, default=0,
        help="Base pairs downstream of the gene with maximal distance weight. "
        "Default: 0.",
    )  # fmt: skip
    parser.add_argument(
        "--gene_size_weight", dest="gene_size_weight", action="store_true",
        help="Weight regions by the size of the gene (median gene size / gene size).",
    )  # fmt: skip
    parser.add_argument(
        "--remove_promoters", dest="remove_promoters", action="store_true",
        help="Ignore distal regions overlapping promoters of other genes.",
    )  # fmt: skip
    parser.add_argument(
        "--extend_tss", dest="extend_tss", type=int, nargs=2, default=[10, 10],
        metavar=("UPSTREAM", "DOWNSTREAM"),
        help="Space around the TSS considered as promoter. Default: 10 10.",
    )  # fmt: skip
    parser.add_argument(
        "--no_gini_weight", dest="gini_weight", action="store_false",
        help="Do not weight regions by the Gini index of their imputed accessibility.",
    )  # fmt: skip
    parser.add_argument(
        "--no_average", dest="average_scores", action="store_false",
        help="Sum instead of average the weighted imputed accessibility of regions.",
    )  # fmt: skip
    parser.add_argument(
        "--scale_factor", dest="scale_factor", type=float, default=1.0,
        help="Multiply gene activity by this value (if not 1, values are rounded and "
        "genes without activity are removed). Default: 1.",
    )  # fmt: skip
    parser.add_argument(
        "--chunk_size", dest="chunk_size", type=int, default=20000,
        help="Number of regions for which imputed accessibility is calculated at "
        "once. Default: 20000.",
    )  # fmt: skip
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Print progress."
    )

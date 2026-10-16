from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_convert(args):
    from pycisTopic.legacy import cistopic_object_to_anndata, read_legacy_pickle

    print(f'Reading CistopicObject from "{args.cistopic_object}" ...')
    cistopic_obj = read_legacy_pickle(args.cistopic_object)
    adata = cistopic_object_to_anndata(
        cistopic_obj, store_matrix=args.store_matrix
    )
    print(f"Created AnnData object with shape: {adata.shape}")
    print(f'Writing AnnData to "{args.output_adata}" ...')
    adata.write_h5ad(args.output_adata)


def add_parser_convert(subparsers: _SubParsersAction[ArgumentParser]):
    """Creates an ArgumentParser to read the options for this script."""
    parser_convert = subparsers.add_parser(
        "convert",
        help="Convert a CistopicObject (pycisTopic < 3) to a pycisTopic AnnData file.",
        description="Convert a pickled CistopicObject of pycisTopic < 3 to a pycisTopic "
        "AnnData (.h5ad) file (cells x regions) with the selected topic model in "
        ".obsm['X_topic'] and .varm['topic'], cell data in .obs, region data in .var "
        "and cell projections (e.g. UMAP) in .obsm. Only convert pickles from trusted "
        "sources: unpickling can execute arbitrary code.",
    )
    parser_convert.set_defaults(func=run_convert)

    parser_convert.add_argument(
        "-i",
        "--cistopic_object",
        dest="cistopic_object",
        action="store",
        type=str,
        required=True,
        help="Path to the pickled CistopicObject file to convert.",
    )
    parser_convert.add_argument(
        "-o",
        "--output_adata",
        dest="output_adata",
        action="store",
        type=str,
        required=True,
        help="Path to the output .h5ad file.",
    )
    parser_convert.add_argument(
        "--store_matrix",
        dest="store_matrix",
        action="store_true",
        help="Store the binary accessibility matrix in .X. Default: do not store it "
        "(smaller file).",
    )

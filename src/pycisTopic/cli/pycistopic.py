from __future__ import annotations

import argparse
import os

from pycisTopic.cli.subcommand.convert import add_parser_convert
from pycisTopic.cli.subcommand.count_matrix import add_parser_count_matrix
from pycisTopic.cli.subcommand.dars import add_parser_dars
from pycisTopic.cli.subcommand.gene_activity import add_parser_gene_activity
from pycisTopic.cli.subcommand.peak_calling import add_parser_peak_calling
from pycisTopic.cli.subcommand.qc import add_parser_qc
from pycisTopic.cli.subcommand.signature_enrichment import (
    add_parser_signature_enrichment,
)
from pycisTopic.cli.subcommand.topic_modeling import add_parser_topic_modeling
from pycisTopic.cli.subcommand.topic_qc import add_parser_topic_qc
from pycisTopic.cli.subcommand.tss import add_parser_tss

# Arguments which are always output files (or output prefixes) or output directories.
OUTPUT_FILE_ARGUMENTS = (
    "output_prefix",
    "output",
    "output_bed",
    "output_adata",
    "out_file",
    "weights",
    "table",
    "plot",
    "hv_plot",
)
OUTPUT_DIR_ARGUMENTS = ("output_dir", "out_dir")


def create_output_dirs(args: argparse.Namespace) -> None:
    """
    Create output directories before running a command.

    Avoids that a command fails at the end, after all calculations are done, because
    the directory of an output file does not exist.
    """
    for argument in OUTPUT_FILE_ARGUMENTS + OUTPUT_DIR_ARGUMENTS:
        path = getattr(args, argument, None)
        if not isinstance(path, str) or not path:
            continue
        directory = path if argument in OUTPUT_DIR_ARGUMENTS else os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="pycisTopic CLI.")

    subparsers = parser.add_subparsers(
        title="Commands",
        description="List of available commands for pycisTopic CLI.",
        dest="command",
        help="Command description.",
    )
    subparsers.required = True

    add_parser_qc(subparsers)
    add_parser_topic_modeling(subparsers)
    add_parser_topic_qc(subparsers)
    add_parser_tss(subparsers)
    add_parser_count_matrix(subparsers)
    add_parser_dars(subparsers)
    add_parser_signature_enrichment(subparsers)
    add_parser_gene_activity(subparsers)
    add_parser_peak_calling(subparsers)
    add_parser_convert(subparsers)

    args = parser.parse_args()
    create_output_dirs(args)
    args.func(args)


if __name__ == "__main__":
    main()

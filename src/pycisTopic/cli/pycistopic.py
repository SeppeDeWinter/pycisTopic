from __future__ import annotations

import argparse

from pycisTopic.cli.subcommand.count_matrix import add_parser_count_matrix
from pycisTopic.cli.subcommand.qc import add_parser_qc
from pycisTopic.cli.subcommand.signature_enrichment import (
    add_parser_signature_enrichment,
)
from pycisTopic.cli.subcommand.topic_modeling import add_parser_topic_modeling
from pycisTopic.cli.subcommand.topic_qc import add_parser_topic_qc
from pycisTopic.cli.subcommand.tss import add_parser_tss
from pycisTopic.cli.subcommand.dars import add_parser_dars
from pycisTopic.cli.subcommand.gene_activity import add_parser_gene_activity
from pycisTopic.cli.subcommand.peak_calling import add_parser_peak_calling
from pycisTopic.cli.subcommand.convert import add_parser_convert


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
    args.func(args)


if __name__ == "__main__":
    main()

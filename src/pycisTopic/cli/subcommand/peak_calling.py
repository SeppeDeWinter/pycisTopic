from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_pseudobulk(args):
    import polars as pl

    from pycisTopic.consensus_peaks import read_chrom_sizes
    from pycisTopic.count_matrix import (
        COL_NAME_FRAGMENTS_FILENAME,
        COL_NAME_SAMPLE,
        FIELD_SEP,
        read_mapping,
    )
    from pycisTopic.peak_calling import export_pseudobulk

    if (args.adata is None) == (args.cell_data is None):
        raise ValueError("Provide either an AnnData file (-a) or cell data (--cell_data).")

    if args.adata is not None:
        import anndata

        cell_data = anndata.read_h5ad(args.adata, backed="r").obs
        barcode_column = None
    else:
        cell_data = pl.read_csv(args.cell_data, separator="\t", infer_schema=False)
        barcode_column = args.barcode_column

    if args.group_by not in cell_data.columns:
        raise ValueError(f'Column "{args.group_by}" not found in the cell data.')

    pseudobulk_files = export_pseudobulk(
        cell_data=cell_data,
        group_by=args.group_by,
        sample_to_fragments=read_mapping(
            filename=args.sample_to_fragments,
            key=COL_NAME_SAMPLE,
            value=COL_NAME_FRAGMENTS_FILENAME,
            separator=FIELD_SEP,
        ),
        chrom_sizes=read_chrom_sizes(args.chrom_sizes),
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        barcode_column=barcode_column,
        cb_sample_separator=args.cb_sample_separator,
        cb_end_to_add=args.cb_end_to_add,
        n_cpu=args.n_cpu,
        verbose=args.verbose,
    )
    for name, filename in pseudobulk_files.items():
        print(f"{name}\t{filename}")


def run_callpeaks(args):
    from pycisTopic.peak_calling import call_peaks_macs3, find_pseudobulk_fragment_files

    fragments_files = find_pseudobulk_fragment_files(args.pseudobulk_dir)
    if not fragments_files:
        raise FileNotFoundError(
            f'No "*.fragments.tsv.gz" files found in "{args.pseudobulk_dir}".'
        )
    summits_files = call_peaks_macs3(
        fragments_files=fragments_files,
        output_dir=args.output_dir,
        genome_size=args.genome_size,
        qvalue=args.qvalue,
        shift=args.shift,
        extsize=args.extsize,
        n_jobs=args.n_jobs,
        macs3_path=args.macs3_path,
    )
    for name, filename in summits_files.items():
        print(f"{name}\t{filename}")


def run_consensus_peaks(args):
    from pycisTopic.consensus_peaks import (
        get_consensus_peaks,
        read_chrom_sizes,
        read_peaks_bed,
        write_peaks_bed,
    )
    from pycisTopic.fragments import read_bed_to_polars_df

    peaks_files: list[str] = []
    for path in args.peaks:
        if os.path.isdir(path):
            peaks_files.extend(
                os.path.join(path, f)
                for f in sorted(os.listdir(path))
                if f.endswith("_summits.bed")
            )
        else:
            peaks_files.append(path)
    if not peaks_files:
        raise FileNotFoundError("No peak files (*_summits.bed) found.")

    peaks_per_sample = {}
    for peaks_file in peaks_files:
        name = os.path.basename(peaks_file).removesuffix(".bed").removesuffix("_summits")
        print(f'Reading peaks for "{name}" from "{peaks_file}".')
        peaks_per_sample[name] = read_peaks_bed(peaks_file)

    blacklist = None
    if args.blacklist is not None:
        import polars as pl

        blacklist = read_bed_to_polars_df(args.blacklist).with_columns(
            pl.col("Chromosome").cast(pl.Utf8)
        )

    consensus_peaks = get_consensus_peaks(
        peaks_per_sample=peaks_per_sample,
        chrom_sizes=read_chrom_sizes(args.chrom_sizes),
        peak_half_width=args.peak_half_width,
        blacklist=blacklist,
    )
    print(f'Writing {consensus_peaks.height} consensus peaks to "{args.output_bed}".')
    write_peaks_bed(consensus_peaks, args.output_bed)


def add_parser_peak_calling(subparsers: _SubParsersAction[ArgumentParser]):
    """Add pseudobulk, callpeaks and consensus_peaks subcommands."""
    # ---------- pseudobulk ----------
    parser_pseudobulk = subparsers.add_parser(
        "pseudobulk",
        help="Create pseudobulk fragment files per group of cells.",
        description="Create pseudobulk fragment files ({group}.fragments.tsv.gz) per "
        "group of cells (e.g. cell type) with scatac_fragment_tools. Spaces and "
        "slashes in group names are replaced by underscores. Fragment files need to "
        "be bgzipped and tabix indexed.",
    )
    parser_pseudobulk.set_defaults(func=run_pseudobulk)
    parser_pseudobulk.add_argument(
        "-a",
        "--adata",
        dest="adata",
        default=None,
        help="pycisTopic AnnData (.h5ad) file: cell barcodes (CB___sample_id) and "
        "groups are taken from .obs.",
    )
    parser_pseudobulk.add_argument(
        "--cell_data",
        dest="cell_data",
        default=None,
        help="TSV file with cell barcodes (CB___sample_id) and groups (alternative to "
        "-a).",
    )
    parser_pseudobulk.add_argument(
        "--barcode_column",
        dest="barcode_column",
        default="cell_barcode",
        help="Column with cell barcodes in --cell_data. Default: cell_barcode.",
    )
    parser_pseudobulk.add_argument(
        "-g",
        "--group_by",
        dest="group_by",
        required=True,
        help="Column with the groups (e.g. cell types) to create pseudobulks for.",
    )
    parser_pseudobulk.add_argument(
        "-f",
        "--sample_fragments",
        dest="sample_to_fragments",
        required=True,
        help="TSV file mapping sample IDs to fragment files (columns: sample, "
        "fragments_filename), as used by `pycistopic count_matrix create`.",
    )
    parser_pseudobulk.add_argument(
        "-c",
        "--chrom_sizes",
        dest="chrom_sizes",
        required=True,
        help="Chromosome sizes file (*.chrom.sizes or *.fa.fai).",
    )
    parser_pseudobulk.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Output directory for pseudobulk fragment files.",
    )
    parser_pseudobulk.add_argument(
        "-t",
        "--temp_dir",
        dest="temp_dir",
        default=None,
        help="Directory for temporary files. Default: OUTPUT_DIR/tmp.",
    )
    parser_pseudobulk.add_argument(
        "-s",
        "--sample_sep",
        dest="cb_sample_separator",
        default="___",
        help='Separator between cell barcode and sample ID. Default: "___".',
    )
    parser_pseudobulk.add_argument(
        "-d",
        "--cb_end_add",
        dest="cb_end_to_add",
        default=None,
        help='Suffix to add to cell barcodes to match the fragment files (e.g. "-1" if '
        "it was removed when creating the count matrix). Default: None.",
    )
    parser_pseudobulk.add_argument(
        "-n",
        "--n_cpu",
        dest="n_cpu",
        type=int,
        default=1,
        help="Number of cores to use. Default: 1.",
    )
    parser_pseudobulk.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Print progress."
    )

    # ---------- callpeaks ----------
    parser_callpeaks = subparsers.add_parser(
        "callpeaks",
        help="Call peaks on pseudobulk fragment files with MACS3.",
        description="Call peaks with MACS3 (macs3 callpeak --format FRAG --max-count 1 "
        "--nomodel --nolambda --call-summits) on each *.fragments.tsv.gz file in a "
        "directory.",
    )
    parser_callpeaks.set_defaults(func=run_callpeaks)
    parser_callpeaks.add_argument(
        "-i",
        "--bulk_path",
        dest="pseudobulk_dir",
        required=True,
        help="Directory with pseudobulk fragment files (*.fragments.tsv.gz).",
    )
    parser_callpeaks.add_argument(
        "-o",
        "--outdir",
        dest="output_dir",
        required=True,
        help="Output directory for MACS3 output files.",
    )
    parser_callpeaks.add_argument(
        "-g",
        "--gsize",
        dest="genome_size",
        required=True,
        help="Effective genome size: number or MACS3 shortcut (hs, mm, ce, dm).",
    )
    parser_callpeaks.add_argument(
        "-q",
        "--qvalue",
        dest="qvalue",
        type=float,
        default=0.05,
        help="Q-value cutoff to call significant regions. Default: 0.05.",
    )
    parser_callpeaks.add_argument(
        "--shift",
        dest="shift",
        type=int,
        default=73,
        help="Shift (bp) of cut sites. Default: 73.",
    )
    parser_callpeaks.add_argument(
        "--extsize",
        dest="extsize",
        type=int,
        default=146,
        help="Extension size (bp) of cut sites. Default: 146.",
    )
    parser_callpeaks.add_argument(
        "-n",
        "--ncores",
        dest="n_jobs",
        type=int,
        default=1,
        help="Number of MACS3 processes to run in parallel. Default: 1.",
    )
    parser_callpeaks.add_argument(
        "-e",
        "--exe",
        dest="macs3_path",
        default="macs3",
        help="MACS3 executable. Default: macs3.",
    )

    # ---------- consensus_peaks ----------
    parser_consensus = subparsers.add_parser(
        "consensus_peaks",
        help="Get consensus peaks from MACS3 summits with iterative peak filtering.",
        description="Get consensus peaks from MACS3 summits (*_summits.bed) per "
        "pseudobulk: summits are extended by --peak_half_width in both directions, "
        "overlapping peaks are filtered iteratively per pseudobulk (keeping the most "
        "significant peak), scores are normalized per pseudobulk and peaks of all "
        "pseudobulks are filtered iteratively again. Consensus peaks have a width of "
        "2 * peak_half_width.",
    )
    parser_consensus.set_defaults(func=run_consensus_peaks)
    parser_consensus.add_argument(
        "-i",
        "--peaks",
        dest="peaks",
        nargs="+",
        required=True,
        help="MACS3 summits BED files or directories with *_summits.bed files.",
    )
    parser_consensus.add_argument(
        "-c",
        "--chrom_sizes",
        dest="chrom_sizes",
        required=True,
        help="Chromosome sizes file (*.chrom.sizes or *.fa.fai). Peaks on other "
        "chromosomes are removed.",
    )
    parser_consensus.add_argument(
        "-o",
        "--output",
        dest="output_bed",
        required=True,
        help="Output BED file with consensus peaks.",
    )
    parser_consensus.add_argument(
        "-w",
        "--peak_half_width",
        dest="peak_half_width",
        type=int,
        default=250,
        help="Number of base pairs to extend summits in each direction. Default: 250.",
    )
    parser_consensus.add_argument(
        "-b",
        "--blacklist",
        dest="blacklist",
        default=None,
        help="BED file with blacklisted regions: consensus peaks overlapping them are "
        "removed.",
    )

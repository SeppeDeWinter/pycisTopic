"""
Pseudobulk fragment files and peak calling with MACS3.

Pseudobulk fragment files are created with ``scatac_fragment_tools``. Peaks are called
with the ``macs3`` executable, which is installed together with pycisTopic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pandas as pd

FRAGMENTS_SUFFIX = ".fragments.tsv.gz"


def pseudobulk_name(group: str) -> str:
    """Pseudobulk name for a group, safe to use in filenames (no spaces or slashes)."""
    return str(group).replace(" ", "_").replace("/", "_")


def get_cell_type_barcodes_per_sample(
    cell_barcodes: Sequence[str],
    groups: Sequence[str],
    samples: Sequence[str],
    cb_sample_separator: str = "___",
    cb_end_to_add: str | None = None,
) -> dict[str, dict[str, list[str]]]:
    """
    Map pycisTopic cell barcodes to the cell barcodes in the fragment files per sample.

    Parameters
    ----------
    cell_barcodes
        pycisTopic cell barcodes (``CB___sample_id``).
    groups
        Group (e.g. cell type) of each cell. Cells without group (None or NaN) are
        skipped. Groups are renamed with :func:`pseudobulk_name`.
    samples
        Sample IDs that have a fragments file. If cell barcodes do not contain a
        sample ID, there should be only one sample.
    cb_sample_separator
        Separator between cell barcode and sample ID.
    cb_end_to_add
        Suffix to add to the cell barcodes to get the barcodes in the fragment files,
        e.g. ``"-1"`` when it was removed with ``cb_end_to_remove``.

    Returns
    -------
    Dictionary with sample IDs as keys and dictionaries with groups as keys and cell
    barcodes (as in the fragment files) as values.

    """
    if len(cell_barcodes) != len(groups):
        raise ValueError(
            f"Number of cell barcodes ({len(cell_barcodes)}) and groups "
            f"({len(groups)}) differ."
        )
    samples = list(samples)
    cell_type_barcodes: dict[str, dict[str, list[str]]] = {s: {} for s in samples}
    for cell_barcode, group in zip(cell_barcodes, groups, strict=True):
        if group is None or group != group:  # Skip cells without group (NaN).
            continue
        if cb_sample_separator and cb_sample_separator in cell_barcode:
            cb, sample = cell_barcode.rsplit(cb_sample_separator, 1)
        elif len(samples) == 1:
            cb, sample = cell_barcode, samples[0]
        else:
            raise ValueError(
                f'Cell barcode "{cell_barcode}" does not contain a sample ID '
                f'(separator: "{cb_sample_separator}"), but there are multiple samples.'
            )
        if sample not in cell_type_barcodes:
            raise ValueError(
                f'No fragments file for sample "{sample}" (from cell barcode '
                f'"{cell_barcode}").'
            )
        if cb_end_to_add:
            cb = cb + cb_end_to_add
        cell_type_barcodes[sample].setdefault(pseudobulk_name(group), []).append(cb)
    return cell_type_barcodes


def export_pseudobulk(
    cell_data: pd.DataFrame | pl.DataFrame,
    group_by: str,
    sample_to_fragments: Mapping[str, str | Path],
    chrom_sizes: Mapping[str, int],
    output_dir: str | Path,
    temp_dir: str | Path | None = None,
    barcode_column: str | None = None,
    cb_sample_separator: str = "___",
    cb_end_to_add: str | None = None,
    n_cpu: int = 1,
    verbose: bool = False,
) -> dict[str, str]:
    """
    Create pseudobulk fragment files per group of cells with ``scatac_fragment_tools``.

    Parameters
    ----------
    cell_data
        Cell annotations, e.g. ``adata.obs`` of a pycisTopic AnnData object.
    group_by
        Column of ``cell_data`` with the groups (e.g. cell types) to create pseudobulks
        for.
    sample_to_fragments
        Dictionary with sample IDs as keys and fragment files as values.
    chrom_sizes
        Chromosome sizes (see :func:`pycisTopic.consensus_peaks.read_chrom_sizes`).
    output_dir
        Output directory for the ``{group}.fragments.tsv.gz`` files.
    temp_dir
        Directory for temporary files. Default: ``{output_dir}/tmp``.
    barcode_column
        Column of ``cell_data`` with pycisTopic cell barcodes (``CB___sample_id``).
        Default: the index (pandas) or ``"cell_barcode"`` (polars).
    cb_sample_separator
        Separator between cell barcode and sample ID.
    cb_end_to_add
        Suffix to add to the cell barcodes to get the barcodes in the fragment files,
        e.g. ``"-1"`` when it was removed with ``cb_end_to_remove``.
    n_cpu
        Number of cores to use.
    verbose
        Print progress.

    Returns
    -------
    Dictionary with pseudobulk names (see :func:`pseudobulk_name`) as keys and
    fragment files as values.

    """
    from scatac_fragment_tools.library.split.split_fragments_by_cell_type import (
        split_fragment_files_by_cell_type,
    )

    if isinstance(cell_data, pl.DataFrame):
        barcode_column = barcode_column or "cell_barcode"
        cell_barcodes = cell_data.get_column(barcode_column).cast(pl.Utf8).to_list()
        groups = cell_data.get_column(group_by).to_list()
    else:
        cell_barcodes = (
            cell_data[barcode_column].astype(str).tolist()
            if barcode_column
            else cell_data.index.astype(str).tolist()
        )
        groups = cell_data[group_by].tolist()

    names = {pseudobulk_name(g) for g in set(groups) if g is not None and g == g}
    n_groups = len({g for g in set(groups) if g is not None and g == g})
    if len(names) != n_groups:
        raise ValueError(
            "Some groups have the same name after replacing spaces and slashes with "
            "underscores."
        )

    cell_type_barcodes = get_cell_type_barcodes_per_sample(
        cell_barcodes=cell_barcodes,
        groups=groups,
        samples=list(sample_to_fragments),
        cb_sample_separator=cb_sample_separator,
        cb_end_to_add=cb_end_to_add,
    )

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = str(temp_dir) if temp_dir else os.path.join(output_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)

    split_fragment_files_by_cell_type(
        sample_to_fragment_file={s: str(f) for s, f in sample_to_fragments.items()},
        path_to_temp_folder=temp_dir,
        path_to_output_folder=output_dir,
        sample_to_cell_type_to_cell_barcodes=cell_type_barcodes,
        chromsizes=dict(chrom_sizes),
        n_cpu=n_cpu,
        verbose=verbose,
        clear_temp_folder=True,
    )
    pseudobulk_files = {
        name: os.path.join(output_dir, f"{name}{FRAGMENTS_SUFFIX}")
        for name in sorted(names)
    }
    missing = [f for f in pseudobulk_files.values() if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"Pseudobulk fragment files were not created: {missing}")
    return pseudobulk_files


def find_pseudobulk_fragment_files(directory: str | Path) -> dict[str, str]:
    """Get ``{name}.fragments.tsv.gz`` files in a directory as ``{name: path}``."""
    return {
        name.removesuffix(FRAGMENTS_SUFFIX): os.path.join(str(directory), name)
        for name in sorted(os.listdir(directory))
        if name.endswith(FRAGMENTS_SUFFIX)
    }


def _check_macs3(macs3_path: str) -> str:
    executable = shutil.which(macs3_path)
    if executable is None and os.sep not in macs3_path:
        # MACS3 installed in the same environment as pycisTopic, which is not
        # necessarily on PATH (e.g. when running a notebook kernel).
        executable = shutil.which(
            macs3_path, path=os.path.dirname(os.path.realpath(sys.executable))
        ) or shutil.which(macs3_path, path=os.path.dirname(sys.executable))
    if executable is None:
        raise FileNotFoundError(
            f'MACS3 executable "{macs3_path}" not found. Install MACS3 ("pip install '
            'macs3") or provide the path to the macs3 executable.'
        )
    return executable


def macs3_callpeak_command(
    fragments_file: str | Path,
    name: str,
    output_dir: str | Path,
    genome_size: str | int,
    qvalue: float = 0.05,
    shift: int = 73,
    extsize: int = 146,
    macs3_path: str = "macs3",
) -> list[str]:
    """
    Get the MACS3 ``callpeak`` command for a pseudobulk fragments file.

    Same settings as used for pycisTopic consensus peaks: fragments are counted once
    (``--max-count 1``), no model is built (``--nomodel --shift --extsize``), the
    background lambda is used as local lambda (``--nolambda``) and summits are called
    (``--call-summits``).
    """
    return [
        macs3_path,
        "callpeak",
        "--treatment",
        str(fragments_file),
        "--name",
        name,
        "--outdir",
        str(output_dir),
        "--format",
        "FRAG",
        "--gsize",
        str(genome_size),
        "--qvalue",
        str(qvalue),
        "--max-count",
        "1",
        "--nomodel",
        "--shift",
        str(shift),
        "--extsize",
        str(extsize),
        "--nolambda",
        "--call-summits",
    ]


def call_peaks_macs3(
    fragments_files: Mapping[str, str | Path],
    output_dir: str | Path,
    genome_size: str | int,
    qvalue: float = 0.05,
    shift: int = 73,
    extsize: int = 146,
    n_jobs: int = 1,
    macs3_path: str = "macs3",
) -> dict[str, str]:
    """
    Call peaks with MACS3 for each (pseudobulk) fragments file.

    Parameters
    ----------
    fragments_files
        Dictionary with names as keys and fragment files as values
        (see :func:`export_pseudobulk` and :func:`find_pseudobulk_fragment_files`).
    output_dir
        Output directory for MACS3 output files.
    genome_size
        Effective genome size (``--gsize``): number or MACS3 shortcut (``"hs"``,
        ``"mm"``, ``"ce"``, ``"dm"``).
    qvalue
        Q-value cutoff to call significant regions.
    shift
        Shift (bp) of cut sites.
    extsize
        Extension size (bp) of cut sites.
    n_jobs
        Number of MACS3 processes to run in parallel.
    macs3_path
        MACS3 executable.

    Returns
    -------
    Dictionary with names as keys and MACS3 summits BED files as values.

    """
    executable = _check_macs3(macs3_path)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def run(name: str, fragments_file: str | Path) -> str:
        command = macs3_callpeak_command(
            fragments_file=fragments_file,
            name=name,
            output_dir=output_dir,
            genome_size=genome_size,
            qvalue=qvalue,
            shift=shift,
            extsize=extsize,
            macs3_path=executable,
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f'MACS3 failed for "{name}" ({" ".join(command)}):\n{result.stderr}'
            )
        return os.path.join(output_dir, f"{name}_summits.bed")

    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as executor:
        futures = {
            name: executor.submit(run, name, fragments_file)
            for name, fragments_file in fragments_files.items()
        }
        return {name: future.result() for name, future in futures.items()}

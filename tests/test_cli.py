from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from pycisTopic.cli import pycistopic


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    for name in dir(pycistopic):
        if name.startswith("add_parser_"):
            getattr(pycistopic, name)(subparsers)
    return parser


def _subcommand_paths(parser: argparse.ArgumentParser, prefix=()):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                yield (*prefix, name)
                yield from _subcommand_paths(subparser, (*prefix, name))


SUBCOMMAND_PATHS = list(_subcommand_paths(_build_parser()))


def test_expected_top_level_commands():
    top_level = {path[0] for path in SUBCOMMAND_PATHS}
    assert {"qc", "topic_modeling", "tss", "count_matrix"} <= top_level


@pytest.mark.parametrize("path", SUBCOMMAND_PATHS, ids=" ".join)
def test_help(path):
    result = subprocess.run(
        [sys.executable, "-m", "pycisTopic.cli.pycistopic", *path, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_leaf_commands_have_a_function():
    parser = _build_parser()
    leaf_paths = [
        path
        for path in SUBCOMMAND_PATHS
        if not any(other[: len(path)] == path and other != path for other in SUBCOMMAND_PATHS)
    ]
    for path in leaf_paths:
        subparser = parser
        for name in path:
            subparser = next(
                a for a in subparser._actions if isinstance(a, argparse._SubParsersAction)
            ).choices[name]
        assert callable(subparser.get_default("func")), path


def test_create_output_dirs(tmp_path):
    import argparse

    from pycisTopic.cli.pycistopic import create_output_dirs

    args = argparse.Namespace(
        output_prefix=str(tmp_path / "a" / "prefix"),
        output=str(tmp_path / "b" / "file.h5ad"),
        output_dir=str(tmp_path / "c" / "d"),
        plot=None,
        regions_bed_filename=str(tmp_path / "input" / "regions.bed"),
    )
    create_output_dirs(args)
    assert (tmp_path / "a").is_dir()
    assert not (tmp_path / "a" / "prefix").exists()
    assert (tmp_path / "b").is_dir()
    assert (tmp_path / "c" / "d").is_dir()
    # Input files are not touched.
    assert not (tmp_path / "input").exists()

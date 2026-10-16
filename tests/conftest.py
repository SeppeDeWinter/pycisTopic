from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Small hand-made dataset with known answers.
#
# Regions (BED):
#   r1 chr1:100-200, r2 chr1:300-400, r3 chr1:1000-1100, r4 chr2:50-150
# Fragments (Chromosome, Start, End, CB, CB_count):
#   AAA-1: chr1:120-180 (x2), chr1:150-350, chr1:500-600
#   BBB-1: chr1:1050-1060 (x3), chr2:60-70, chr1:310-320
#   CCC-1: chr2:60-70 (not in barcodes file)
REGIONS_BED = """\
chr1\t100\t200
chr1\t300\t400
chr1\t1000\t1100
chr2\t50\t150
"""

FRAGMENTS_TSV = """\
# comment line
chr1\t120\t180\tAAA-1\t2
chr1\t150\t350\tAAA-1\t1
chr1\t310\t320\tBBB-1\t1
chr1\t500\t600\tAAA-1\t1
chr1\t1050\t1060\tBBB-1\t3
chr2\t60\t70\tBBB-1\t1
chr2\t60\t70\tCCC-1\t1
"""

BARCODES_TSV = """\
AAA-1
BBB-1
"""

BLACKLIST_BED = """\
chr1\t1000\t1010
"""


@pytest.fixture
def small_files(tmp_path: Path) -> dict[str, Path]:
    """Write the small hand-made dataset to disk."""
    files = {
        "regions": tmp_path / "regions.bed",
        "fragments": tmp_path / "fragments.tsv",
        "barcodes": tmp_path / "barcodes.tsv",
        "blacklist": tmp_path / "blacklist.bed",
    }
    files["regions"].write_text(REGIONS_BED)
    files["fragments"].write_text(FRAGMENTS_TSV)
    files["barcodes"].write_text(BARCODES_TSV)
    files["blacklist"].write_text(BLACKLIST_BED)
    return files


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def topic_model(rng: np.random.Generator):
    """Random region-topic (regions x topics) and cell-topic (topics x cells) model."""
    n_regions, n_topics, n_cells = 400, 8, 120
    region_topic = (
        rng.dirichlet(np.full(n_regions, 0.3), size=n_topics).T.astype(np.float32)
    )
    cell_topic = rng.dirichlet(np.full(n_topics, 0.5), size=n_cells).T.astype(
        np.float32
    )
    region_names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(n_regions)]
    cell_names = [f"cell{i}" for i in range(n_cells)]
    return region_topic, cell_topic, region_names, cell_names

"""Create small bgzipped + tabix indexed fragment files used as test fixtures."""
import os
import sys

import pysam

out_dir = sys.argv[1]
rows = {
    # sample: list of (chrom, start, end, barcode, count)
    "sample1": [("chr1", 100 + i * 10, 150 + i * 10, cb, 1) for i, cb in enumerate(["AAA-1", "BBB-1", "CCC-1"] * 3)],
    "sample2": [("chr1", 200 + i * 10, 260 + i * 10, cb, 1) for i, cb in enumerate(["AAA-1", "BBB-1", "CCC-1"] * 3)]
    + [("chr2", 50, 90, "AAA-1", 2)],
}
for sample, fragments in rows.items():
    tsv = os.path.join(out_dir, f"{sample}.fragments.tsv")
    with open(tsv, "w") as fh:
        for row in sorted(fragments, key=lambda r: (r[0], r[1], r[2])):
            fh.write("\t".join(map(str, row)) + "\n")
    pysam.tabix_index(tsv, preset="bed", force=True)  # writes .gz and .gz.tbi

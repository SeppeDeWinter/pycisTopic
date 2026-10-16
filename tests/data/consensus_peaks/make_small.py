"""Create the small MACS-like summits used as consensus peaks test fixture."""

import numpy as np

rng = np.random.default_rng(11)
chrom_sizes = {"chr1": 60_000, "chr2": 30_000}
with open("chrom.sizes", "w") as fh:
    for c, s in chrom_sizes.items():
        fh.write(f"{c}\t{s}\n")
for sample in ("B_cells", "T_cells", "Mono"):
    rows = []
    for chrom, size in chrom_sizes.items():
        centers = rng.integers(500, size - 500, 12)
        summits = np.concatenate([(centers[:, None] + rng.integers(-600, 600, (12, 4))).ravel(), rng.integers(0, size, 15), [10, size - 10]])
        summits = np.clip(summits, 0, size - 1)
        scores = np.round(rng.gamma(2, 5, len(summits)), 1)
        rows += [(chrom, int(s), int(s) + 1, f"{sample}_{chrom}_{i}", f"{sc:.5f}") for i, (s, sc) in enumerate(zip(summits, scores))]
    rng.shuffle(rows)
    with open(f"{sample}_summits.bed", "w") as fh:
        fh.writelines("\t".join(map(str, r)) + "\n" for r in rows)

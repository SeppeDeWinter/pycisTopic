#!/usr/bin/bash
# Reference consensus peak workflow with the original iterative peak filtering scripts
# (https://github.com/aertslab/iterative_peak_filtering, commit df2ad7c) and
# bedtools v2.30.0, used to create the *.iterative_peak_filtering_scripts.bed files.
# Set PATH (bedtools, gawk) and IT_PK_FIL before running it in this directory.
set -euo pipefail
IT_PK_FIL="${IT_PK_FIL:?Set IT_PK_FIL to the iterative_peak_filtering directory}"
CHROM_SIZES=chrom.sizes
rm -rf bash_out && mkdir -p bash_out
for f in *_summits.bed; do
    name=$(basename "$f" _summits.bed)
    "${IT_PK_FIL}"/calculate_center_peaks_and_extend.sh "$f" bash_out/${name}_summits_extended.bed "${CHROM_SIZES}" 250
    "${IT_PK_FIL}"/iterative_peak_filtering.sh bash_out/${name}_summits_extended.bed bash_out/${name}_summits_extended_filtered.bed "${CHROM_SIZES}"
    "${IT_PK_FIL}"/normalize_macs2_peak_scores.sh bash_out/${name}_summits_extended_filtered.bed bash_out/${name}_summits_extended_filtered_norm.bed
done
cat bash_out/*_summits_extended_filtered_norm.bed | LC_ALL=C sort -k 1,1 -k 2,2n > bash_out/peaks_merged.bed
"${IT_PK_FIL}"/iterative_peak_filtering.sh bash_out/peaks_merged.bed bash_out/_consensus_peaks.bed "${CHROM_SIZES}"
awk -F '\t' 'BEGIN {OFS = FS} {$3 = $3 - 1; print}' bash_out/_consensus_peaks.bed > bash_out/consensus_peaks.bed

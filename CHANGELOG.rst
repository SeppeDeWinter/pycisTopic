Changelog
=========

3.0.0 (unreleased)
------------------

pycisTopic 3 is a major rewrite. See the migration guide in the documentation for how
pycisTopic 2 functions map to pycisTopic 3.

Added
^^^^^

- ``pycistopic`` command line tool covering the whole workflow: ``tss``, ``pseudobulk``,
  ``callpeaks``, ``consensus_peaks``, ``qc``, ``count_matrix``, ``topic_modeling``,
  ``topic_qc``, ``dars``, ``signature_enrichment``, ``gene_activity`` and ``convert``.
- Single AnnData container for topic models (``pycisTopic.adata``): cell-topic
  probabilities in ``.obsm["X_topic"]``, region-topic probabilities in ``.varm["topic"]``
  and model parameters and statistics in ``.uns["pycistopic"]``.
- Consensus peaks without gawk/bedtools (``pycisTopic.consensus_peaks``), identical to
  the iterative peak filtering scripts.
- Pseudobulk fragment files with scATAC fragment tools and peak calling with MACS3
  (``pycisTopic.peak_calling``).
- Fast fragment, QC, count matrix and genomic range operations with polars.
- Gene annotation (TSS) from Ensembl BioMart and chromosome sizes and name mapping from
  UCSC and NCBI (``pycisTopic.gene_annotation``).
- AUCell implementation for signature enrichment and AUCell topic binarization without
  pySCENIC/ctxcore.
- Conversion of pycisTopic 2 ``CistopicObject`` pickles to AnnData
  (``pycistopic convert``).
- Optional ``ruranges`` intersection engine (``pip install pycisTopic[ruranges]``) for
  count matrix creation.
- Parallel LDA topic modeling with collapsed Gibbs sampling in numba
  (``pycistopic topic_modeling lda``, :mod:`pycisTopic.topic_modeling.lda`): same model
  and output files as Mallet, about 10 times faster (10x Multiome PBMC, 30 topics, 500
  iterations: 3.7 minutes instead of 40 minutes) and converges like sequential Gibbs
  sampling (faster than Mallet's parallel sampler); no Java needed.
- Test suite.

Changed
^^^^^^^

- Requires Python 3.12 or newer.
- Topic modeling with the built-in LDA sampler or Mallet (2.1.1, Java 17).
- Imputed accessibility is calculated on the fly instead of being stored.
- DARs: Benjamini-Hochberg correction over all tested regions instead of per chunk of
  regions (changes adjusted p-values and selected DARs).
- Gene activity without pyranges; the search space of minus strand genes is fixed when
  gene boundaries are used.
- Topic QC and topic annotation work on the AnnData container.
- QC is about 40 times faster (10x Multiome PBMC sample with 362 million fragments:
  1.5 minutes instead of 62 minutes): TSS profile and fragments in peaks are
  calculated with numba and the KDE for the QC plots is approximated on a grid for large
  numbers of barcodes.
- The normalized TSS profile per cell barcode is only calculated on request
  (``pycistopic qc run --tss_per_cb``); ``pycistopic qc run`` has no
  ``--intersection_engine`` option anymore.
- AUCell topic binarization is deterministic.

Fixed
^^^^^

- Log-likelihood of topic models (only the last term was used and zero counts were not
  handled correctly; now the same as Mallet's). It is no longer plotted by
  ``plot_stats``, as it is not a model selection metric.
- Reading fragments files with 4 columns.
- Selecting cell barcodes given as a polars Series.
- Detecting the chromosome name source of BED files.
- Error message when fragments and TSS annotation do not overlap.
- Chromosome sizes and alias mapping from UCSC.
- Gene activity: genes with a copy on a contig that is not in the chromosome sizes
  (e.g. hg38 alt contigs; 1,372 genes with the Ensembl hg38 annotation, including
  LILRA4) were missing.
- Barcode rank plot of ``pycistopic qc plot``.
- Barcode rank of barcodes with the same number of fragments is deterministic.
- ``pycistopic qc filter/plot -f`` accept the (decimal) automatic thresholds.

Removed
^^^^^^^

- ``CistopicObject``, ``CistopicImputedFeatures`` and pickled objects (use AnnData).
- Clustering and visualization (``clust_vis``), label transfer, loom export and pyGREAT:
  use scanpy and other dedicated packages.
- In-Python LDA (``run_cgs_models``).
- Dependencies on pyranges, pyscenic, ctxcore, Ray, lda and tmtoolkit.

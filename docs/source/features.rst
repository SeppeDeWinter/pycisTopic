********
Workflow
********

pycisTopic is a Python implementation of cisTopic, which uses Latent Dirichlet
Allocation (LDA) topic modeling to simultaneously cluster cells and co-accessible
regions into regulatory topics. Outside of the `SCENIC+
<https://scenicplus.readthedocs.io/>`_ framework, it can be used to analyze any
scATAC-seq data.

All steps are available as ``pycistopic`` commands (see ``pycistopic --help``) and as
Python functions. The :ref:`tutorial <tutorials>` runs the full workflow on 10x Genomics
PBMC multiome data.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Step
     - Command
     - Python
   * - Gene annotation and chromosome sizes
     - ``pycistopic tss get_tss``, ``pycistopic tss get_ucsc_chrom_sizes_and_alias_mapping``
     - :mod:`pycisTopic.gene_annotation`
   * - Pseudobulk fragment files per cell type
     - ``pycistopic pseudobulk``
     - :func:`pycisTopic.peak_calling.export_pseudobulk`
   * - Peak calling with MACS3
     - ``pycistopic callpeaks``
     - :func:`pycisTopic.peak_calling.call_peaks_macs3`
   * - Consensus peaks
     - ``pycistopic consensus_peaks``
     - :func:`pycisTopic.consensus_peaks.get_consensus_peaks`
   * - QC and cell selection
     - ``pycistopic qc run / filter / plot``
     - :func:`pycisTopic.qc.compute_qc_stats`
   * - Count matrix
     - ``pycistopic count_matrix create / subset``
     - :func:`pycisTopic.count_matrix.create_count_matrix_from_fragment_files`
   * - Topic modeling and model selection
     - ``pycistopic topic_modeling lda run / stats / plot_stats``
     - :func:`pycisTopic.topic_modeling.lda.run_lda_topic_modeling`
   * - AnnData with the topic model
     - ``pycistopic topic_modeling lda create_anndata``
     - :func:`pycisTopic.topic_modeling.create_anndata.create_anndata_from_mallet`
   * - Topic binarization
     - ``pycistopic topic_modeling lda binarize``
     - :func:`pycisTopic.topic_binarization.binarize_topics`
   * - Topic QC and annotation
     - ``pycistopic topic_qc``
     - :func:`pycisTopic.topic_qc.compute_topic_metrics`, :func:`pycisTopic.topic_qc.topic_annotation`
   * - Differentially accessible regions
     - ``pycistopic dars compute_hv_regions / compute_dars``
     - :func:`pycisTopic.diff_features.find_diff_accessible_regions`
   * - Signature enrichment
     - ``pycistopic signature_enrichment``
     - :func:`pycisTopic.signature_enrichment.signature_enrichment`
   * - Gene activity
     - ``pycistopic gene_activity``
     - :func:`pycisTopic.gene_activity.get_gene_activity`
   * - Convert a pycisTopic 2 ``CistopicObject``
     - ``pycistopic convert``
     - :func:`pycisTopic.legacy.cistopic_object_to_anndata`

Clustering, dimensionality reduction (UMAP, t-SNE), batch effect correction (e.g.
Harmony) and label transfer are done with `scanpy <https://scanpy.readthedocs.io/>`_ and
related tools on the cell-topic probabilities (``sc.pp.neighbors(adata,
use_rep="X_topic")``).


Data format
===========

A topic model is stored in a single `AnnData <https://anndata.readthedocs.io/>`_ object
(``.h5ad`` file) with cells as observations and regions as variables, in the same way
as scanpy stores a PCA:

.. code-block:: text

    AnnData  n_obs = cells  x  n_vars = regions
      X              binary accessibility matrix (sparse, optional)
      obs            cell metadata and QC statistics
      var            region coordinates (Chromosome, Start, End)
      obsm["X_topic"]  cell-topic probabilities (cells x topics)
      varm["topic"]    region-topic probabilities (regions x topics)
      uns["pycistopic"]
          version    pycisTopic version
          model      topic model parameters and metrics
          topics     per topic statistics (assignments, coherence, Gini index,
                     annotation, ...)

Helper functions to access the topic model are available in :mod:`pycisTopic.adata`.

Imputed accessibility (cell-topic probabilities x topic-region probabilities) is never
stored: it is calculated on the fly (in chunks) when needed, e.g. for DARs, signature
enrichment and gene activity.


Steps
=====

Consensus peaks
---------------
Peaks are called per cell type, as using pseudobulk profiles improves the signal
compared to calling peaks on all cells together, especially for rare cell types. Cell
type labels can come from, for example, the RNA modality of multiome data or a
preliminary clustering of scATAC-seq data on predefined regions (e.g. SCREEN).

#. ``pycistopic pseudobulk`` creates fragment files per cell type with
   `scATAC fragment tools <https://github.com/aertslab/scatac_fragment_tools>`_.
#. ``pycistopic callpeaks`` calls peaks on each pseudobulk with MACS3
   (``--format FRAG --max-count 1 --nomodel --shift 73 --extsize 146 --nolambda
   --call-summits``).
#. ``pycistopic consensus_peaks`` derives consensus peaks with the iterative peak
   filtering approach of Corces et al. (2018). Summits are extended by
   ``peak_half_width`` (default: 250 bp) in both directions and overlapping peaks are
   filtered iteratively per cell type. Within each group of overlapping peaks:

   #. 1 peak: the peak is kept.
   #. 2 peaks: the peak with the highest score is kept.
   #. 3 or more peaks: the peak with the highest score is kept and all peaks
      overlapping it are removed. This is repeated with the next most significant
      remaining peak until all peaks are processed.

   Peak scores are then normalized per cell type (score per million) and the peaks of
   all cell types are filtered iteratively again. Optionally, peaks overlapping
   blacklisted regions are removed.

QC and cell selection
---------------------
``pycistopic qc run`` computes QC statistics per sample and per cell barcode:

* **Barcode rank plot**: number of unique fragments per barcode. A steep drop-off
  ('knee') indicates a good separation between cells and empty droplets.
* **Insert size distribution**: in a good library, there is a peak below 100 bp
  (nucleosome free) and peaks at ~200 bp (mono-nucleosome) and multiples of it.
* **TSS enrichment**: aggregated cut sites around transcription start sites, normalized
  to the flanks. High signal at the TSS indicates good signal to noise.
* **Fraction of fragments in peaks (FRiP)** and **duplication rate**.

The normalized TSS profile per cell barcode is only written with ``--tss_per_cb``, as
this file is large.

``pycistopic qc filter`` selects cell barcodes based on the number of unique fragments
in peaks, TSS enrichment and FRiP, with thresholds set automatically (Otsu's method) or
by the user.

Count matrix
------------
``pycistopic count_matrix create`` creates a binary (or count) region x cell matrix from
the fragment files of multiple samples, the consensus peaks and the selected cell
barcodes. Cell barcodes are named ``BARCODE___SAMPLE``.

Topic modeling and model selection
----------------------------------
Topic models with different numbers of topics are trained with latent Dirichlet
allocation (LDA) and collapsed Gibbs sampling by ``pycistopic topic_modeling lda run``,
directly on the binary count matrix. The sampler is parallel: cells and regions are
divided in as many groups as threads and in each step, every thread samples the
tokens of a different (cell group, region group) block, so cell-topic and
region-topic counts are always up-to-date. It converges like sequential Gibbs
sampling and is about 10 times faster than Mallet (which is still available with
``pycistopic topic_modeling mallet``; both produce the same output files, so the
``stats``, ``plot_stats``, ``create_anndata`` and ``binarize`` subcommands work for
both).

As the optimal number of topics is not known beforehand,
``pycistopic topic_modeling lda stats`` computes model selection metrics:

* **Mimno_2011**: average coherence of the top topics (Mimno et al., 2011). Higher is
  better.
* **Arun_2010**: density-based metric (Arun et al., 2010). Lower is better.
* **Cao_Juan_2009**: divergence-based metric (Cao Juan et al., 2009). Lower is better.

``pycistopic topic_modeling lda plot_stats`` plots the scaled metrics (Arun_2010 and
Cao_Juan_2009 inverted, so higher is better for all). The metrics often do not agree:
choose a number of topics for which the metrics level off and, most importantly, the
cell types or states of interest are captured by separate topics. The exact number of
topics is not critical.

The joint log likelihood of the model (``loglikelihood`` in ``*.stats.json``, as
reported by Mallet) is not a model selection metric: with the default
``alpha = 50 / n_topics``, it usually decreases with more topics.

Topic binarization and QC
-------------------------
For motif enrichment analysis (e.g. with `pycistarget
<https://pycistarget.readthedocs.io/>`_) and other downstream analyses, topics are
converted to region sets. Available methods: ``otsu`` (Otsu, 1979), ``yen`` (Yen et al.,
1995), ``li`` (Li & Lee, 1993), ``aucell`` (Van de Sande et al., 2020) and ``ntop`` (top
n regions per topic). Otsu and Yen work well for topic-region distributions, ``ntop``
gives region sets of equal size and ``aucell`` is recommended for cell-topic
distributions.

``pycistopic topic_qc`` computes per topic:

* the **number of assignments** and the **marginal topic distribution**: how much each
  topic contributes to the model,
* the **topic coherence** (Mimno et al., 2011): whether the top regions of a topic are
  co-accessible in the data,
* the **Gini index**: specificity of the topic across cells (0: general, 1: specific),

and optionally annotates topics with a cell annotation: a topic is assigned to the cell
groups that are enriched in its binarized cells, and marked as *general* when it is
present in many more cells than those groups.

Differentially accessible regions
---------------------------------
Differentially accessible regions (DARs) are found on the imputed accessibility of
highly variable regions with a Wilcoxon rank-sum test between each group of cells and
the other cells (or user-defined contrasts), with Benjamini-Hochberg correction over all
tested regions. By default, regions with an adjusted p-value < 0.05 and log2 fold
change > log2(1.5) are reported.

Signature enrichment
--------------------
Region sets (e.g. ChIP-seq peaks, DARs or binarized topics) are mapped to the regions in
the dataset. Regions are ranked per cell by imputed accessibility and the enrichment of
each region set is scored with AUCell. By default, the top 5% of the ranked regions is
used to calculate the AUC (``--auc-threshold``); increase it for small region sets.

Gene activity
-------------
Gene activity summarizes the imputed accessibility of the regions around a gene:

* **Search space**: regions up to a minimum and maximum distance upstream and
  downstream of the gene, by default stopping at the promoters of other genes (or at
  predefined boundaries, e.g. TADs).
* **Distance weight**: exponentially decaying with the distance to the gene
  (``decay_rate``), with maximal weight in the gene body (extended by
  ``extend_gene_body_upstream`` and ``extend_gene_body_downstream``).
* **Gene size weight** (optional): the median gene size divided by the size of the gene,
  as large genes have more regions by chance.
* **Gini weight**: regions with cell type specific accessibility get a higher weight.

The weighted imputed accessibility of the regions is averaged per gene.

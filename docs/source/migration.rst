.. _migration:

***************************
Migrating from pycisTopic 2
***************************

pycisTopic 3 is a rewrite focused on speed, memory usage, reproducibility and an easy
installation:

* All steps are available as ``pycistopic`` commands, which read and write standard
  files (fragments, BED, Matrix Market, Parquet, h5ad).
* Data is processed with `polars <https://pola.rs/>`_, numpy and numba instead of
  pandas, pyranges and Ray.
* The ``CistopicObject`` is replaced by a single AnnData file, which works directly
  with scanpy.
* Imputed accessibility is calculated on the fly, instead of being stored in a
  ``CistopicImputedFeatures`` object.


Installation
============

* pycisTopic 3 requires **Python 3.12** or newer.
* Topic modeling is done with a new built-in parallel LDA sampler
  (``pycistopic topic_modeling lda``), about 10 times faster than Mallet, or with
  **Mallet** (2.1.1, Java 17), which is installed separately (see
  :ref:`installation`). The previous in-Python LDA implementation
  (``run_cgs_models``) is removed.
* MACS3 and scATAC fragment tools are dependencies of pycisTopic; pyranges, pyscenic,
  ctxcore, Ray, lda, tmtoolkit and pyGREAT are no longer needed.


From CistopicObject to AnnData
==============================

A ``CistopicObject`` with a selected model maps to the pycisTopic AnnData format as
follows (see :mod:`pycisTopic.adata`):

.. list-table::
   :header-rows: 1

   * - pycisTopic 2
     - pycisTopic 3
   * - ``cistopic_obj.cell_names`` / ``region_names``
     - ``adata.obs_names`` / ``adata.var_names``
   * - ``cistopic_obj.binary_matrix`` (regions x cells)
     - ``adata.X`` (cells x regions, optional)
   * - ``cistopic_obj.cell_data`` / ``region_data``
     - ``adata.obs`` / ``adata.var``
   * - ``selected_model.cell_topic`` (topics x cells)
     - ``adata.obsm["X_topic"]`` (cells x topics), :func:`~pycisTopic.adata.get_cell_topic`
   * - ``selected_model.topic_region`` (regions x topics)
     - ``adata.varm["topic"]``, :func:`~pycisTopic.adata.get_region_topic`
   * - ``selected_model.parameters`` / ``metrics``
     - ``adata.uns["pycistopic"]["model"]``
   * - ``selected_model.topic_qc_metrics``, coherence, assignments
     - ``adata.uns["pycistopic"]["topics"]``, :func:`~pycisTopic.adata.get_topics`
   * - ``cistopic_obj.projections["cell"]["UMAP"]``
     - ``adata.obsm["X_umap"]``

Existing pickled ``CistopicObject`` files can be converted (the old pycisTopic code is
not needed for this)::

    pycistopic convert -i cistopic_obj.pkl -o cistopic_obj.h5ad --store_matrix

or in Python with :func:`pycisTopic.legacy.read_legacy_pickle` and
:func:`pycisTopic.legacy.cistopic_object_to_anndata`.


Functions
=========

.. list-table::
   :header-rows: 1

   * - pycisTopic 2
     - pycisTopic 3
   * - ``export_pseudobulk``
     - ``pycistopic pseudobulk``, :func:`~pycisTopic.peak_calling.export_pseudobulk`
   * - ``peak_calling`` (MACS2)
     - ``pycistopic callpeaks``, :func:`~pycisTopic.peak_calling.call_peaks_macs3` (MACS3)
   * - ``get_consensus_peaks``
     - ``pycistopic consensus_peaks``, :func:`~pycisTopic.consensus_peaks.get_consensus_peaks`
   * - ``compute_qc_stats``, ``plot_sample_metrics``, ``plot_barcode_metrics``
     - ``pycistopic qc run / plot / filter``, :func:`~pycisTopic.qc.compute_qc_stats`
   * - ``create_cistopic_object_from_fragments``
     - ``pycistopic count_matrix create``
   * - ``run_cgs_models``, ``run_cgs_models_mallet``, ``evaluate_models``
     - ``pycistopic topic_modeling lda run / stats / plot_stats`` (or ``mallet``)
   * - ``add_LDA_model``
     - ``pycistopic topic_modeling lda create_anndata``
   * - ``run_umap``, ``run_tsne``, ``find_clusters``, ``harmony``, ``plot_metadata``,
       ``plot_topic``, ``cell_topic_heatmap``
     - scanpy, e.g. ``sc.pp.neighbors(adata, use_rep="X_topic")``, ``sc.tl.umap``,
       ``sc.tl.leiden``, ``sc.pl.umap``
   * - ``binarize_topics``
     - ``pycistopic topic_modeling lda binarize``,
       :func:`~pycisTopic.topic_binarization.binarize_topics`
   * - ``compute_topic_metrics``, ``plot_topic_qc``, ``topic_annotation``
     - ``pycistopic topic_qc``, :mod:`pycisTopic.topic_qc` (take an AnnData object)
   * - ``impute_accessibility``, ``normalize_scores``
     - :func:`~pycisTopic.imputed_accessibility.impute_accessibility_chunked`
       (not stored)
   * - ``find_highly_variable_features``, ``find_diff_features``
     - ``pycistopic dars compute_hv_regions / compute_dars``,
       :mod:`pycisTopic.diff_features`
   * - ``get_gene_activity``
     - ``pycistopic gene_activity``, :func:`~pycisTopic.gene_activity.get_gene_activity`
   * - ``signature_enrichment``
     - ``pycistopic signature_enrichment``,
       :func:`~pycisTopic.signature_enrichment.signature_enrichment`


Removed features
================

* **Clustering and visualization** (``clust_vis``): use scanpy on
  ``adata.obsm["X_topic"]``.
* **Label transfer**: use dedicated packages (e.g. scanpy ``ingest``, scANVI) on the
  gene activity AnnData.
* **Loom export** (SCope): use the h5ad file.
* **pyGREAT**.
* **In-Python LDA with the lda package** (``run_cgs_models``): use
  ``pycistopic topic_modeling lda``.
* **Stored imputed accessibility** (``CistopicImputedFeatures``): it is faster to
  calculate it on the fly.


Changes in results
==================

Some bugs of pycisTopic 2 were fixed, which changes results:

* **DARs**: the Benjamini-Hochberg correction is now applied over all tested regions.
  Before, it was applied per chunk of 20,000 regions, so with more regions adjusted
  p-values were too low and more DARs were reported.
* **Model log-likelihood**: only the last term was used before (and zero counts were
  not handled correctly). The log likelihood reported by
  ``pycistopic topic_modeling lda stats`` is now the joint log likelihood of the model
  (as reported by Mallet), which usually decreases with more topics. It is no longer
  shown in the model selection plot.
* **Gene activity**: the search space of genes on the minus strand was wrong when gene
  boundaries were used (upstream and downstream were swapped by pyranges). Gene
  activity of minus strand genes changes.
* **AUCell topic binarization** is deterministic (same thresholds as pySCENIC, without a
  random Gaussian mixture initialization).
* **QC**: the density (KDE) used to color the QC scatter plots is approximated on a
  grid for large numbers of barcodes (< 1% difference). The normalized TSS profile per
  cell barcode is only calculated on request (``pycistopic qc run --tss_per_cb``).
* **Consensus peaks** follow the `iterative peak filtering scripts
  <https://github.com/aertslab/iterative_peak_filtering>`_ on MACS3 summits; peak start
  coordinates can differ by 1 bp from pycisTopic 2 (peaks are 500 bp wide in both).

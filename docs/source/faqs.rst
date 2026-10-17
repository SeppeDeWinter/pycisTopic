****
FAQs
****

I have a pycisTopic 2 analysis (``CistopicObject``), can I continue with pycisTopic 3?
======================================================================================

Yes. Convert the pickled object to the pycisTopic AnnData format::

    pycistopic convert -i cistopic_obj.pkl -o cistopic_obj.h5ad --store_matrix

The selected topic model, cell and region data and projections (e.g. UMAP) are kept.
See :ref:`migration` for how pycisTopic 2 functions map to pycisTopic 3.


I have a cisTopic analysis in R, can I use it in pycisTopic?
============================================================

Yes, there is no need to repeat the topic modeling. Export the cell-topic and
region-topic probabilities in R::

    library(arrow)
    cisTopic_obj <- readRDS(PATH_TO_YOUR_CISTOPICOBJECT)
    cell_topic <- as.data.frame(t(modelMatSelection(cisTopic_obj, "cell", "Probability")))
    region_topic <- as.data.frame(t(modelMatSelection(cisTopic_obj, "region", "Probability", all.regions = TRUE)))
    cell_topic$cell <- rownames(cell_topic)
    region_topic$region <- rownames(region_topic)
    write_feather(cell_topic, "cell_topic.feather")
    write_feather(region_topic, "region_topic.feather")

and create a pycisTopic AnnData object in Python (region names are in
``chrom:start-end`` format)::

    import polars as pl
    from pycisTopic.adata import create_anndata

    cell_topic = pl.read_ipc("cell_topic.feather")
    region_topic = pl.read_ipc("region_topic.feather")
    adata = create_anndata(
        cell_topic=cell_topic.drop("cell").to_numpy(),
        region_topic=region_topic.drop("region").to_numpy(),
        cell_barcodes=cell_topic["cell"].to_list(),
        region_ids=region_topic["region"].to_list(),
    )
    adata.write_h5ad("cistopic.h5ad")


I processed my data with another tool (e.g. Signac/ArchR), can I use pycisTopic?
================================================================================

Yes. ``pycistopic topic_modeling lda run`` takes a binary
region x cell matrix in Matrix Market format (``PREFIX.matrix.mtx``) with the cell
barcodes and region IDs (``chrom:start-end``) in ``PREFIX.cell_barcodes.tsv`` and
``PREFIX.region_ids.tsv``, so you can start topic modeling from a count matrix created
with another tool (binarize the counts first)::

    import scipy.io
    import scipy.sparse

    binary_matrix = scipy.sparse.csr_matrix(counts_regions_x_cells > 0, dtype="int32")
    scipy.io.mmwrite("matrix.matrix.mtx", binary_matrix)

We recommend to use consensus peaks called on pseudobulks per cell type (see the
tutorial).


How do I correct for batch effects?
===================================

Cell-topic probabilities can be corrected with Harmony, as implemented in scanpy::

    import scanpy as sc

    sc.external.pp.harmony_integrate(
        adata, key="sample_id", basis="X_topic", adjusted_basis="X_topic_harmony"
    )
    sc.pp.neighbors(adata, use_rep="X_topic_harmony")

Topic modeling itself is not affected by the correction, so topics, DARs and gene
activity are calculated on the uncorrected model.


Mallet fails with ``UnsupportedClassVersionError`` or runs out of memory
========================================================================

Only relevant when using Mallet instead of ``pycistopic topic_modeling lda``.

* ``UnsupportedClassVersionError``: Mallet 2.1 needs Java 17 or newer (check with
  ``java -version``).
* ``OutOfMemoryError``: increase the memory available to Mallet with ``-m`` (in GB) of
  the ``pycistopic topic_modeling mallet`` commands.


Signature enrichment gives (almost) zero AUC values
===================================================

AUCell only considers the top ranked regions of each cell (5% by default). With few
regions in the dataset or small signatures, increase ``--auc-threshold`` of
``pycistopic signature_enrichment``. Also check that the signature uses the same
chromosome names (e.g. ``chr1`` or ``1``) and genome assembly as the regions.

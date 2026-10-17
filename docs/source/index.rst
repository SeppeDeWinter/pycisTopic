Welcome to the documentation of pycisTopic!
===========================================

pycisTopic is a Python package to simultaneously identify cell states and
cis-regulatory topics from single cell epigenomics data, using Latent Dirichlet
Allocation (LDA) topic modeling. It is part of the `SCENIC+
<https://scenicplus.readthedocs.io/>`_ workflow, but can be used to analyze any
scATAC-seq data.

pycisTopic covers the full scATAC-seq workflow, from fragment files to topics:

* consensus peaks from pseudobulk profiles (MACS3 and iterative peak filtering),
* QC and cell selection,
* fast parallel topic modeling (LDA) and model selection,
* topic binarization, QC and annotation,
* differentially accessible regions, signature enrichment and gene activity.

All steps are available as ``pycistopic`` commands and as Python functions. Results are
stored as AnnData, so clustering and visualization can be done with `scanpy
<https://scanpy.readthedocs.io/>`_.

.. note::

    This is the documentation of pycisTopic 3, which is not compatible with pycisTopic 2.
    See :ref:`migration` to update existing analyses.

.. toctree::
   :hidden:

   installation
   features
   tutorials
   migration
   changelog
   faqs
   api

Reference
---------

Bravo González-Blas, C., De Winter, S., et al. SCENIC+: single-cell multiomic inference
of enhancers and gene regulatory networks. Nat Methods 20, 1355–1367 (2023).
https://doi.org/10.1038/s41592-023-01938-4

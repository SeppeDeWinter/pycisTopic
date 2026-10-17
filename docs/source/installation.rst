.. _installation:

************
Installation
************

pycisTopic requires **Python 3.12 or newer**.

Install pycisTopic
==================

With `uv <https://docs.astral.sh/uv/>`_::

    git clone https://github.com/aertslab/pycisTopic.git
    cd pycisTopic
    uv venv --python 3.12
    uv pip install ".[plot]"

Or with ``pip`` in an environment of your choice (e.g. ``conda``)::

    conda create --name pycistopic python=3.12 -y
    conda activate pycistopic
    git clone https://github.com/aertslab/pycisTopic.git
    cd pycisTopic
    pip install ".[plot]"

This installs the ``pycistopic`` command line tool and the ``pycisTopic`` Python
package, including `MACS3 <https://github.com/macs3-project/MACS>`_ (peak calling) and
`scATAC fragment tools <https://github.com/aertslab/scatac_fragment_tools>`_
(pseudobulk fragment files).

.. note::

    MACS3 is distributed as source code on PyPI, so it is compiled during installation.
    This needs a C compiler (e.g. ``gcc``; ``conda install -c conda-forge compilers``
    when you do not have one).

Optional dependencies (extras):

* ``plot``: matplotlib and adjustText, needed for QC and topic model plots.
* ``ruranges``: faster intersection engine for count matrix creation
  (``--intersection_engine ruranges``).
* ``docs``: tools to build this documentation.
* ``development``: tools for development (pytest, ruff, mypy).

The tutorial additionally uses `scanpy <https://scanpy.readthedocs.io/>`_ (with
``leidenalg``) for clustering and visualization::

    uv pip install scanpy leidenalg

Install Mallet (optional)
=========================

Topic modeling is done with the built-in parallel LDA sampler
(``pycistopic topic_modeling lda``), which needs no extra software. Alternatively,
topic models can be trained with `Mallet <https://mimno.github.io/Mallet/>`_ (same
model, about 10 times slower), which needs **Java 17 or newer**. Download and unpack
the latest Mallet release (2.1.1) from
https://github.com/mimno/Mallet/releases/tag/v2.1.1::

    wget https://github.com/mimno/Mallet/releases/download/v2.1.1/mallet-2.1.1-bin.zip
    unzip mallet-2.1.1-bin.zip

    # Check that Java (17 or newer) and Mallet work.
    java -version
    mallet-2.1.1/bin/mallet

Pass the path to the ``mallet`` executable (``mallet-2.1.1/bin/mallet``) to the
``pycistopic topic_modeling mallet`` commands with ``-b``.

.. tip::

    If ``java -version`` reports a version older than 17, install a newer Java
    (e.g. ``conda install -c conda-forge openjdk=17`` or a module on your HPC system).

Check the installation
======================

::

    pycistopic --help

or in Python::

    import pycisTopic
    pycisTopic.__version__

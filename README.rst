.. image:: https://zenodo.org/badge/329905726.svg
   :target: https://zenodo.org/badge/latestdoi/329905726

pycisTopic
==========

pycisTopic is a Python module to simultaneously identify cell states and cis-regulatory topics from single cell epigenomics data.

Installation
************

pycisTopic requires Python 3.12 or newer. Install it, for example with `uv`_::

	git clone https://github.com/aertslab/pycisTopic.git
	cd pycisTopic
	uv venv --python 3.12
	uv pip install ".[plot]"

or with ``pip`` in a ``conda`` environment::

	conda create --name pycistopic python=3.12 -y
	conda activate pycistopic
	git clone https://github.com/aertslab/pycisTopic.git
	cd pycisTopic
	pip install ".[plot]"

MACS3 is installed as a dependency and compiled during installation (a C compiler is
needed).

Topic modeling is done with the built-in parallel LDA sampler
(``pycistopic topic_modeling lda``). Optionally, `Mallet`_ (2.1.1, needs Java 17 or
newer) can be used instead; it is installed separately::

	wget https://github.com/mimno/Mallet/releases/download/v2.1.1/mallet-2.1.1-bin.zip
	unzip mallet-2.1.1-bin.zip

.. _uv: https://docs.astral.sh/uv/
.. _Mallet: https://github.com/mimno/Mallet/releases/tag/v2.1.1

You can also install `SCENIC+`_, which includes pycisTopic.

.. _SCENIC+: https://github.com/aertslab/scenicplus

Check version
*************

To check your pycisTopic version::

	pycistopic --help

	import pycisTopic
	pycisTopic.__version__

Tutorials & documentation
*************************

Tutorial and documentation are available at https://pycistopic.readthedocs.io/.

pycisTopic 3 is not compatible with pycisTopic 2. See the migration guide in the
documentation and ``CHANGELOG.rst`` for the changes.

Questions?
**********

* If you have **technical questions or problems**, such as bug reports or ideas for new features, please open an issue under the issues tab.
* If you have **questions about the interpretation of results or your analysis**, please start a Discussion under the Discussions tab.


Reference
*********

`Bravo González-Blas, C., De Winter, S., et al. SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. Nat Methods 20, 1355–1367 (2023). <https://doi.org/10.1038/s41592-023-01938-4>`_

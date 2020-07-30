==================================================
maxsmooth: Derivative Constrained Function Fitting
==================================================

Introduction
------------

:maxsmooth: Derivative Constrained Function Fitting
:Author: Harry Thomas Jones Bevins
:Version: 1.1.0
:Homepage: https://github.com/htjb/maxsmooth
:Documentation: https://maxsmooth.readthedocs.io/

.. image:: https://travis-ci.com/htjb/maxsmooth.svg?branch=master
   :target: https://travis-ci.com/htjb/maxsmooth
   :alt: Build Status
.. image:: https://codecov.io/gh/htjb/maxsmooth/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/htjb/maxsmooth
   :alt: Test Coverage Status
.. image:: https://readthedocs.org/projects/maxsmooth/badge/?version=latest
   :target: https://maxsmooth.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/htjb/maxsmooth/blob/master/LICENSE
   :alt: License information
.. image:: https://pypip.in/v/maxsmooth/badge.png
   :target: https://pypi.org/project/maxsmooth/#description
   :alt: Latest PyPI version
.. image:: https://pypip.in/d/maxsmooth/badge.png
   :alt: Number of PyPI downloads

Derivative Constrained Functions and ``maxsmooth``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``maxsmooth`` is an open source software for fitting derivative constrained
functions, DCFs such as Maximally Smooth Functions
, MSFs to data sets. MSFs are functions for which there are no zero
crossings in derivatives of order m >= 2 within the domain of interest.
They are designed to prevent the loss of
signals when fitting out dominant smooth foregrounds or large magnitude signals that
mask signals of interest. Here "smooth" means that the foregrounds follow power
law structures and do not feature turning points in the band of interest.
In some cases DCFs can be used to
highlight systematics in the data. More generally for DCFs the minimum
constrained derivative order, m can take on any value or a set of
specific high order derivatives can be constrained.

``maxsmooth`` uses quadratic programming implemented with ``CVXOPT`` to fit
data subject to a fixed linear constraint, Ga <= 0, where the product
Ga is a matrix of derivatives.
The constraint on an MSF are not explicitly
linear and each constrained derivative can be positive or negative.
``maxsmooth`` is, however, designed to test the <= 0 constraint multiplied
by a positive or negative sign. Where a positive sign in front of the m\ :sup:`th`
order derivative forces the derivative
to be negative for all x. For an N\ :sup:`th` order polynomial ``maxsmooth`` can test
every available sign combination but by default it implements a sign navigating algorithm.
This is detailed in the ``maxsmooth`` paper (see citation), is summarized
below and in the software documentation.

The available sign combinations act as discrete parameter spaces all with
global minima and ``maxsmooth`` is capable of finding the minimum of these global
minima by implementing a cascading algorithm which is followed by a directional
exploration. The cascading routine typically finds an approximate to the global
minimum and then the directional exploration is a complete search
of the sign combinations in the neighbourhood
of that minimum. The searched region is limited by factors
that encapsulate enough of the neighbourhood to confidently return the global minimum.

The sign navigating method is reliant on the problem being "well defined" but this
is not always the case and it is in these instances it is possible to run the code testing
every available sign combination on the constrained derivatives. For a definition of
a "well defined" problem and it's counter part see the ``maxsmooth`` paper and the
documentation.

``maxsmooth`` features a built in library of DCFs or
allows the user to define their own. The addition of possible inflection points
and zero crossings in higher order derivatives is also available to the user.
The software has been designed with these two
applications in mind and is a simple interface.

Example Fit
~~~~~~~~~~~

Shown below is an example MSF fit performed with ``maxsmooth`` to data that
follows a y = x\ :sup:`-2.5` power law with a randomly generated Gaussian
noise with a standard deviation 0.02. The top panel shows the data and the
bottom panel shows the residual
after subtraction of the MSF fit. The software using one of the built in DCF models
and fitting normalised data is shown to be capable of recovering the
random noise.

.. image:: https://github.com/htjb/maxsmooth/raw/master/docs/images/README.png
  :width: 400
  :align: center

Installation
~~~~~~~~~~~~

The software can be pip installed from the PYPI repository like so,

.. code::

  pip install maxsmooth

or alternatively it can be installed from the git repository via,

.. code::

  git clone https://github.com/htjb/maxsmooth.git
  cd maxsmooth
  python setup.py install --user

Licence and Citation
~~~~~~~~~~~~~~~~~~~~

The software is free to use on the MIT open source license. However if you use
the software for academic purposes we request that you cite the ``maxsmooth``
paper.

  H. T. J. Bevins et al., `maxsmooth: Rapid maximally smooth function fitting with
  applications in Global 21-cm cosmology <https://arxiv.org/abs/2007.14970>`__,
  arXiv e-print, arXiv:2007.14970, 2020.

Below is the BibTex citation,

.. code:: bibtex

  @ARTICLE{maxsmooth,
       author = {{Bevins}, H.~T.~J. and {Handley}, W.~J. and {Fialkov}, A. and
         {de Lera Acedo}, E. and {Greenhill}, L.~J. and {Price}, D.~C.},
        title = "{maxsmooth: Rapid maximally smooth function fitting with applications in Global 21-cm cosmology}",
      journal = {arXiv e-prints},
         year = 2020,
        month = jul,
          eid = {arXiv:2007.14970},
        pages = {arXiv:2007.14970},
  archivePrefix = {arXiv},
       eprint = {2007.14970},
  primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200714970B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }

Documentation
~~~~~~~~~~~~~
The documentation is available at: https://maxsmooth.readthedocs.io/

Alternatively, it can be compiled locally from the git repository and requires
`sphinx <https://pypi.org/project/Sphinx/>`__ to be installed.
You can do this via:

.. code::

  cd docs/
  make SOURCEDIR=source html

or

.. code::

  cd docs/
  make SOURCEDIR=source latexpdf

The resultant docs can be found in the docs/_build/html/ and docs/_build/latex/
respectively.

Requirements
~~~~~~~~~~~~

To run the code you will need the following additional packages:

- `matplotlib <https://pypi.org/project/matplotlib/>`__
- `numpy <https://pypi.org/project/numpy/>`__
- `CVXOPT <https://pypi.org/project/cvxopt/>`__
- `scipy <https://pypi.org/project/scipy/>`__
- `progressbar <https://pypi.org/project/progressbar/>`__

To compile the documentation locally you will need:

- `sphinx <https://pypi.org/project/Sphinx/>`__
- `numpydoc <https://pypi.org/project/numpydoc/>`__

To run the test suit you will need:

- `pytest <https://pypi.org/project/pytest/>`__

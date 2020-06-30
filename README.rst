============================================
maxsmooth: maximally smooth function fitting
============================================

Introduction
------------

:maxsmooth: maximally smooth function fitting
:Author: Harry Thomas Jones Bevins
:Version: 1.0.0
:Homepage: https://github.com/htjb/maxsmooth

``maxsmooth`` is an open source software for fitting derivative constrained
functions, DCFs such as Maximally Smooth Functions
, MSFs to data sets. MSFs are functions for which there are no zero
crossings in derivatives of order :math:`m \geq 2` within the domain of interest.
They are designed to prevent the loss of
signals when fitting out dominant foregrounds and in some cases can be used to
highlight systematics left in the data. More generally for DCFs the minimum
constrained derivative order, m can take on any value or a set of
specific high order derivatives can be constrained.

You can read more about MSFs here ..

``maxsmooth`` uses quadratic programming implemented with ``CVXOPT`` to fit
data subject to a linear constraint. The constraint on an MSF can be codefied
like so,

.. math::

  \frac{d^m~y}{d~x^m}~\geq~0~~\textnormal{or}~~\frac{d^m~y}{d~x^m}~\leq~0.

This constraint is itself not linear but ``maxsmooth`` is designed to test the
constraint,

.. math::

  \pm \frac{d^m~y}{d~x^m}~\leq~0

where a positive sign infront of the :math:`m^{th}` order derivative forces the derivative
to be negative for all x. For an :math:`N^{th}` order polynomial ``maxsmooth`` can test
every available sign combination but by default it implements a 'sign-smapling'
algorithm. This is detailed in the ``maxsmooth`` paper (see citation) but is summarised
below.

The available sign combinations act as discrete parameter spaces all with
global minima and ``maxsmooth`` is capable of finding the minimum of these global
minima by implementing a descent algorithm which is followed by a directional
exploration. The descent routine typically finds an approximate to the global
minimum and then the directional exploration is a complete search
of the sign combinations in the neighbourhood
of that minimum. The searched region is limited by factors
that encapsulate enough of the neighbourhood to confidently return the global minimum.

The sign sampling method is reliant on the problem being 'well defined' but this
is not always the case and it is in these instances possible to run the code testing
every available sign combination on the constrained derivatives. For a definition of
a 'well defined' problem and it's counter part see the ``maxsmooth`` paper.

``maxsmooth`` features a built in library of DCFs or
allows the user to define their own. The addition of possible inflection points
and zero crossings in higher order derivatives is also available to the user.
The software has been designed with these two
applications in mind and is a simple interface.

Instalation
~~~~~~~~~~~

Documentation
~~~~~~~~~~~~~
The documentation can be compiled from the git repository by...

Dependencies
~~~~~~~~~~~~

Basic requirements:

- Python version..
- `matplotlib <https://pypi.org/project/matplotlib/>`__
- `numpy <https://pypi.org/project/numpy/>`__
- `CVXOPT <https://pypi.org/project/cvxopt/>`__

Citation
~~~~~~~~

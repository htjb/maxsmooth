============================================
maxsmooth: maximally smooth function fitting
============================================

Introduction
------------

:maxsmooth: maximally smooth function fitting
:Author: Harry Thomas Jones Bevins
:Version: 0.0.0
:Homepage: https://github.com/htjb/maxsmooth

``maxsmooth`` is an open source software for fitting maximally smooth functions
,hearafter MSFs, to data sets. MSFs are functions for which there are no
inflection points or, in other words, the high order derivatives do not cross
zero within the domain of interest. They are designed to prevent the loss of
signals when fitting out dominant foregrounds and in some cases can be used to
highlight systematics left in the data.

You can read more about MSFs here ..

``maxsmooth`` uses quadratic programming implemented with ``cvxopt`` to fit
data subject to a linear constraint. The constraint on an MSF can be codefied
like so,

.. math::

  \frac{d^m~y}{d~x^m}~>~0~~\textnormal{or}~~\frac{d^m~y}{d~x^m}~<~0.

This constraint is itself not linear but ``maxsmooth`` is designed to test the
constraint,

.. math::

  \pm \frac{d^m~y}{d~x^m}~<~0

where a positive sign infront of the :math:`m^{th}` order derivative forces the derivative
to be negative for all x. For an :math:`N^{th}` order polynomial ``maxsmooth`` tests
every combination of possible signs in front of the derivatives with :math:`m~>2` for
:math:`N~<=~10`. For :math:`N~>~10` a smaller subset of the 'sign-space' is
tested to reduce runtime but sufficiently large subset to return an accurate
fit.

``maxsmooth`` features a built in library of maximally smooth functions or
allows the user to define their own. The addition of possible inflection points
is also available to the user. The software has been designed with these two
applications in mind and is a simple interface.

Instalation
~~~~~~~~~~~

Dependencies
~~~~~~~~~~~~

Basic requirements:

- Python version..
- `pylab <https://pypi.org/project/pylab/>`__
- `numpy <https://pypi.org/project/numpy/>`__
- `cvxopt <https://pypi.org/project/cvxopt/>`__

Citation
~~~~~~~~

.. toctree::
   :maxdepth: 6

``maxsmooth`` Theory and Algorithm
----------------------------------

.. include:: theory.rst

``maxsmooth`` Example Codes
---------------------------

This section is designed to introduce the user to the software and the form
in which it is run. It provides basic examples of data fitting with a built in
MSF model and a user defined model.

There are also examples of functions that can be used pre-fitting and post-fitting
for various purposes including; determination of the best DCF model from the
built in library for the problem being fitted, analysis of the :math:`{\chi^2}`
distribution as a function of the discrete sign spaces and analysis of the
parameter space surrounding the optimum results.

The data used for all of this examples is available
`here <https://github.com/htjb/maxsmooth/tree/master/example_codes/Data>`__.

The example codes can be found
`here <https://github.com/htjb/maxsmooth/tree/master/example_codes>`__ and
corresponding Jupyter Notebooks are provided
`here <https://mybinder.org/v2/gh/htjb/maxsmooth/master?filepath=example_notebooks%2F>`__.

Simple Example code
~~~~~~~~~~~~~~~~~~~
.. include:: simple_program.rst

Turning Points and Inflection Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: turning_points.rst

New Basis Example
~~~~~~~~~~~~~~~~~

.. include:: new_basis_example.rst

Best Basis Example
~~~~~~~~~~~~~~~~~~

.. include:: best_basis.rst

:math:`{\chi^2}` Distribution Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: chi_dist_example.rst

Parameter Plotter Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: param_plotter_example.rst

``maxsmooth`` Functions
-----------------------

This section details the specifics of the built in functions in ``maxsmooth`` including
the relevant keyword arguments and default parameters for all. Where keyword arguments
are essential for the functions to run this is stated.

smooth()
~~~~~~~~

.. automodule:: maxsmooth.DCF
   :members: smooth

best_basis()
~~~~~~~~~~~~

.. automodule:: maxsmooth.best_basis
  :members: basis_test

chidist_plotter()
~~~~~~~~~~~~~~~~~

.. automodule:: maxsmooth.chidist_plotter
   :members: chi_plotter

parameter_plotter()
~~~~~~~~~~~~~~~~~~~

.. automodule:: maxsmooth.parameter_plotter
  :members: param_plotter

Change Log
----------

.. include:: CHANGELOG.rst

This function can be used to identify which of the built in DCFs
fits the data best before running joint fits.

To use it we begin by loading in the data,

.. code::

  import numpy as np

  x = np.load('Data/x.npy')
  y = np.load('Data/y.npy')

and then importing the basis_test() function.

.. code::

  from maxsmooth.best_basis import basis_test

To call the function we use,

.. code::

  basis_test(x, y, base_dir='examples/', N=np.arange(3, 16, 1))

The function only requires the data but we can provide it with a base directory,
fit type and range of DCF orders to test. By defualt it uses the sign navigating
algorithm and tests :math:`{N = 3 - 13}`. Here we test the range
:math:``{N = 3 - 15}``.
The resultant graph is saved in the
base directory and the example generated here is shown below.

.. image:: https://github.com/htjb/maxsmooth/raw/master/docs/images/Basis_functions.png
  :width: 400
  :align: center

The graph shows us which basis is the optimum for solving this problem from the
built in library (that which can reach the minimum :math:``{\chi^2}``). If we
were to go to higher N we would also find that the :math:``{\chi^2}`` value
would stop decreasing in value. The value of N for which this occurs at is the
optimum DCF order. (See the ``maxsmooth`` paper for a real world application
of this concept.)

We can also provide this function with additional arguments such as the
fit type, minimum constrained derivative, directional exploration limits
ect. (see the ``maxsmooth`` Functions section).

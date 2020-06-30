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

  basis_test(x, y, base_dir='examples/')

The function only requires the data but we can provide it with a base directory,
fit type and range of DCF orders to test. By defualt it uses the sign sampling
algorithm and tests :math:`{N = 3 - 13}`. The resultant graph is saved in the
base directory and the example generated here is shown below.

.. image:: /images/Basis_functions.png
  :width: 400
  :align: center

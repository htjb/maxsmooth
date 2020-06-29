.. highlight:: python

In order to run the ``maxsmooth`` software using the built
in DCF models for a simple fit the user can follow the simple structure detailed here.

The user should begin by importing the `smooth` class from `maxsmooth.DCF`.

.. code::

    from maxsmooth.DCF import smooth

The user should then import the data they wish to fit.

.. code::

    import numpy as np

    x = np.load('Data/x.npy')
    y = np.load('Data/y.npy')

and define the polynomial orders they wish to fit.

.. code::

    N = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    for i in range(len(N)):
        `act on N[i]`

or for example,

.. code::

    N = 10

`smooth` can be called like so,

.. code::

    result = smooth(x, y, N, **kwargs)

where the kwargs are detailed below. It's resulting attributes can be accessed by writing
:code:`result.attribute_name`. For example printing the outputs is done like
so,

.. code::

    print('Objective Funtion Evaluations:\n', result.optimum_chi)
    print('RMS:\n', result.rms)
    print('Parameters:\n', result.optimum_params)
    print('Fitted y:\n', result.y_fit)
    print('Sign Combinations:\n', result.optimum_signs)
    print('Derivatives:\n', result.derivatives)

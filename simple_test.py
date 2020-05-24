"""
This section is designed to introduce the user to the software and the form
in which it is run. In order to run the `maxsmooth` software using the built
in MSFs the user can follow the simple structure detailed here.

The user should begin by importing the `smooth` class from `maxsmooth.DCP`.

.. code:: bash
    from maxsmooth.msf import smooth

"""
from maxsmooth.DCP import smooth

"""
The user should then import the data they wish to fit.

.. code:: bash
    import numpy as np

    x = np.load('Data/x.npy')
    y = np.load('Data/y.npy')

and define the polynomial orders they wish to fit.

.. code:: bash

    N = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    for i in range(len(N)):
        `act on N[i]`

or for example,

.. code:: bash

    N = 10

"""
import numpy as np

x = np.load('Data/x.npy')
y = np.load('Data/y.npy')

N = 10

"""
`smooth` can be called like so,

.. code:: bash

    result = smooth(x, y, N, **kwargs)

and it's resulting attributes can be accessed by writing
:code:`result.attribute_name`. For example printing the outputs is done like
so,

.. code:: bash

    print('Objective Funtion Evaluations:\n', result.Optimum_chi)
    print('RMS:\n', result.rms)
    print('Parameters:\n', result.Optimum_params)
    print('Fitted y:\n', result.y_fit)
    print('Sign Combinations:\n', result.Optimum_signs)
    print('Derivatives:\n', result.derivatives)

"""

result = smooth(x, y, N, model_type='normalised_polynomial')
print('Accessing Fit Attributes:')
print('Objective Funtion Evaluations:\n', result.Optimum_chi)
print('RMS:\n', result.rms)
#print('Parameters:\n', result.Optimum_params)
#print('Fitted y:\n', result.y_fit)
print('Sign Combinations:\n', result.Optimum_signs)
#print('Derivatives:\n', result.derivatives)

import matplotlib.pyplot as plt

plt.plot(x, y - result.y_fit)
plt.xlabel('x', fontsize=12)
plt.ylabel(r'$\delta y$', fontsize=12)
plt.tight_layout()
plt.show()

"""
This section is designed to introduce the user to the software and the form
in which it is run. In order to run the `maxsmooth` software using the built
in DCFs the user can follow the simple structure detailed here.

An important point to make is that by default ``maxsmooth`` fits a
Maximally Smooth Function or MSF to the data. An MSF, as stated in
the introduction to the documentation, is a function which has
derivatives of order :math:`{m \geq 2}` constrained so that they do not cross
0. This means that they do not have inflection points or non smooth
structure produced by higher order derivatives. More generally a DCF
follows the constraint,

.. math:

    \frac{\delta^m y}{\delta x^m} \leq 0 ~~\mathrm{or}~~ \frac{\delta^m y}{\delta x^m} \geq 0 $

for every constrained order :math:`{m}`. The set of :math:`{m}` can be any set of
derivative orders as long as those derivatives exist for the function.

This means we can use ``maxsmooth`` to produce different DCF
models. MSFs are one of two special cases of DCF and we can also
have a Completely Smooth Function (CSF) with orders :math:`{m \geq 1}`
constrained. Alternatively we can have Partially Smooth Functions
(PSF) which are much more general and can have arbitrary sets of
derivatives constrained. We illustrate how this is implemented
towards the end of this example but we begin with the default case
fitting a MSF.

The user should begin by importing the `smooth` class from `maxsmooth.DCF`.

.. code:: bash
    from maxsmooth.DCF import smooth

"""
from maxsmooth.DCF import smooth

"""
The user should then import the data they wish to fit.

.. code:: bash

    import numpy as np

    x = np.load('Data/x.npy')
    y = np.load('Data/y.npy') + np.random.normal(0, 0.02, len(x))

and define the polynomial orders they wish to fit.

.. code:: bash

    N = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    for i in range(len(N)):
        `act on N[i]`

or for example,

.. code:: bash

    N = 15

We can also plot the data to illustrate what is happening.
Here the data is a scaled :math:`{x^{-2.5}}` power law and I have added gaussian
noise in with a standard deviation of 0.02.

.. code:: bash

    import matplotlib.pyplot as plt

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

"""
import numpy as np

x = np.load('Data/x.npy')
y = np.load('Data/y.npy') + np.random.normal(0, 0.02, len(x))

N = 15

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('../docs/images/simple_program_data.png')
plt.show()

"""
`smooth` can be called as is shown below. It takes the x and y data as standard
inputs as well as the order of the fit. There are a set of keyword arguments
also available that change the type of function being fitted and these are
detailed in the documentation.

.. code:: bash

    result = smooth(x, y, N)

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

    plt.plot(x, y - result.y_fit)
    plt.xlabel('x', fontsize=12)
    plt.ylabel(r'$\delta y$', fontsize=12)
    plt.tight_layout()
    plt.show()

"""

result = smooth(x, y, N)

print('Accessing Fit Attributes:')
print('Objective Funtion Evaluations:\n', result.optimum_chi)
print('RMS:\n', result.rms)
#print('Parameters:\n', result.Optimum_params)
#print('Fitted y:\n', result.y_fit)
print('Sign Combinations:\n', result.optimum_signs)
#print('Derivatives:\n', result.derivatives)

plt.plot(x, y - result.y_fit)
plt.xlabel('x', fontsize=12)
plt.ylabel(r'$\delta y$', fontsize=12)
plt.tight_layout()
plt.savefig('../docs/images/simple_program_msf_residuals.png')
plt.show()

"""
To fit the data with a CSF we can use the 'constraints' keyword
argument in smooth(). 'constraints' sets the minimum constrained
derivative for the function which for a CSF we want to be one.

.. code:: bash

    res = smooth(
        x, y, N, constraints=1)
"""

res = smooth(
    x, y, N, constraints=1)

"""
Note in the printed results the number of constrained derivatives has
increased by 1 and the only derivative that is allowed to cross through 0
(Zero Crossings Used?) is the the :math:`{0^{th}}` order i.e. the data.

.. code:: bash

    plt.plot(x, y - res.y_fit)
    plt.xlabel('x', fontsize=12)
    plt.ylabel(r'$\delta y$', fontsize=12)
    plt.tight_layout()
    plt.show()
"""

plt.plot(x, y - res.y_fit)
plt.xlabel('x', fontsize=12)
plt.ylabel(r'$\delta y$', fontsize=12)
plt.tight_layout()
plt.savefig('../docs/images/simple_program_csf_residuals.png')
plt.show()

"""
A Partially Smooth Function can have derivatives constrained via :math:`{m \geq a}`
where :math:`{a}` is
any order above 2 or it can have a set of derivatives that are allowed to cross
zero. For the first case we can once again use the 'constraints' keyword
argument. For example we can constrain derivatives with orders :math:`{\geq 3}` which will
allow the :math:`{1^{st}}` and :math:`{2^{nd}}` order derivatives to cross zero.
This is useful when our
data features an inflection point we want to model with our fit.

.. code:: bash

    res = smooth(x, y, N, constraints=3)

    plt.plot(x, y - res.y_fit)
    plt.xlabel('x', fontsize=12)
    plt.ylabel(r'$\delta y$', fontsize=12)
    plt.tight_layout()
    plt.show()

"""

res = smooth(x, y, N, constraints=3)

plt.plot(x, y - res.y_fit)
plt.xlabel('x', fontsize=12)
plt.ylabel(r'$\delta y$', fontsize=12)
plt.tight_layout()
plt.savefig('../docs/images/simple_program_psf1_residuals.png')
plt.show()

"""
To allow a particular set of derivatives to cross zero we use the
'zero_crossings' keyword. In the example below we are lifting the constraints
on the :math:`{3^{rd}}`, :math:`{4^{th}}` and :math:`{5^{th}}` order derivatives
but our minimum constrained derivative is still set at the default 2. Therefore
this PSF has derivatives of order :math:`{m = [2, 6, 7, 8, 9]}`
constrained via the condition at the begining of this example code.

.. code::

    res = smooth(x, y, N, zero_crossings=[3, 4, 5])

    plt.plot(x, y - res.y_fit)
    plt.xlabel('x', fontsize=12)
    plt.ylabel(r'$\delta y$', fontsize=12)
    plt.tight_layout()
    plt.show()
"""

res = smooth(x, y, N, zero_crossings=[3, 4, 5])

plt.plot(x, y - res.y_fit)
plt.xlabel('x', fontsize=12)
plt.ylabel(r'$\delta y$', fontsize=12)
plt.tight_layout()
plt.savefig('../docs/images/simple_program_psf2_residuals.png')
plt.show()

"""
While PSFs can seem like an attractive way to improve the quality of fit they
are less 'smooth' than a MSF or CSF and consequently they can introduce
additional turning points in to your residuals obscuring any signals of
intrest.
"""

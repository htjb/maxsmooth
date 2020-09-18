"""
This example will walk the user through implementing DCF fits to data sets with
turning points and inflection points. It builds on the details in the
'Simple Example Code' and uses the 'constraints' keyword argument introduced
there. The 'constraints' keyword argument is used to adjust the type of DCF that
is being fitted. Recall that by default ``maxsmooth`` implements a Maximally
Smooth Function or MSF with constraints=2 i.e. derivatives of order :math:`{m \geq 2}`
constrained so that they do not cross zero. This allows for turning points in the
DCF as illustrated below.

We start by generating some noisy data that we know will include a turning point
and defining the order of the DCF we would like to fit.

.. code:: bash

    import numpy as np

    x = np.linspace(-10, 10, 100)
    noise = np.random.normal(0, 0.02, 100)
    y = x**(2) + noise

    N = 10

"""
import numpy as np

x = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 0.02, 100)
y = x**(2) + noise

N = 10

"""

We can go ahead and plot this data just to double check it features a turning
point.

.. code:: bash

    import matplotlib.pyplot as plt

    plt.plot(x, y)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.show()

"""

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()

"""
As already stated ``maxsmooth`` does not constrain the first derivative of the
DCF by deafult so we can go ahead and fit the data.

.. code:: bash

    from maxsmooth.DCF import smooth

    res = smooth(x, y, N)

"""

from maxsmooth.DCF import smooth

res = smooth(x, y, N)

"""
If we than plot the resultant residuals we will see that despite the data
having a turning point present we have recovered the Gaussian noise.

.. code:: bash

    plt.plot(x, y- res.y_fit, label='Recovered Noise')
    plt.plot(x, noise, label='Actual Noise')
    plt.ylabel(r'$\delta y$', fontsize=12)
    plt.xlabel('x', fontsize=12)
    plt.legend()
    plt.show()

"""

plt.plot(x, y- res.y_fit, label='Recovered Noise')
plt.plot(x, noise, label='Actual Noise')
plt.ylabel(r'$\delta y$', fontsize=12)
plt.xlabel('x', fontsize=12)
plt.legend()
plt.show()

"""
To illustrate what happens when there is an inflection point in the data we can
define some sinusoidal data as so.

.. code:: bash

    x = np.linspace(1, 5, 100)
    noise = np.random.normal(0, 0.02, 100)
    y = np.sin(x) + noise

    N = 10

    plt.plot(x, y)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.show()

"""
x = np.linspace(1, 5, 100)
noise = np.random.normal(0, 0.02, 100)
y = np.sin(x) + noise

N = 10

plt.plot(x, y)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()

"""
If we proceed to fit this with smooth() in its default settings we will get a
poor fit as by default the second derivative is constrained. We need to lift this
constraint to allow for the prominant inflection point to be modelled. We do this
by setting the keyword argument constraints=3 creating a Partially Smooth Function
or PSF.

.. code:: bash

    res_msf = smooth(x, y, N)
    res_psf = smooth(x, y, N, constraints=3)

    plt.plot(x, y, label='Data')
    plt.plot(x, res_msf.y_fit, label=r'MSF fit, $m \geq 2$')
    plt.plot(x, res_psf.y_fit, label=r'PSF fit, $m \geq 3$')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    plt.show()

"""

res_msf = smooth(x, y, N)
res_psf = smooth(x, y, N, constraints=3)

plt.plot(x, y, label='Data')
plt.plot(x, res_msf.y_fit, label=r'MSF fit, $m \geq 2$')
plt.plot(x, res_psf.y_fit, label=r'PSF fit, $m \geq 3$')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.show()

"""
Finally, we can plot the residuals to further see that by lifting the constraint on the
second derivative we have allowed an inflection point in the data.

.. code::

    plt.plot(x, y- res_psf.y_fit, label='Recovered Noise')
    plt.plot(x, noise, label='Actual Noise')
    plt.ylabel(r'$\delta y$', fontsize=12)
    plt.xlabel('x', fontsize=12)
    plt.legend()
    plt.show()

"""

plt.plot(x, y - res_psf.y_fit, label='Recovered Noise')
plt.plot(x, noise, label='Actual Noise')
plt.ylabel(r'$\delta y$', fontsize=12)
plt.xlabel('x', fontsize=12)
plt.legend()
plt.show()

"""
This example code illustrates how to define your own basis function for the
DCF model.
It implements a modified version of the built in normalised polynomial model
but the structure is the same for more elaborate models.

As always we need to import the data, define an order :math:`{N}`
and import the function fitting routine, smooth().
"""

import numpy as np
from maxsmooth.DCF import smooth

x = np.load('Data/x.npy')
y = np.load('Data/y.npy')

N=10

"""
There are several requirements needed to define a new basis function completely
for ``maxsmooth`` to be able to fit it. They are as summarised below and then
examples of each are given in more detail,

    * **args:** Additional non-standard  arguments needed in the definition of the
    basis. The standard arguments are the data (x and y), the order of the fit N,
    the pivot point about which a model can be fit,
    the derivative order :math:`{m}` and the params. While the
    pivot point is not strictly needed it is a required argument for the
    functions defining a new basis to help the user in their definition.

    * **basis_functions:** This function defines the basis of the DCF model,
    :math:`{\phi}` where the model can be generally defined as,

    .. math::

        y = \sum_{k = 0}^N a_k \phi_k(x)

    where :math:`{a_k}` are the fit parameters.

    * **model:** This is the function described by the equation above.

    * **derivative:** This function defines the :math:`{m^{th}}` order derivative.

    * **derivative_pre:** This function defines the prefactors,
    :math:`{\mathbf{G}}` on the derivatives where ``CVXOPT``, the quadratic
    programming routine used, evaluates the constraints as,

    .. math:

        \mathbf{Ga} \leq \mathbf{h}

    where :math:`{\mathbf{a}}` is the matrix of parameters and :math:`{\mathbf{h}}`
    is the matrix of constraint limits. For more details on this see the ``maxsmooth``
    paper.


We can begin defining our new basis function by defining the aditional arguments
needed to fit the model as a list,
"""
arguments = [x[-1]*10, y[-1]*10]

"""
The next step is to define the basis functions :math:`{\phi}`. This needs to be
done in a function that has the arguments *(x, y, pivot_point, N, \*args)*. 'args'
is optional but since we need them for this basis we are passing it in.

The basis functions, :math:`{\phi}`, should be an array of dimensions len(x)
by N and consequently evaluated at each N and x data point as shown below.
"""

def basis_functions(x, y, pivot_point, N, *args):

    phi = np.empty([len(x), N])
    for h in range(len(x)):
        for i in range(N):
            phi[h, i] = args[1]*(x[h]/args[0])**i

    return phi

"""
We can define the model that we are fitting in a function like that shown below.
This is used for evaluating :math:`{\chi^2}` and returning the optimum fitted model
once the code has finished running. It requires the arguments
*(x, y, pivot_point, N, params, \*args)* in that order and again where 'args' is optional.
'params' is the parameters of the fit, :math:`{\mathbf{a}}` which should have length
:math:`{N}`.

The function should return the fitted estimate of y.
"""

def model(x, y, pivot_point, N, params, *args):

    y_sum = args[1]*np.sum([
        params[i]*(x/args[0])**i
        for i in range(N)], axis=0)

    return y_sum

"""
Next we have to define a function for the derivatives of the model which
takes arguments *(m, x, y, N, pivot_point, params, *args)* where :math:`{m}` is
the derivative order. The function should return the :math:`{m^{th}}` order
derivative evaluation and is used for checking that the constraints have been
met and returning the derivatives of the optimum fit to the user.
"""

def derivative(m, x, y, N, pivot_point, params, *args):

    mth_order_derivative = []
    for i in range(N):
        if i <= m - 1:
            mth_order_derivative.append([0]*len(x))
    for i in range(N - m):
            mth_order_derivative_term = args[1]*np.math.factorial(m+i) / \
                np.math.factorial(i) * \
                params[int(m)+i]*(x)**i / \
                (args[0])**(i + 1)
            mth_order_derivative.append(
                mth_order_derivative_term)

    return mth_order_derivative

"""
Finally we have to define :math:`{\mathbf{G}}` which is used by ``CVXOPT`` to
build the derivatives and constrain the functions. It takes arguments
*(m, x, y, N, pivot_point, \*args)* and should return the prefactor on the
:math:`{m^{th}}` order derivative. For a more thorough definition of the
prefactor on the derivative and an explination of how the problem is
constrained in quadratic programming see the ``maxsmooth`` paper.
"""

def derivative_pre(m, x, y, N, pivot_point, *args):

    mth_order_derivative = []
    for i in range(N):
        if i <= m - 1:
            mth_order_derivative.append([0]*len(x))
    for i in range(N - m):
            mth_order_derivative_term = args[1]*np.math.factorial(m+i) / \
                np.math.factorial(i) * \
                (x)**i / \
                (args[0])**(i + 1)
            mth_order_derivative.append(
                mth_order_derivative_term)

    return mth_order_derivative

"""
With our functions and additional arguments defined we can pass these
to the ``maxsmooth`` smooth() function as is shown below. This overwrites the
built in DCF model but you are still able to modify the fit type i.e. testing all
available sign combinations or sampling them.
"""

result = smooth(x, y, N,
    basis_functions=basis_functions, model=model,
    derivatives=derivative, der_pres=derivative_pre, args=arguments)

"""
The output of the fit can be accessed as before,
"""

print('Objective Funtion Evaluations:\n', result.optimum_chi)
print('RMS:\n', result.rms)
print('Parameters:\n', result.optimum_params[2])
print('Fitted y:\n', result.y_fit)
print('Sign Combinations:\n', result.optimum_signs)
print('Derivatives:\n', result.derivatives)

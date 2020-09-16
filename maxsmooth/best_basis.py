"""
As demonstrated, this function allows you to test the built in basis and their
ability to
fit the data. It produces a plot that shows chi squared as a function of
:math:`{N}` for the 7 built in models and saves the figure to the base
directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from maxsmooth.DCF import smooth


class basis_test(object):

    r"""
    **Parameters:**

        x: **numpy.array**
            | The x data points for the set being fitted.

        y: **numpy.array**
            | The y data points for fitting.

    **Kwargs:**

        fit_type: **Default = 'qp-sign_flipping'**
            | This kwarg allows the user to switch between sampling the
                available discrete sign spaces (default)
                or testing all sign combinations on the derivatives which can
                be accessed by setting to 'qp'.

        base_dir: **Default = 'Fitted_Output/'**
            | The location of the outputted
                graph from function. This must be a string and end in '/'. If
                the file does not exist then the function will create it.

        **N: Default = [3, .., 13] in steps of 1 else list or numpy array**
        **of integers**
            | The DCF orders to test each basis function with. In
                some instances the basis function may fail for a given
                :math:`{N}` and higher orders due to overflow/underflow
                errors or ``CVXOPT`` errors.

        **pivot_point: Default = len(x)//2 otherwise an integer between**
        **-len(x) and len(x)**
            | Some of the built in
                models rely on pivot points in the data sets which by defualt
                is set as the middle index. This can be altered via
                this kwarg which can occasionally lead to a better quality fit.

        **constraints: Default = 2 else an integer less than or equal**
        **to N - 1**
            | The minimum constrained derivative order which is set by default
                to 2 for a Maximally Smooth Function.

        zero_crossings: **Default = None else list of integers**
            | Allows you to
                specify if the conditions should be relaxed on any
                of the derivatives between constraints and the highest order
                derivative. e.g. a 6th order fit with just a constrained 2nd
                and 3rd order derivative would have zero_crossings = [4, 5].

        cap: **Default = (len(available_signs)//N) + N else an integer**
            | Determines the maximum number of signs explored either side of
                the minimum :math:`{\chi^2}` value found after the decent
                algorithm has terminated.

        chi_squared_limit: **Default = 2 else float or int**
            | The prefactor on the maximum allowed increase in :math:`{\chi^2}`
                during the directional exploration which is defaulted at 2.
                If this value multiplied by the minimum :math:`{\chi^2}`
                value found after the descent algorithm is exceeded then the
                exploration in one direction is stopped and started in the
                other. For more details on this and 'cap' see the ``maxsmooth``
                paper.

        cvxopt_maxiter: **Default = 10000 else integer**
            | This shouldn't need
                changing for most problems however if ``CVXOPT`` fails with a
                'maxiters reached' error message this can be increased.
                Doing so arbitrarily will however increase the run time of
                ``maxsmooth``.

    """

    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y

        for keys, values in kwargs.items():
            if keys not in set([
                    'fit_type', 'base_dir', 'N', 'pivot_point',
                    'constraints', 'zero_crossings', 'chi_squared_limit',
                    'cap', 'cvxopt_maxiter']):
                raise KeyError(
                    "Unexpected keyword argument in basis_test().")

        self.fit_type = kwargs.pop('fit_type', 'qp-sign_flipping')
        if self.fit_type not in set(['qp', 'qp-sign_flipping']):
            raise KeyError(
                "Invalid 'fit_type'. Valid entries include 'qp'\n"
                + "'qp-sign_flipping'")

        self.base_dir = kwargs.pop('base_dir', 'Fitted_Output/')
        if type(self.base_dir) is not str:
            raise KeyError("'base_dir' must be a string ending in '/'.")
        elif self.base_dir.endswith('/') is False:
            raise KeyError("'base_dir' must end in '/'.")

        self.N = kwargs.pop('N', np.arange(3, 14, 1))
        for i in range(len(self.N)):
            if self.N[i] % 1 != 0:
                raise ValueError(
                    'N must be an integer or whole number float.')

        self.pivot_point = kwargs.pop('pivot_point', len(self.x)//2)
        if type(self.pivot_point) is not int:
            raise TypeError('Pivot point is not an integer index.')
        elif self.pivot_point >= len(self.x) or \
                self.pivot_point < -len(self.x):
            raise ValueError(
                'Pivot point must be in the range -len(x) - len(x).')

        self.constraints = kwargs.pop('constraints', 2)
        if type(self.constraints) is not int:
            raise TypeError("'constraints' is not an integer")
        if type(self.N) is list:
            self.N = np.array(self.N)
        if self.constraints >= self.N.min() and \
                self.constraints < self.N.max():
            self.N = np.arange(self.constraints + 1, self.N.max()+1, 1)
        elif self.constraints >= self.N.max():
            raise ValueError(
                "'constraints' exceeds the number of derivatives" +
                " for highest value N provided to the function." +
                " Lower constraints or increase the range of N being" +
                " tested.")

        self.zero_crossings = kwargs.pop('zero_crossings', None)
        if self.zero_crossings is not None:
            for i in range(len(self.zero_crossings)):
                if type(self.zero_crossings[i]) is not int:
                    raise TypeError(
                        "Entries in 'zero_crossings'" +
                        " are not integer.")
                if self.zero_crossings[i] < self.constraints:
                    raise ValueError(
                        'One or more specified derivatives for' +
                        ' zero crossings is less than the minimum' +
                        ' constrained' +
                        ' derivative.\n zero_crossings = ' +
                        str(self.zero_crossings)
                        + '\n' + ' Minimum Constrained Derivative = '
                        + str(self.constraints))

        self.chi_squared_limit = kwargs.pop('chi_squared_limit', None)
        self.cap = kwargs.pop('cap', None)
        if self.chi_squared_limit is not None:
            if isinstance(self.chi_squared_limit, int) or \
                    isinstance(self.chi_squared_limit, float):
                pass
            else:
                raise TypeError(
                    "Limit on maximum allowed increase in chi squared" +
                    ", 'chi_squared_limit', is not an integer or float.")
        if self.cap is not None:
            if type(self.cap) is not int:
                raise TypeError(
                    "The cap on directional exploration" +
                    ", 'cap', is not an integer.")

        self.cvxopt_maxiter = kwargs.pop('cvxopt_maxiter', 10000)
        if type(self.cvxopt_maxiter) is not int:
            raise ValueError("'cvxopt_maxiter' is not integer.")

        self.test()

    def test(self):

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        def fit(model_type):

            chi = []
            N_passed = []
            for i in range(len(self.N)):
                try:

                    result = smooth(
                        self.x, self.y, self.N[i],
                        model_type=model_type,
                        fit_type=self.fit_type, pivot_point=self.pivot_point,
                        constraints=self.constraints,
                        cvxopt_maxiter=self.cvxopt_maxiter,
                        zero_crossings=self.zero_crossings,
                        cap=self.cap, chi_squared_limit=self.chi_squared_limit)

                    if model_type == 'loglog_polynomial':
                        chi.append(np.sum((self.y - result.y_fit)**2))
                    else:
                        chi.append(result.optimum_chi)
                    N_passed.append(self.N[i])
                except Exception:
                    print(
                        'Unable to fit with N = ' + str(self.N[i]) + ' and ' +
                        str(model_type) + '.')

            if chi != []:
                chi = np.array(chi)
                min_chi = chi.min()
                for i in range(len(chi)):
                    if chi[i] == chi.min():
                        best_N = self.N[i]
            else:
                chi = np.nan
                min_chi = np.nan
                N_passed = np.nan
                best_N = np.nan

            return chi, min_chi, N_passed, best_N

        chi_poly, min_chi_poly, N_poly, best_N_poly = fit('polynomial')
        chi_np, min_chi_np, N_np, best_N_np = fit('normalised_polynomial')
        chi_log, min_chi_log, N_log, best_N_log = fit('log_polynomial')
        chi_loglog, min_chi_loglog, N_loglog, best_N_loglog = \
            fit('loglog_polynomial')
        chi_leg, min_chi_leg, N_leg, best_N_leg = fit('legendre')
        chi_exp, min_chi_exp, N_exp, best_N_exp = fit('exponential')
        chi_dif, min_chi_dif, N_dif, best_N_dif = fit('difference_polynomial')

        plt.figure()
        plt.plot(N_np, chi_np, label='Normalised Polynomial', c='b')
        plt.plot(N_poly, chi_poly, label='Polynomial', c='k')
        plt.plot(N_dif, chi_dif, label='Difference Polynomial', c='r')
        plt.plot(N_log, chi_log, label=r'Log Polynomial', c='g')
        plt.plot(N_loglog, chi_loglog, label=r'Log Log Polynomial', c='orange')
        plt.plot(N_leg, chi_leg, label='Legendre', c='purple')
        plt.plot(N_exp, chi_exp, label='Exponential', c='magenta')
        plt.legend()
        plt.xlabel(r'N')
        plt.xticks(self.N)
        plt.ylabel(r'$\chi^2$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(self.base_dir + 'Basis_functions.pdf')
        plt.close()

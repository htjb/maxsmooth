"""
As demonstrated, this function allows you to test the built in basis and their
ability to
fit the data. It produces a plot that shows :math:`{\chi^2}` as a function of
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
    """

    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y

        for keys, values in kwargs.items():
            if keys not in set(['fit_type', 'base_dir', 'N']):
                raise KeyError("Unexpected keyword argument in basis_test().")

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
                        fit_type=self.fit_type)

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

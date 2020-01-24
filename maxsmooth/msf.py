
"""
*smooth* is used to call the fitting routine by the user.

"""

from maxsmooth.qp import qp_class
from maxsmooth.Models import Models_class
from maxsmooth.derivatives import derivative_class
from maxsmooth.Data_save import save, save_optimum
from itertools import product
import numpy as np
import pylab as pl
import warnings
import time
import os
import sys


class smooth(object):

    r"""

    **Parameters:**

        x: *numpy.array* The x data points for the set being fitted.

        y: *numpy.array* The y data points for fitting.

        N: *list* The number of terms in the MSF polynomial function.

        setting: *class atributes* The settings determined by
        `maxsmooth.settings.setting` and called before smooth().

    **Kwargs:**

        initial_params: *list of length N* Allows the user to overwrite the
            the default initial parameters which are a list of length N given
            by,

            .. code:: bash

                params0 = [(y[-1]-y[0])/2]*(self.N)

            or equivalently in log-space for the 'logarithmic_polynomial'
            model_type(see Settings),

            .. code:: bash

                params0 = [(np.log10(y[-1])-np.log10(y[0]))/2] *
                    (self.N)

        The following Kwargs can be used by the user to define thier own basis
        function. **Further details on the structures of the following matrix
        and functions can be found in the section `Designing A Basis Function`.
        **

        data_matrix: *CVXOPT dense matrix of dimensions (len(y),1)* The data
            matrix is a matrix of y values to be fitted by cvxopt.
            The default data matrix used by *smooth* is,

            .. code:: bash

                b = matrix(y, (len(y), 1), 'd').

            See CVXOPT documentation for details on building a dense matrix.
            This will only need to be changed on rare occasions when the
            fitting space is changed. For example smooth will automatically
            adjust this matrix to,

            .. code:: bash

                b = matrix(np.log10(y), (len(y), 1), 'd'),

            when model_type is set to 'logarithmic_polynomial' (see settings).

        basis_function: *function with parameters (x, y, mid_point, N)* This is
            a function of basis functions for the quadratic programming.
            The variable mid_point is the index at the middle of the datasets
            x and y.

        model: *function with parameters (x, y, mid_point, N, params)* This is
            a user defined function describing the model to be fitted to the
            data.

        der_pres: *function with parameters (m, i, x, y, mid_point)*
            This function describes the prefactors on the ith term of the mth
            order derivative used in defining the constraint.

        derivatives: *function with parameters (m, i, x, y, mid_point, params)*
            User defined function describing the ith term of the mth
            order derivative used to check that conditions are being met.

        args: *list* of extra arguments for `smooth` to pass to the functions
            detailed above.

    **Output**

        If N is a list with length greater than 1 then the outputs from smooth
        are lists and arrays with dimension 0 equal to len(N).

        y_fit:
            *numpy.array* The fitted arrays of y data from `smooth`.

        Optimum_chi:
            *numpy.array* The optimum chi squared values for the fit calculated
            by,

            .. math::

                {X^2=\sum(y-y_{fit})^2}.

        Optimum_params:
            *numpy.array* The set of parameters corresponding to the optimum
            fits.

        rms:
            *list* The rms value of the residuals :math:`{y_{res}=y-y_{fit}}`
            calculated by,

            .. math::

                {rms=\sqrt{\frac{\sum(y-y_{fit})^2}{n}}}

            where :math:`n` is the number of data points.

        derivatives:
            *numpy.array* The :math:`m^{th}` order derivatives.

        Optimum_signs:
            *numpy.array* The sign combinations corresponding to the
            optimal results.

    """

    def __init__(self, x, y, N, setting, **kwargs):
        self.x = x
        self.y = y
        self.N = N
        self.fit_type, self.base_dir, self.model_type, self.filtering, \
            self.all_output, self.cvxopt_maxiter, self.ifp, \
            self.ifp_list, self.data_save, self.ud_initial_params, \
            self.warnings = setting.fit_type, \
            setting.base_dir, setting.model_type, setting.filtering, \
            setting.all_output, setting.cvxopt_maxiter, setting.ifp, \
            setting.ifp_list, setting.data_save, setting.ud_initial_params, \
            setting.warnings

        if ('initial_params' in kwargs):
            self.initial_params = kwargs['initial_params']
        else:
            self.initial_params = None

        if ('basis_functions' in kwargs):
            self.basis_functions = kwargs['basis_functions']
        else:
            self.basis_functions = None

        if ('data_matrix' in kwargs):
            self.data_matrix = kwargs['data_matrix']
            if self.warnings is True:
                warnings.warn('Data matrix has been changed.', stacklevel=2)
        else:
            self.data_matrix = None

        if ('der_pres' in kwargs):
            self.der_pres = kwargs['der_pres']
        else:
            self.der_pres = None

        if ('model' in kwargs):
            self.model = kwargs['model']
        else:
            self.model = None

        if ('derivatives' in kwargs):
            self.derivatives_function = kwargs['derivatives']
        else:
            self.derivatives_function = None

        if ('args' in kwargs):
            self.args = kwargs['args']
        else:
            self.args = None

        self.basis_change = [
            self.basis_functions, self.der_pres,
            self.derivatives_function, self.model]
        if np.any(self.basis_change) is None:
            if np.any(self.basis_change) is not None:
                print(
                    'Error: Attempt to change basis functions failed.' +
                    ' One or more functions not defined.' +
                    ' Please consult documentation.')
                sys.exit(1)

        if np.all(self.basis_change) is not None:
            self.model_type = 'User Defined'
            if self.data_matrix is None:
                if self.warnings is True:
                    warnings.warn('Warning: Data matrix unchanged.')

        self.y_fit, self.Optimum_signs, self.Optimum_params, self.derivatives,\
            self.Optimum_chi, self.rms = self.fitting()

    def fitting(self):

        def signs_array(nums):
            return np.array(list(product(*((x, -x) for x in nums))))

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        def qp(x, y, N, mid_point):
            print(
                '######################################################' +
                '#######')
            start = time.time()
            if self.data_save is True:
                if not os.path.exists(
                    self.base_dir + 'MSF_Order_' + str(N) + '_'
                        + self.fit_type + '/'):
                    os.mkdir(
                        self.base_dir + 'MSF_Order_' + str(N) + '_' +
                        self.fit_type + '/')

            signs = signs_array([1]*(N-2))

            params, chi_squared, pass_fail, passed_signs = [], [], [], []
            append_params, append_chi, append_pf, append_passed_signs = \
                params.append, chi_squared.append, pass_fail.append, \
                passed_signs.append
            for j in range(signs.shape[0]):
                fit = qp_class(
                    x, y, N, signs[j, :], mid_point,
                    self.model_type, self.cvxopt_maxiter, self.filtering,
                    self.all_output, self.ifp, self.ifp_list,
                    self.initial_params, self.basis_functions,
                    self.data_matrix, self.der_pres, self.model,
                    self.derivatives_function, self.args,
                    self.warnings)

                if self.all_output is True:
                    print(
                        '-----------------------------------------------' +
                        '-----')
                    print('Polynomial Order:', N)
                    print('Number of Derivatives:', N-2)
                    print('Signs :', signs[j, :])
                    print('Objective Function Value:', fit.chi_squared)
                    print('Parameters:', (fit.parameters).T)
                    print('Method:', self.fit_type)
                    print('Model:', self.model_type)
                    print('Inflection Points?:', self.ifp)
                    if self.ifp is True:
                        print('Inflection Point Derivatives:', self.ifp_list)
                        print(
                            'Inflection Points Used? (0 signifies Yes):',
                            fit.pass_fail)
                    print(
                        '-----------------------------------------------' +
                        '-----')

                if self.filtering is False:
                    append_params(fit.parameters)
                    append_chi(fit.chi_squared)
                    append_pf(fit.pass_fail)
                    append_passed_signs(signs[j, :])
                if self.filtering is True:
                    if fit.pass_fail == []:
                        pass
                    else:
                        append_params(fit.parameters)
                        append_chi(fit.chi_squared)
                        append_pf(fit.pass_fail)
                        append_passed_signs(signs[j, :])
                        if self.data_save is True:
                            save(
                                self.base_dir, fit.parameters, fit.chi_squared,
                                signs[j, :], N, self.fit_type)
            params, chi_squared, pass_fail, passed_signs = np.array(params), \
                np.array(chi_squared), np.array(pass_fail), \
                np.array(passed_signs)

            Optimum_chi_squared = chi_squared.min()
            for f in range(len(chi_squared)):
                if chi_squared[f] == chi_squared.min():
                    Optimum_params = params[f, :]
                    Optimum_sign_combination = passed_signs[f, :]

            y_fit = Models_class(
                Optimum_params, x, y, N, mid_point, self.model_type,
                self.model, self.args).y_sum
            der = derivative_class(
                x, y, Optimum_params, N,
                Optimum_sign_combination, mid_point, self.model_type, self.ifp,
                self.derivatives_function, self.args, self.warnings)
            derivatives, Optimum_pass_fail = der.derivatives, der.pass_fail

            end = time.time()

            print(
                '######################################################' +
                '#######')
            print(
                '----------------------OPTIMUM RESULT--------------------' +
                '-----')
            print('Time:', end-start)
            print('Polynomial Order:', N)
            print('Number of Derivatives:', N-2)
            print('Signs :', Optimum_sign_combination)
            print('Objective Function Value:', Optimum_chi_squared)
            print('Parameters:', Optimum_params.T)
            print('Method:', self.fit_type)
            print('Model:', self.model_type)
            print('Inflection Points?:', self.ifp)
            if self.ifp is True:
                print('Inflection Point Derivatives:', self.ifp_list)
                print(
                    'Inflection Points Used? (0 signifies Yes):',
                    Optimum_pass_fail)
            print(
                '-------------------------------------------------------' +
                '------')
            print(
                '######################################################' +
                '#######')

            save_optimum(
                self.base_dir, end-start, N,
                Optimum_sign_combination, Optimum_chi_squared,
                Optimum_params, self.fit_type, self.model_type, self.ifp,
                self.ifp_list, Optimum_pass_fail)

            return y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination

        def qp_sign_flipping(x, y, N, mid_point):
            print(
                '######################################################' +
                '#######')
            start = time.time()

            if self.data_save is True:
                if not os.path.exists(
                        self.base_dir + 'MSF_Order_' + str(N) +
                        '_' + self.fit_type + '/'):
                    os.mkdir(
                        self.base_dir + 'MSF_Order_' + str(N) + '_' +
                        self.fit_type + '/')

            runs = np.arange(0, 2*(N-2)**2, 1)
            Run_Optimum_params, Run_Optimum_chi_squared, \
                Run_Optimum_sign_combination = [], [], []
            for k in range(len(runs)):
                if self.filtering is True:
                    pass_fail = []
                    while pass_fail == []:
                        signs = []
                        append_signs = signs.append
                        for l in range(N-2):
                            sign = np.random.randint(0, 2)
                            if sign == 0:
                                sign = -1
                            append_signs(sign)
                        signs = np.array(signs)
                        fit = qp_class(
                            x, y, N, signs, mid_point,
                            self.model_type, self.cvxopt_maxiter,
                            self.filtering, self.all_output, self.ifp,
                            self.ifp_list, self.initial_params,
                            self.basis_functions,
                            self.data_matrix, self.der_pres, self.model,
                            self.derivatives_function, self.args,
                            self.warnings)
                        chi_squared_old, pass_fail = fit.chi_squared, \
                            fit.pass_fail
                        if self.all_output is True:
                            print(
                                '----------------------------------------' +
                                '------------')
                            print('Polynomial Order:', N)
                            print('Number of Derivatives:', N-2)
                            print('Signs :', signs)
                            print('Objective Function Value:', chi_squared_old)
                            print('Parameters:', (fit.parameters).T)
                            print('Method:', self.fit_type)
                            print('Model:', self.model_type)
                            print('Inflection Points?:', self.ifp)
                            if self.ifp is True:
                                print(
                                    'Inflection Point Derivatives:',
                                    self.ifp_list)
                                print(
                                    'Inflection Points Used? (0 signifies' +
                                    'Yes):', fit.pass_fail)
                            print(
                                '--------------------------------------' +
                                '--------------')
                        if pass_fail != []:
                            if self.data_save is True:
                                save(
                                    self.base_dir, fit.parameters,
                                    chi_squared_old, signs, N, self.fit_type)
                else:
                    signs = []
                    append_signs = signs.append
                    for l in range(N-2):
                        sign = np.random.randint(0, 2)
                        if sign == 0:
                            sign = -1
                        append_signs(sign)
                    signs = np.array(signs)
                    fit = qp_class(
                        x, y, N, signs, mid_point,
                        self.model_type, self.cvxopt_maxiter, self.filtering,
                        self.all_output, self.ifp, self.ifp_list,
                        self.initial_params, self.basis_functions,
                        self.data_matrix, self.der_pres, self.model,
                        self.derivatives_function, self.args,
                        self.warnings)
                    chi_squared_old = fit.chi_squared
                    if self.all_output is True:
                        print(
                            '--------------------------------------' +
                            '--------------')
                        print('Polynomial Order:', N)
                        print('Number of Derivatives:', N-2)
                        print('Signs :', signs)
                        print('Objective Function Value:', chi_squared_old)
                        print('Parameters:', (fit.parameters).T)
                        print('Method:', self.fit_type)
                        print('Model:', self.model_type)
                        print('Inflection Points?:', self.ifp)
                        if self.ifp is True:
                            print(
                                'Inflection Point Derivatives:',
                                self.ifp_list)
                            print(
                                'Inflection Points Used? (0 signifies' +
                                'Yes):', fit.pass_fail)
                        print(
                            '--------------------------------------' +
                            '--------------')

                chi_squared_new = 0
                parameters = []
                parameters.append(fit.parameters)
                tested_signs = []
                tested_signs.append(signs)
                h = 0
                while chi_squared_old > chi_squared_new:
                    if h > 0:
                        if chi_squared_new != 0:
                            chi_squared_old = chi_squared_new
                    if h < len(signs):
                        new_signs = np.empty(len(signs))
                        for m in range(len(signs)):
                            if m == h:
                                new_signs[h] = signs[h]*-1
                            else:
                                new_signs[m] = signs[m]
                    fit = qp_class(
                        x, y, N, new_signs, mid_point,
                        self.model_type, self.cvxopt_maxiter,
                        self.filtering, self.all_output, self.ifp,
                        self.ifp_list, self.initial_params,
                        self.basis_functions,
                        self.data_matrix, self.der_pres, self.model,
                        self.derivatives_function, self.args,
                        self.warnings)
                    chi_squared_new = fit.chi_squared
                    if self.all_output is True:
                        print(
                            '--------------------------------------' +
                            '--------------')
                        print('Polynomial Order:', N)
                        print('Number of Derivatives:', N-2)
                        print('Signs :', new_signs)
                        print('Objective Function Value:', chi_squared_new)
                        print('Parameters:', (fit.parameters).T)
                        print('Method:', self.fit_type)
                        print('Model:', self.model_type)
                        print('Inflection Points?:', self.ifp)
                        if self.ifp is True:
                            print(
                                'Inflection Point Derivatives:',
                                self.ifp_list)
                            print(
                                'Inflection Points Used? (0 signifies' +
                                'Yes):', fit.pass_fail)
                        print(
                            '--------------------------------------' +
                            '--------------')
                    if self.filtering is True:
                        if fit.pass_fail == []:
                            chi_squared_new = 0
                        else:
                            parameters.append(fit.parameters)
                            tested_signs.append(new_signs)
                            if self.data_save is True:
                                save(
                                    self.base_dir, fit.parameters,
                                    chi_squared_new, new_signs, N,
                                    self.fit_type)
                    if self.filtering is False:
                        parameters.append(fit.parameters)
                        tested_signs.append(new_signs)
                    if h <= len(signs):
                        pass
                    else:
                        break
                    h += 1
                Run_Optimum_chi_squared.append(chi_squared_old)
                parameters = np.array(parameters)
                if parameters.shape[0] > 1:
                    Run_Optimum_params.append(parameters[-2])
                    Run_Optimum_sign_combination.append(tested_signs[-2])
                else:
                    Run_Optimum_params.append(parameters[0])
                    Run_Optimum_sign_combination.append(tested_signs[0])
            Run_Optimum_chi_squared = np.array(Run_Optimum_chi_squared)
            for j in range(len(Run_Optimum_chi_squared)):
                if Run_Optimum_chi_squared[j] == Run_Optimum_chi_squared.min():
                    Optimum_chi_squared = Run_Optimum_chi_squared[j]
                    Optimum_params = Run_Optimum_params[j]
                    Optimum_sign_combination = Run_Optimum_sign_combination[j]

            y_fit = Models_class(
                Optimum_params, x, y, N, mid_point,
                self.model_type, self.model, self.args).y_sum
            der = derivative_class(
                x, y, Optimum_params, N,
                Optimum_sign_combination, mid_point, self.model_type, self.ifp,
                self.derivatives_function, self.args, self.warnings)
            derivatives, Optimum_pass_fail = der.derivatives, der.pass_fail

            end = time.time()

            print(
                '######################################################' +
                '#######')
            print(
                '----------------------OPTIMUM RESULT--------------------' +
                '-----')
            print('Time:', end-start)
            print('Polynomial Order:', N)
            print('Number of Derivatives:', N-2)
            print('Signs :', Optimum_sign_combination)
            print('Objective Function Value:', Optimum_chi_squared)
            print('Parameters:', Optimum_params.T)
            print('Method:', self.fit_type)
            print('Model:', self.model_type)
            print('Inflection Points?:', self.ifp)
            if self.ifp is True:
                print('Inflection Point Derivatives:', self.ifp_list)
                print(
                    'Inflection Points Used? (0 signifies Yes):',
                    Optimum_pass_fail)
            print(
                '----------------------------------------------------' +
                '---------')
            print(
                '####################################################' +
                '#########')

            save_optimum(
                self.base_dir, end-start, N,
                Optimum_sign_combination, Optimum_chi_squared,
                Optimum_params, self.fit_type, self.model_type, self.ifp,
                self.ifp_list, Optimum_pass_fail)

            return y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination

        def plotting(x, y, N, y_fit, derivatives):
            for i in range(len(N)):
                pl.subplot(111)
                pl.plot(
                    x, y_fit[i, :], c='r', label='Fitted MSF with N = '
                    + str(N[i]))
                pl.plot(x, y, label='Data')
                pl.legend(loc=0)
                pl.xlabel('x')
                pl.ylabel('y')
                pl.tight_layout()
                pl.savefig(
                    self.base_dir + 'MSF_Order_' + str(N[i]) + '_' +
                    self.fit_type + '/Fit.pdf')
                pl.close()

                rms = (np.sqrt(np.sum((y-y_fit[i, :])**2)/len(y)))
                np.save(
                    self.base_dir + 'MSF_Order_' + str(N[i]) +
                    '_' + self.fit_type + '/RMS.npy', rms)

                pl.subplot(111)
                pl.plot(x, y - y_fit[i, :], label='RMS = %2.5f' % (rms))
                pl.fill_between(
                    x, np.array([rms]*len(x)),
                    -np.array([rms]*len(x)), color='r', alpha=0.5)
                pl.legend(loc=0)
                pl.ylabel('Residuals')
                pl.xlabel('x')
                pl.tight_layout()
                pl.savefig(
                    self.base_dir + 'MSF_Order_' + str(N[i]) +
                    '_' + self.fit_type + '/RMS.pdf')
                pl.close()

                pl.subplot(111)
                [pl.plot(
                    x, derivatives[i][j, :], label='M:' + str(j + 2) +
                    ' Minimum: %2.2e' % (derivatives[i][j, :].min()))
                    for j in range(derivatives[i].shape[0])]
                pl.legend(loc=0, fontsize='small')
                pl.xlabel(r'x')
                pl.ylabel('M Order Derivatives')
                pl.tight_layout()
                pl.savefig(
                    self.base_dir + 'MSF_Order_' + str(N[i]) + '_'
                    + self.fit_type + '/Derivatives.pdf')
                pl.close()

        mid_point = len(self.x)//2
        if self.fit_type == 'qp':
            y_fit, Optimum_sign_combinations, derivatives, Optimum_params, \
                Optimum_chi_squareds = [], [], [], [], []
            for i in range(len(self.N)):
                y_result, derive, obj, params, signs = \
                    qp(self.x, self.y, self.N[i], mid_point)
                y_fit.append(y_result)
                Optimum_sign_combinations.append(signs)
                derivatives.append(derive)
                Optimum_params.append(params)
                Optimum_chi_squareds.append(obj)
            y_fit, Optimum_sign_combinations, derivatives, Optimum_params, \
                Optimum_chi_squareds = np.array(y_fit), \
                np.array(Optimum_sign_combinations), np.array(derivatives), \
                np.array(Optimum_params), np.array(Optimum_chi_squareds)
        if self.fit_type == 'qp-sign_flipping':
            y_fit, Optimum_sign_combinations, derivatives, Optimum_params, \
                Optimum_chi_squareds = [], [], [], [], []
            for i in range(len(self.N)):
                if self.N[i] <= 10:
                    y_result, derive, obj, params, signs = \
                        qp(self.x, self.y, self.N[i], mid_point)
                else:
                    y_result, derive, obj, params, signs = \
                        qp_sign_flipping(self.x, self.y, self.N[i], mid_point)
                y_fit.append(y_result)
                Optimum_sign_combinations.append(signs)
                derivatives.append(derive)
                Optimum_params.append(params)
                Optimum_chi_squareds.append(obj)
            y_fit, Optimum_sign_combinations, derivatives, Optimum_params, \
                Optimum_chi_squareds = np.array(y_fit), \
                np.array(Optimum_sign_combinations), np.array(derivatives), \
                np.array(Optimum_params), np.array(Optimum_chi_squareds)

        rms = [
            (np.sqrt(np.sum((self.y-y_fit[i, :])**2)/len(self.y)))
            for i in range(len(self.N))]

        if self.data_save is True:
            plotting(self.x, self.y, self.N, y_fit, derivatives)

        return y_fit, Optimum_sign_combinations, Optimum_params, derivatives, \
            Optimum_chi_squareds, rms

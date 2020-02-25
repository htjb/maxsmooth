
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
import shutil


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
            fitting space is changed.

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
        self.fit_type, self.base_dir, self.model_type, \
            self.all_output, self.cvxopt_maxiter, \
            self.ifp_list, self.data_save, \
            self.warnings, self.constraints = setting.fit_type, \
            setting.base_dir, setting.model_type, \
            setting.all_output, setting.cvxopt_maxiter, \
            setting.ifp_list, setting.data_save, \
            setting.warnings, setting.constraints

        if self.ifp_list is not None:
            for i in range(len(self.ifp_list)):
                if self.ifp_list[i] < self.constraints:
                    print('ERROR: One or more specified derivatives for' +
                        ' inflection points is less than the minimum constrained' +
                        ' derivative.\n ifp_list = ' + str(self.ifp_list) + '\n' +
                        ' Minimum Constrained Derivative = ' + str(self.constraints))
                    sys.exit(1)

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
            self.Optimum_chi, self.rms, self.Optimum_ifp_dict \
            = self.fitting()

    def fitting(self):

        def signs_array(nums):
            return np.array(list(product(*((x, -x) for x in nums))))

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        if os.path.isdir(self.base_dir+'Output_Parameters/'):
            shutil.rmtree(self.base_dir+'Output_Parameters/')
        if os.path.isdir(self.base_dir+'Output_Signs/'):
            shutil.rmtree(self.base_dir+'Output_Signs/')
        if os.path.isdir(self.base_dir+'Output_Evaluation/'):
            shutil.rmtree(self.base_dir+'Output_Evaluation/')

        def qp(x, y, N, mid_point): # Testing all signs
            print(
                '######################################################' +
                '#######')
            start = time.time()

            if self.ifp_list is not None:
                signs = signs_array([1]*(N-self.constraints-len(self.ifp_list)))
            else:
                signs = signs_array([1]*(N-self.constraints))

            params, chi_squared, ifp_dict, passed_signs = [], [], [], []
            append_params, append_chi, append_ifp_dict, append_passed_signs = \
                params.append, chi_squared.append, ifp_dict.append, \
                passed_signs.append
            for j in range(signs.shape[0]):
                fit = qp_class(
                    x, y, N, signs[j, :], mid_point,
                    self.model_type, self.cvxopt_maxiter,
                    self.all_output, self.ifp_list,
                    self.initial_params, self.basis_functions,
                    self.data_matrix, self.der_pres, self.model,
                    self.derivatives_function, self.args,
                    self.warnings, self.constraints)

                if self.all_output is True:
                    print(
                        '-----------------------------------------------' +
                        '-----')
                    print('Polynomial Order:', N)
                    if self.ifp_list is not None:
                        print('Number of Constrained Derivatives:',
                            N-self.constraints-len(self.ifp_list))
                    else:
                        print('Number of Constrained Derivatives:',
                            N-self.constraints)
                    print('Signs :', signs[j, :])
                    print('Objective Function Value:', fit.chi_squared)
                    print('Parameters:', (fit.parameters).T)
                    print('Method:', self.fit_type)
                    print('Model:', self.model_type)
                    print('Constraints: m >=', self.constraints)
                    if self.ifp_list is None:
                        print(
                            'Inflection Points Used? (0 signifies Yes):',
                            fit.ifp_dict)
                    if self.ifp_list is not None:
                        print('Inflection Point Derivatives:', self.ifp_list)
                        print(
                            'Inflection Points Used? (0 signifies Yes):',
                            fit.ifp_dict)
                    print(
                        '-----------------------------------------------' +
                        '-----')

                append_params(fit.parameters)
                append_chi(fit.chi_squared)
                append_ifp_dict(fit.ifp_dict)
                append_passed_signs(signs[j, :])
                if self.data_save is True:
                    save(
                        self.base_dir, fit.parameters, fit.chi_squared,
                        signs[j, :], N, self.fit_type)

            params, chi_squared, ifp_dict, passed_signs = np.array(params), \
                np.array(chi_squared), np.array(ifp_dict), \
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
                Optimum_sign_combination, mid_point, self.model_type, self.ifp_list,
                self.derivatives_function, self.args, self.warnings,
                self.constraints)
            derivatives, Optimum_ifp_dict = der.derivatives, der.ifp_dict

            end = time.time()

            print(
                '######################################################' +
                '#######')
            print(
                '----------------------OPTIMUM RESULT--------------------' +
                '-----')
            print('Time:', end-start)
            print('Polynomial Order:', N)
            if self.ifp_list is not None:
                print('Number of Constrained Derivatives:',
                    N-self.constraints-len(self.ifp_list))
            else:
                print('Number of Constrained Derivatives:',
                    N-self.constraints)
            print('Signs :', Optimum_sign_combination)
            print('Objective Function Value:', Optimum_chi_squared)
            print('Parameters:', Optimum_params.T)
            print('Method:', self.fit_type)
            print('Model:', self.model_type)
            print('Constraints: m >=', self.constraints)
            if self.ifp_list is None:
                print(
                    'Inflection Points Used? (0 signifies Yes):',
                    Optimum_ifp_dict)
            if self.ifp_list is not None:
                print('Inflection Point Derivatives:', self.ifp_list)
                print(
                    'Inflection Points Used? (0 signifies Yes):',
                    Optimum_ifp_dict)
            print(
                '-------------------------------------------------------' +
                '------')
            print(
                '######################################################' +
                '#######')

            save_optimum(
                self.base_dir, end-start, N,
                Optimum_sign_combination, Optimum_chi_squared,
                Optimum_params, self.fit_type, self.model_type,
                self.ifp_list, Optimum_ifp_dict, self.constraints)

            return y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_ifp_dict

        def qp_sign_flipping(x, y, N, mid_point): # Steepest Descent
            print(
                '######################################################' +
                '#######')
            start = time.time()

            # Generate all sign combinations using function defined
            # at top of fitting
            if self.ifp_list is not None:
                array_signs = signs_array([1]*(N-self.constraints-len(self.ifp_list)))
            else:
                array_signs = signs_array([1]*(N-self.constraints))

            # Randomly pick one of the generated sign combinations
            r = np.random.randint(0, len(array_signs), 1)
            signs = array_signs[r][0]

            # Calculate the starting chi value
            fit = qp_class(
                x, y, N, signs, mid_point,
                self.model_type, self.cvxopt_maxiter,
                self.all_output, self.ifp_list,
                self.initial_params, self.basis_functions,
                self.data_matrix, self.der_pres, self.model,
                self.derivatives_function, self.args,
                self.warnings, self.constraints)
            chi_squared_old = fit.chi_squared

            if self.all_output is True:
                print(
                    '--------------------------------------' +
                    '--------------')
                print('Polynomial Order:', N)
                if self.ifp_list is not None:
                    print('Number of Constrained Derivatives:',
                        N-self.constraints-len(self.ifp_list))
                else:
                    print('Number of Constrained Derivatives:',
                        N-self.constraints)
                print('Signs :', signs)
                print('Objective Function Value:', chi_squared_old)
                print('Parameters:', (fit.parameters).T)
                print('Method:', self.fit_type)
                print('Model:', self.model_type)
                print('Constraints: m >=', self.constraints)
                if self.ifp_list is None:
                    print(
                        'Inflection Points Used? (0 signifies Yes):',
                        fit.ifp_dict)
                if self.ifp_list is not None:
                    print(
                        'Inflection Point Derivatives:',
                        self.ifp_list)
                    print(
                        'Inflection Points Used? (0 signifies' +
                        'Yes):', fit.ifp_dict)
                print(
                    '--------------------------------------' +
                    '--------------')
            if self.data_save is True:
                save(
                    self.base_dir, fit.parameters, fit.chi_squared,
                    signs, N, self.fit_type)

            # Transforms or 'steps' of sign combination
            sign_transform = []
            for i in range(len(signs)):
                base = np.array([1]*len(signs))
                base[i] = -1
                sign_transform.append(base)
            sign_transform = np.array(sign_transform)
            chi_squared_new = 0 # Initialize new chi squared value as 0
            previous_signs = signs # Original Signs that were randomly chosen

            # Steepest descent algorithm
            while chi_squared_new < chi_squared_old:
                # If we enter back in the loop after first iter replace old
                # chi with new
                if chi_squared_new != 0:
                    chi_squared_old = chi_squared_new
                for h in range(sign_transform.shape[0]):
                    # Transform old signs and calculate chi
                    signs = previous_signs * sign_transform[h]
                    fit = qp_class(
                        x, y, N, signs, mid_point,
                        self.model_type, self.cvxopt_maxiter,
                        self.all_output,
                        self.ifp_list, self.initial_params,
                        self.basis_functions,
                        self.data_matrix, self.der_pres, self.model,
                        self.derivatives_function, self.args,
                        self.warnings, self.constraints)
                    if fit.chi_squared < chi_squared_old:
                        # If new chi is a step down hill update the value of
                        # chi new and the best sign combination ('previous_signs')
                        chi_squared_new = fit.chi_squared
                        previous_signs = signs
                        break
                    if h == sign_transform.shape[0] - 1:
                        # If no step down break for loop and break while loop
                        # by setting...
                        chi_squared_new = chi_squared_old
                        break
                if self.all_output is True:
                    print(
                        '--------------------------------------' +
                        '--------------')
                    print('Polynomial Order:', N)
                    if self.ifp_list is not None:
                        print('Number of Constrained Derivatives:',
                            N-self.constraints-len(self.ifp_list))
                    else:
                        print('Number of Constrained Derivatives:',
                            N-self.constraints)
                    print('Signs :', signs)
                    print('Objective Function Value:', fit.chi_squared)
                    print('Parameters:', fit.parameters.T)
                    print('Method:', self.fit_type)
                    print('Model:', self.model_type)
                    print('Constraints: m >=', self.constraints)
                    if self.ifp_list is None:
                        print(
                            'Inflection Points Used? (0 signifies Yes):',
                            fit.ifp_dict)
                    if self.ifp_list is not None:
                        print(
                            'Inflection Point Derivatives:',
                            self.ifp_list)
                        print(
                            'Inflection Points Used? (0 signifies' +
                            'Yes):', fit.ifp_dict)
                    print(
                        '--------------------------------------' +
                        '--------------')
                if self.data_save is True:
                    save(
                        self.base_dir, fit.parameters, fit.chi_squared,
                        signs, N, self.fit_type)

            fit = qp_class(
                x, y, N, previous_signs, mid_point,
                self.model_type, self.cvxopt_maxiter,
                self.all_output,
                self.ifp_list, self.initial_params,
                self.basis_functions,
                self.data_matrix, self.der_pres, self.model,
                self.derivatives_function, self.args,
                self.warnings, self.constraints)

            Optimum_params = fit.parameters
            Optimum_sign_combination = previous_signs
            Optimum_chi_squared = fit.chi_squared

            # Re-calculate fitted y and derivatives. Could return this from
            # the qp_class but it is easier to redo rather than save at each
            # iteration.
            y_fit = Models_class(
                Optimum_params, x, y, N, mid_point,
                self.model_type, self.model, self.args).y_sum
            der = derivative_class(
                x, y, Optimum_params, N,
                Optimum_sign_combination, mid_point, self.model_type, self.ifp_list,
                self.derivatives_function, self.args, self.warnings,
                self.constraints)
            derivatives, Optimum_ifp_dict = der.derivatives, der.ifp_dict

            end = time.time()

            print(
                '######################################################' +
                '#######')
            print(
                '----------------------OPTIMUM RESULT--------------------' +
                '-----')
            print('Time:', end-start)
            print('Polynomial Order:', N)
            if self.ifp_list is not None:
                print('Number of Constrained Derivatives:',
                    N-self.constraints-len(self.ifp_list))
            else:
                print('Number of Constrained Derivatives:',
                    N-self.constraints)
            print('Signs :', Optimum_sign_combination)
            print('Objective Function Value:', Optimum_chi_squared)
            print('Parameters:', Optimum_params.T)
            print('Method:', self.fit_type)
            print('Model:', self.model_type)
            print('Constraints: m >=', self.constraints)
            if self.ifp_list is None:
                print(
                    'Inflection Points Used? (0 signifies Yes):',
                    Optimum_ifp_dict)
            if self.ifp_list is not None:
                print('Inflection Point Derivatives:', self.ifp_list)
                print(
                    'Inflection Points Used? (0 signifies Yes):',
                    Optimum_ifp_dict)
            print(
                '----------------------------------------------------' +
                '---------')
            print(
                '####################################################' +
                '#########')

            save_optimum(
                self.base_dir, end-start, N,
                Optimum_sign_combination, Optimum_chi_squared,
                Optimum_params, self.fit_type, self.model_type,
                self.ifp_list, Optimum_ifp_dict, self.constraints)

            return y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_ifp_dict

        mid_point = len(self.x)//2
        if self.fit_type == 'qp':
            y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_ifp_dict = \
                qp(self.x, self.y, self.N, mid_point)
        if self.fit_type == 'qp-sign_flipping':
            y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_ifp_dict = \
                qp_sign_flipping(self.x, self.y, self.N, mid_point)

        rms = (np.sqrt(np.sum((self.y-y_fit)**2)/len(self.y)))


        return y_fit, Optimum_sign_combination, Optimum_params, derivatives, \
            Optimum_chi_squared, rms, Optimum_ifp_dict


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

        N: *int* The number of terms in the MSF polynomial function.

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

    def __init__(self, x, y, N, **kwargs):
        self.x = x
        self.y = y
        self.N = N

        for keys, values in kwargs.items():
            if keys not in set(['fit_type', 'model_type', 'base_dir',
                'all_output', 'cvxopt_maxiter', 'ifp_list', 'data_save',
                'warnings', 'constraints', 'chi_squared_limit', 'cap',
                'initial_params','basis_functions','der_pres', 'model',
                'derivatives', 'args']):
                print("Error: Unexpected keyword argument in smooth.")
                sys.exit(1)

        self.fit_type = kwargs.pop('fit_type', 'qp-sign_flipping')
        if self.fit_type not in set(['qp', 'qp-sign_flipping']):
            print("Error: Invalid 'fit_type'. Valid entries include 'qp'\n" +
                "'qp-sign_flipping'")
            sys.exit(1)

        self.base_dir = kwargs.pop('base_dir', 'Fitted_Output/')
        self.model_type = kwargs.pop('model_type', 'difference_polynomial')
        if self.model_type not in set(['normalised_polynomial', 'polynomial',
            'log_polynomial', 'loglog_polynomial', 'difference_polynomial',
            'exponential', 'legendre']):
            print("Error: Invalid 'model_type'. See documentation for built" +
                "in models.")
            sys.exit(1)

        self.cvxopt_maxiter = kwargs.pop('cvxopt_maxiter', 10000)
        if type(self.cvxopt_maxiter) is not int:
            print("Error: 'cvxopt_maxiter' is not integer.")
            sys.exit(1)

        self.all_output = kwargs.pop('all_output', False)
        self.data_save = kwargs.pop('data_save', False)
        self.warnings = kwargs.pop('warnings', False)
        boolean_kwargs = [self.warnings, self.data_save, self.all_output]
        for i in range(len(boolean_kwargs)):
            if type(boolean_kwargs[i]) is not bool:
                print("Error: Boolean keyword argument with value "
                    + str(boolean_kwargs[i]) +
                    " is not True or False.")
                sys.exit(1)

        self.constraints = kwargs.pop('constraints', 2)
        if type(self.constraints) is not int:
            print("Error: 'constraints' is not an integer")
            sys.exit(1)
        if self.constraints > self.N-1:
            print("Error: 'constraints' exceeds the number of derivatives.")
            sys.exit(1)

        self.ifp_list = kwargs.pop('ifp_list', None)
        if self.ifp_list is not None:
            for i in range(len(self.ifp_list)):
                if type(self.ifp_list[i]) is not int:
                    print("Error: Entries in 'ifp_list' are not integer.")
                    sys.exit(1)
                if self.ifp_list[i] < self.constraints:
                    print('ERROR: One or more specified derivatives for' +
                        ' inflection points is less than the minimum constrained' +
                        ' derivative.\n ifp_list = ' + str(self.ifp_list) + '\n' +
                        ' Minimum Constrained Derivative = ' + str(self.constraints))
                    sys.exit(1)

        self.chi_squared_limit = kwargs.pop('chi_squared_limit', None)
        self.cap = kwargs.pop('cap', None)
        if self.chi_squared_limit is not None:
            if type(self.chi_squared_limit) is not int:
                    if type(self.chi_squared_limit) is not float:
                        print("Error: Limit on maximum allowed increase in chi squared" +
                            ", 'chi_squared_limit', is not an integer or float.")
                        sys.exit(1)
        if self.cap is not None:
            if type(self.cap) is not int:
                    print("Error: The cap on directional exploration" +
                        ", 'cap', is not an integer.")
                    sys.exit(1)

        self.initial_params = kwargs.pop('initial_params', None)
        if self.initial_params is not None and len(self.initial_params) \
            != self.N:
            print("Error: Initial Parameters isnot equal to the number" +
                "of terms in the polynomial, N.")
            sys.exit(1)

        self.basis_functions = kwargs.pop('basis_functions', None)
        self.der_pres = kwargs.pop('der_pres', None)
        self.model = kwargs.pop('model', None)
        self.derivatives_function = kwargs.pop('derivatives', None)
        self.args = kwargs.pop('args', None)

        self.new_basis = {'basis_function':
            self.basis_functions, 'der_pres': self.der_pres,
            'derivatives_function': self.derivatives_function,
            'model': self.model, 'args': self.args}
        if np.all(value is None for value in self.new_basis.values()):
            pass
        else:
            count = 0
            for key, value in self.new_basis.items():
                if value is None and key != 'args':
                    print(
                        'Error: Attempt to change basis functions failed.' +
                        ' One or more functions not defined.' +
                        ' Please consult documentation.')
                    sys.exit(1)
                if value is None and key == 'args':
                    print('Warning: No additional arguments passed to new basis' +
                        'functions')
                count += 1

            if count == len(self.new_basis):
                self.model_type = 'user_defined'

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

        def qp(x, y, mid_point): # Testing all signs
            print(
                '######################################################' +
                '#######')
            start = time.time()

            if self.ifp_list is not None:
                signs = signs_array([1]*(self.N-self.constraints-len(self.ifp_list)))
            else:
                signs = signs_array([1]*(self.N-self.constraints))

            params, chi_squared, ifp_dict, passed_signs = [], [], [], []
            append_params, append_chi, append_ifp_dict, append_passed_signs = \
                params.append, chi_squared.append, ifp_dict.append, \
                passed_signs.append
            for j in range(signs.shape[0]):
                fit = qp_class(
                    x, y, self.N, signs[j, :], mid_point,
                    self.model_type, self.cvxopt_maxiter,
                    self.all_output, self.ifp_list,
                    self.initial_params,
                    self.warnings, self.constraints, self.new_basis)

                if self.all_output is True:
                    print(
                        '-----------------------------------------------' +
                        '-----')
                    print('Polynomial Order:', self.N)
                    if self.ifp_list is not None:
                        print('Number of Constrained Derivatives:',
                            self.N-self.constraints-len(self.ifp_list))
                    else:
                        print('Number of Constrained Derivatives:',
                            self.N-self.constraints)
                    print('Signs :', signs[j, :])
                    print('Objective Function Value:', fit.chi_squared)
                    print('Parameters:', (fit.parameters).T)
                    print('Method:', self.fit_type)
                    print('Model:', self.model_type)
                    print('Constraints: m >=', self.constraints)
                    if self.ifp_list is None:
                        print(
                            'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                            fit.ifp_dict)
                    if self.ifp_list is not None:
                        print('Inflection Point Derivatives:', self.ifp_list)
                        print(
                            'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
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
                        signs[j, :], self.N, self.fit_type)

            params, chi_squared, ifp_dict, passed_signs = np.array(params), \
                np.array(chi_squared), np.array(ifp_dict), \
                np.array(passed_signs)

            Optimum_chi_squared = chi_squared.min()
            for f in range(len(chi_squared)):
                if chi_squared[f] == chi_squared.min():
                    Optimum_params = params[f, :]
                    Optimum_sign_combination = passed_signs[f, :]

            if self.model_type == 'loglog_polynomial':
                y_fit = Models_class(
                    Optimum_params, np.log10(x/x[mid_point]), np.log10(y),
                    self.N, mid_point,
                    self.model_type, self.new_basis).y_sum
                der = derivative_class(
                    np.log10(x/x[mid_point]), np.log10(y), Optimum_params,
                    self.N,
                    mid_point, self.model_type, self.ifp_list, self.warnings,
                    self.constraints, self.new_basis)
                derivatives, Optimum_ifp_dict = der.derivatives, der.ifp_dict
            else:
                y_fit = Models_class(
                    Optimum_params, x, y, self.N, mid_point,
                    self.model_type, self.new_basis).y_sum
                der = derivative_class(
                    x, y, Optimum_params, self.N,
                    mid_point, self.model_type, self.ifp_list, self.warnings,
                    self.constraints, self.new_basis)
                derivatives, Optimum_ifp_dict = der.derivatives, der.ifp_dict

            end = time.time()

            print(
                '######################################################' +
                '#######')
            print(
                '----------------------OPTIMUM RESULT--------------------' +
                '-----')
            print('Time:', end-start)
            print('Polynomial Order:', self.N)
            if self.ifp_list is not None:
                print('Number of Constrained Derivatives:',
                    self.N-self.constraints-len(self.ifp_list))
            else:
                print('Number of Constrained Derivatives:',
                    self.N-self.constraints)
            print('Signs :', Optimum_sign_combination)
            print('Objective Function Value:', Optimum_chi_squared)
            print('Parameters:', Optimum_params.T)
            print('Method:', self.fit_type)
            print('Model:', self.model_type)
            print('Constraints: m >=', self.constraints)
            if self.ifp_list is None:
                print(
                    'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                    Optimum_ifp_dict)
            if self.ifp_list is not None:
                print('Inflection Point Derivatives:', self.ifp_list)
                print(
                    'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                    Optimum_ifp_dict)
            print(
                '-------------------------------------------------------' +
                '------')
            print(
                '######################################################' +
                '#######')

            save_optimum(
                self.base_dir, end-start, self.N,
                Optimum_sign_combination, Optimum_chi_squared,
                Optimum_params, self.fit_type, self.model_type,
                self.ifp_list, Optimum_ifp_dict, self.constraints)

            return y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_ifp_dict

        def qp_sign_flipping(x, y, mid_point): # Steepest Descent
            print(
                '######################################################' +
                '#######')
            start = time.time()

            if self.ifp_list is not None:
                array_signs = signs_array([1]*(self.N-self.constraints-len(self.ifp_list)))
            else:
                array_signs = signs_array([1]*(self.N-self.constraints))

            r = np.random.randint(0, len(array_signs), 1)
            signs = array_signs[r][0]

            tested_indices = []
            chi_squared = []
            parameters =[]
            tested_signs = []
            for i in range(len(array_signs)):
                if i == r:
                    tested_indices.append(i)
            fit = qp_class(
                x, y, self.N, signs, mid_point,
                self.model_type, self.cvxopt_maxiter,
                self.all_output, self.ifp_list,
                self.initial_params,
                self.warnings, self.constraints, self.new_basis)
            chi_squared.append(fit.chi_squared)
            tested_signs.append(signs)
            parameters.append(fit.parameters)
            chi_squared_old = fit.chi_squared
            previous_signs = signs

            if self.all_output is True:
                print(
                    '--------------------------------------' +
                    '--------------')
                print('Polynomial Order:', self.N)
                if self.ifp_list is not None:
                    print('Number of Constrained Derivatives:',
                        self.N-self.constraints-len(self.ifp_list))
                else:
                    print('Number of Constrained Derivatives:',
                        self.N-self.constraints)
                print('Signs :', signs)
                print('Objective Function Value:', chi_squared_old)
                print('Parameters:', (fit.parameters).T)
                print('Method:', self.fit_type)
                print('Model:', self.model_type)
                print('Constraints: m >=', self.constraints)
                if self.ifp_list is None:
                    print(
                        'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                        fit.ifp_dict)
                if self.ifp_list is not None:
                    print(
                        'Inflection Point Derivatives:',
                        self.ifp_list)
                    print(
                        'Inflection Points Used? (0 signifies' +
                        'Yes\n in derivative order "i"):', fit.ifp_dict)
                print(
                    '--------------------------------------' +
                    '--------------')
            if self.data_save is True:
                save(
                    self.base_dir, fit.parameters, fit.chi_squared,
                    signs, self.N, self.fit_type)

            # Transforms or 'steps' of sign combination
            sign_transform = []
            for i in range(len(signs)):
                base = np.array([1]*len(signs))
                base[i] = -1
                sign_transform.append(base)
            sign_transform = np.array(sign_transform)
            chi_squared_new = 0

            while chi_squared_new < chi_squared_old:
                if chi_squared_new != 0:
                    chi_squared_old = chi_squared_new
                for h in range(sign_transform.shape[0]):
                    signs = previous_signs * sign_transform[h]
                    for i in range(len(array_signs)):
                        if np.all(signs == array_signs[i]):
                            ind = i
                    if ind in set(tested_indices):
                        pass
                    else:
                        tested_indices.append(ind)
                        fit = qp_class(
                            x, y, self.N, signs, mid_point,
                            self.model_type, self.cvxopt_maxiter,
                            self.all_output,
                            self.ifp_list, self.initial_params,
                            self.warnings, self.constraints, self.new_basis)
                        if fit.chi_squared < chi_squared_old:
                            chi_squared_new = fit.chi_squared
                            previous_signs = signs
                            chi_squared.append(fit.chi_squared)
                            tested_signs.append(signs)
                            parameters.append(fit.parameters)

                            if self.all_output is True:
                                print(
                                    '--------------------------------------' +
                                    '--------------')
                                print('Polynomial Order:', self.N)
                                if self.ifp_list is not None:
                                    print('Number of Constrained Derivatives:',
                                        self.N-self.constraints-len(self.ifp_list))
                                else:
                                    print('Number of Constrained Derivatives:',
                                        self.N-self.constraints)
                                print('Signs :', signs)
                                print('Objective Function Value:', fit.chi_squared)
                                print('Parameters:', fit.parameters.T)
                                print('Method:', self.fit_type)
                                print('Model:', self.model_type)
                                print('Constraints: m >=', self.constraints)
                                if self.ifp_list is None:
                                    print(
                                        'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                                        fit.ifp_dict)
                                if self.ifp_list is not None:
                                    print(
                                        'Inflection Point Derivatives:',
                                        self.ifp_list)
                                    print(
                                        'Inflection Points Used? (0 signifies' +
                                        'Yes\n in derivative order "i"):', fit.ifp_dict)
                                print(
                                    '--------------------------------------' +
                                    '--------------')
                            if self.data_save is True:
                                save(
                                    self.base_dir, fit.parameters, fit.chi_squared,
                                    signs, self.N, self.fit_type)

                            break
                    if h == sign_transform.shape[0] - 1:
                        chi_squared_new = chi_squared_old
                        break

            if self.chi_squared_limit is not None:
                lim = self.chi_squared_limit*min(chi_squared)
            else:
                lim = 2*min(chi_squared)

            if self.cap is not None:
                cap = self.cap
            else:
                cap = (len(array_signs)//self.N) + self.N

            for i in range(len(array_signs)):
                if np.all(previous_signs == array_signs[i]):
                    index = i

            down_int = 1
            while down_int < cap:
                if index-down_int < 0:
                    break
                elif (index-down_int) in set(tested_indices):
                    chi_down = 0
                    pass
                else:
                    signs = array_signs[index-down_int]
                    tested_indices.append(index - down_int)
                    fit = qp_class(
                        x, y, self.N, signs, mid_point,
                        self.model_type, self.cvxopt_maxiter,
                        self.all_output,
                        self.ifp_list, self.initial_params,
                        self.warnings, self.constraints, self.new_basis)
                    chi_down = fit.chi_squared
                    chi_squared.append(fit.chi_squared)
                    tested_signs.append(signs)
                    parameters.append(fit.parameters)

                    if self.all_output is True:
                        print(
                            '--------------------------------------' +
                            '--------------')
                        print('Polynomial Order:', self.N)
                        if self.ifp_list is not None:
                            print('Number of Constrained Derivatives:',
                                self.N-self.constraints-len(self.ifp_list))
                        else:
                            print('Number of Constrained Derivatives:',
                                self.N-self.constraints)
                        print('Signs :', signs)
                        print('Objective Function Value:', fit.chi_squared)
                        print('Parameters:', fit.parameters.T)
                        print('Method:', self.fit_type)
                        print('Model:', self.model_type)
                        print('Constraints: m >=', self.constraints)
                        if self.ifp_list is None:
                            print(
                                'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                                fit.ifp_dict)
                        if self.ifp_list is not None:
                            print(
                                'Inflection Point Derivatives:',
                                self.ifp_list)
                            print(
                                'Inflection Points Used? (0 signifies' +
                                'Yes\n in derivative order "i"):', fit.ifp_dict)
                        print(
                            '--------------------------------------' +
                            '--------------')
                    if self.data_save is True:
                        save(
                            self.base_dir, fit.parameters, fit.chi_squared,
                            signs, self.N, self.fit_type)

                    if chi_down > lim:
                        break
                down_int += 1

            up_int = 1
            while up_int < cap:
                if index+up_int >= len(array_signs):
                    break
                elif (index + up_int) in set(tested_indices):
                    chi_up = 0
                    pass
                else:
                    signs = array_signs[index+up_int]
                    tested_indices.append(index + up_int)
                    fit = qp_class(
                        x, y, self.N, signs, mid_point,
                        self.model_type, self.cvxopt_maxiter,
                        self.all_output,
                        self.ifp_list, self.initial_params,
                        self.warnings, self.constraints, self.new_basis)
                    chi_up = fit.chi_squared
                    chi_squared.append(fit.chi_squared)
                    tested_signs.append(signs)
                    parameters.append(fit.parameters)

                    if self.all_output is True:
                        print(
                            '--------------------------------------' +
                            '--------------')
                        print('Polynomial Order:', self.N)
                        if self.ifp_list is not None:
                            print('Number of Constrained Derivatives:',
                                self.N-self.constraints-len(self.ifp_list))
                        else:
                            print('Number of Constrained Derivatives:',
                                self.N-self.constraints)
                        print('Signs :', signs)
                        print('Objective Function Value:', fit.chi_squared)
                        print('Parameters:', fit.parameters.T)
                        print('Method:', self.fit_type)
                        print('Model:', self.model_type)
                        print('Constraints: m >=', self.constraints)
                        if self.ifp_list is None:
                            print(
                                'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                                fit.ifp_dict)
                        if self.ifp_list is not None:
                            print(
                                'Inflection Point Derivatives:',
                                self.ifp_list)
                            print(
                                'Inflection Points Used? (0 signifies' +
                                'Yes\n in derivative order "i"):', fit.ifp_dict)
                        print(
                            '--------------------------------------' +
                            '--------------')
                    if self.data_save is True:
                        save(
                            self.base_dir, fit.parameters, fit.chi_squared,
                            signs, self.N, self.fit_type)

                    if chi_up > lim:
                        break
                up_int += 1

            for i in range(len(chi_squared)):
                if chi_squared[i] == min(chi_squared):
                    Optimum_params = parameters[i]
                    Optimum_sign_combination = tested_signs[i]
                    Optimum_chi_squared = chi_squared[i]

            if self.model_type == 'loglog_polynomial':
                y_fit = Models_class(
                    Optimum_params, np.log10(x/x[mid_point]), np.log10(y),
                    self.N, mid_point,
                    self.model_type, self.new_basis).y_sum
                der = derivative_class(
                    np.log10(x/x[mid_point]), np.log10(y), Optimum_params,
                    self.N,
                    mid_point, self.model_type, self.ifp_list, self.warnings,
                    self.constraints, self.new_basis)
                derivatives, Optimum_ifp_dict = der.derivatives, der.ifp_dict
            else:
                y_fit = Models_class(
                    Optimum_params, x, y, self.N, mid_point,
                    self.model_type, self.new_basis).y_sum
                der = derivative_class(
                    x, y, Optimum_params, self.N,
                    mid_point, self.model_type, self.ifp_list, self.warnings,
                    self.constraints, self.new_basis)
                derivatives, Optimum_ifp_dict = der.derivatives, der.ifp_dict

            end = time.time()

            print(
                '######################################################' +
                '#######')
            print(
                '----------------------OPTIMUM RESULT--------------------' +
                '-----')
            print('Time:', end-start)
            print('Polynomial Order:', self.N)
            if self.ifp_list is not None:
                print('Number of Constrained Derivatives:',
                    self.N-self.constraints-len(self.ifp_list))
            else:
                print('Number of Constrained Derivatives:',
                    self.N-self.constraints)
            print('Signs :', Optimum_sign_combination)
            print('Objective Function Value:', Optimum_chi_squared)
            print('Parameters:', Optimum_params.T)
            print('Method:', self.fit_type)
            print('Model:', self.model_type)
            print('Constraints: m >=', self.constraints)
            if self.ifp_list is None:
                print(
                    'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                    Optimum_ifp_dict)
            if self.ifp_list is not None:
                print('Inflection Point Derivatives:', self.ifp_list)
                print(
                    'Inflection Points Used? (0 signifies Yes\n in derivative order "i"):',
                    Optimum_ifp_dict)
            print(
                '----------------------------------------------------' +
                '---------')
            print(
                '####################################################' +
                '#########')

            save_optimum(
                self.base_dir, end-start, self.N,
                Optimum_sign_combination, Optimum_chi_squared,
                Optimum_params, self.fit_type, self.model_type,
                self.ifp_list, Optimum_ifp_dict, self.constraints)

            return y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_ifp_dict

        mid_point = len(self.x)//2
        if self.fit_type == 'qp':
            y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_ifp_dict = \
                qp(self.x, self.y, mid_point)
        if self.fit_type == 'qp-sign_flipping':
            y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_ifp_dict = \
                qp_sign_flipping(self.x, self.y, mid_point)

        rms = (np.sqrt(np.sum((self.y-y_fit)**2)/len(self.y)))


        return y_fit, Optimum_sign_combination, Optimum_params, derivatives, \
            Optimum_chi_squared, rms, Optimum_ifp_dict

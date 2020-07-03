
"""
*smooth*, as demonstrated in the examples section,
is used to call the fitting routine. There are a number
of :math:`{^{**}}` kwargs that can be assigned to the function which change how
the fit is performed, the model that is fit and various other attributes.
These are detailed below.

"""

from maxsmooth.qp import qp_class
from maxsmooth.Models import Models_class
from maxsmooth.derivatives import derivative_class
from maxsmooth.Data_save import save, save_optimum
from itertools import product
import warnings
import numpy as np
import time
import os
import shutil


class smooth(object):

    r"""

    **Parameters:**

        x: **numpy.array**
            | The x data points for the set being fitted.

        y: **numpy.array**
            | The y data points for fitting.

        N: **int**
            | The number of terms in the DCF.

    **Kwargs:**

        fit_type: **Default = 'qp-sign_flipping'**
            | This kwarg allows the user to
                switch between sampling the available discrete sign spaces
                (default) or testing all sign combinations on the derivatives
                which can be accessed by setting to 'qp'.

        model_type: **Default = 'difference_polynomial'**
            | Allows the user to
                access default Derivative Constrained Functions built into the
                software. Available options include the default, 'polynomial',
                'normalised_polynomial', 'legendre', 'log_polynomial',
                'loglog_polynomial' and 'exponential'. For more details on the
                functional form of the built in basis see the ``maxsmooth``
                paper.

        **pivot_point: Default = len(x)//2 otherwise an integer between**
        **-len(x) and len(x)**
            | Some of the built in
                models rely on pivot points in the data sets which by defualt
                is set as the middle index. This can be altered via
                this kwarg which can occasionally lead to a better quality fit.

        base_dir: **Default = 'Fitted_Output/'**
            | The location of the outputted
                data from ``maxsmooth``. This must be a string and end in '/'.
                If the file does not exist then ``maxsmooth`` will create it.
                By default the only outputted data is a summary of the best
                fit but additional data can be recorded by setting the keyword
                argument 'data_save = True'.

        data_save: **Default = False**
            | By setting this to True the algorithm
                will save every tested set of parameters, signs and objective
                function evaluations into files in base_dir. Theses files will
                be over written on repeated runs but they are needed to run the
                'chidist_plotter'.

        all_output: **Default = False**
            | If set to True this outputs to the
                terminal every fit performed by the algorithm. By default the
                only output is the optimal solution once the code is finished.

        cvxopt_maxiter: **Default = 10000 else integer**
            | This shouldn't need
                changing for most problems however if ``CVXOPT`` fails with a
                'maxiters reached' error message this can be increased.
                Doing so arbitrarily will however increase the run time of
                ``maxsmooth``.

        initial_params: **Default = None else list of length N**
            | Allows the user
                to overwrite the default initial parameters used by ``CVXOPT``.

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

        The following Kwargs can be used by the user to define their own basis
        function and will overwrite the 'model_type' kwarg.

        **basis_function: Default = None else function with parameters**
        **(x, y, pivot_point, N)**
            | This is a function of basis functions
                for the quadratic programming. The variable pivot_point is the
                index at the middle of the datasets x and y by default but can
                be adjusted.

        **model: Default = None else function with parameters**
        **(x, y, pivot_point, N, params)**
            | This is
                a user defined function describing the model to be fitted to
                the data.

        **der_pres: Default = None else function with parameters**
        **(m, x, y, N, pivot_point)**
            | This function describes the prefactors on the
                mth order derivative used in defining the constraint.

        **derivatives: Default = None else function with parameters**
        **(m, x, y, N, pivot_point, params)**
            | User defined function describing the mth
                order derivative used to check that conditions are being met.

        **args: Default = None else list**
            | Extra arguments for `smooth`
                to pass to the functions detailed above.

    **Output**

        .y_fit: **numpy.array**
            | The fitted array of y data from smooth().

        .optimum_chi: **float**
            | The optimum :math:`{\chi^2}` value for the fit calculated by,

            .. math::

                {X^2=\sum(y-y_{fit})^2}.

        .optimum_params: **numpy.array**
            | The set of parameters corresponding to the optimum fit.

        .rms: **float**
            | The rms value of the residuals :math:`{y_{res}=y-y_{fit}}`
                calculated by,

            .. math::

                {rms=\sqrt{\frac{\sum(y-y_{fit})^2}{n}}}

            where :math:`n` is the number of data points.

        .derivatives: **numpy.array**
            | The :math:`m^{th}` order derivatives.

        .optimum_signs: **numpy.array**
            | The sign combinations corresponding to the
                optimal result. The nature of the constraint means that a
                negative ``maxsmooth`` sign implies a positive :math:`{m^{th}}`
                order derivative and visa versa.

    """

    def __init__(self, x, y, N, **kwargs):
        self.x = x
        self.y = y

        self.N = N
        if self.N % 1 != 0:
            raise ValueError('N must be an integer or whole number float.')

        for keys, values in kwargs.items():
            if keys not in set(
                    ['fit_type', 'model_type', 'base_dir',
                        'all_output', 'cvxopt_maxiter', 'zero_crossings',
                        'data_save',
                        'constraints', 'chi_squared_limit', 'cap',
                        'initial_params', 'basis_functions',
                        'der_pres', 'model',
                        'derivatives', 'args', 'pivot_point']):
                raise KeyError("Unexpected keyword argument in smooth().")

        self.fit_type = kwargs.pop('fit_type', 'qp-sign_flipping')
        if self.fit_type not in set(['qp', 'qp-sign_flipping']):
            raise KeyError(
                "Invalid 'fit_type'. Valid entries include 'qp'\n" +
                "'qp-sign_flipping'")

        self.pivot_point = kwargs.pop('pivot_point', len(self.x)//2)
        if type(self.pivot_point) is not int:
            raise TypeError('Pivot point is not an integer index.')
        elif self.pivot_point >= len(self.x) or \
                self.pivot_point < -len(self.x):
            raise ValueError(
                'Pivot point must be in the range -len(x) - len(x).')

        self.base_dir = kwargs.pop('base_dir', 'Fitted_Output/')
        if type(self.base_dir) is not str:
            raise KeyError("'base_dir' must be a string ending in '/'.")
        elif self.base_dir.endswith('/') is False:
            raise KeyError("'base_dir' must end in '/'.")

        self.model_type = kwargs.pop('model_type', 'difference_polynomial')
        if self.model_type not in set(
                ['normalised_polynomial', 'polynomial',
                    'log_polynomial', 'loglog_polynomial',
                    'difference_polynomial',
                    'exponential', 'legendre']):
            raise KeyError(
                "Invalid 'model_type'. See documentation for built" +
                "in models.")

        self.cvxopt_maxiter = kwargs.pop('cvxopt_maxiter', 10000)
        if type(self.cvxopt_maxiter) is not int:
            raise ValueError("'cvxopt_maxiter' is not integer.")

        self.all_output = kwargs.pop('all_output', False)
        self.data_save = kwargs.pop('data_save', False)
        boolean_kwargs = [self.data_save, self.all_output]
        for i in range(len(boolean_kwargs)):
            if type(boolean_kwargs[i]) is not bool:
                raise TypeError(
                    "Boolean keyword argument with value "
                    + str(boolean_kwargs[i]) +
                    " is not True or False.")

        self.constraints = kwargs.pop('constraints', 2)
        if type(self.constraints) is not int:
            raise TypeError("'constraints' is not an integer")
        if self.constraints > self.N-1:
            raise ValueError(
                "'constraints' exceeds the number of derivatives.")

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

        self.initial_params = kwargs.pop('initial_params', None)
        if self.initial_params is not None and len(self.initial_params) \
                != self.N:
            raise ValueError(
                "Initial Parameters is not equal to the number" +
                "of terms in the polynomial, N.")
        if self.initial_params is not None and len(self.initial_params) \
                == self.N:
            for i in range(len(self.initial_params)):
                if type(self.initial_params[i]) is not int:
                    if type(self.initial_params[i]) is not float:
                        raise ValueError(
                            'One or more initial' +
                            'parameters is not numeric.')

        self.basis_functions = kwargs.pop('basis_functions', None)
        self.der_pres = kwargs.pop('der_pres', None)
        self.model = kwargs.pop('model', None)
        self.derivatives_function = kwargs.pop('derivatives', None)
        self.args = kwargs.pop('args', None)

        self.new_basis = {
            'basis_function':
            self.basis_functions, 'der_pres': self.der_pres,
            'derivatives_function': self.derivatives_function,
            'model': self.model, 'args': self.args}
        if np.all([value is None for value in self.new_basis.values()]):
            pass
        else:
            count = 0
            for key, value in self.new_basis.items():
                if value is None and key != 'args':
                    raise KeyError(
                        'Attempt to change basis functions failed.' +
                        ' One or more functions not defined.' +
                        ' Please consult documentation.')
                if value is None and key == 'args':
                    warnings.warn(
                        'Warning: No additional arguments passed to' +
                        ' new basis functions')
                count += 1
            if count == len(self.new_basis):
                self.model_type = 'user_defined'

        self.y_fit, self.optimum_signs, self.optimum_params, \
            self.derivatives,\
            self.optimum_chi, self.rms, self.optimum_zc_dict \
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

        def qp(x, y, pivot_point):  # Testing all signs
            print(
                '######################################################' +
                '#######')
            start = time.time()

            if self.zero_crossings is not None:
                signs = signs_array([1]*(
                    self.N-self.constraints-len(self.zero_crossings)))
            else:
                signs = signs_array([1]*(self.N-self.constraints))

            params, chi_squared, zc_dict, passed_signs = [], [], [], []
            append_params, append_chi, append_zc_dict, append_passed_signs = \
                params.append, chi_squared.append, zc_dict.append, \
                passed_signs.append
            for j in range(signs.shape[0]):
                fit = qp_class(
                    x, y, self.N, signs[j, :], pivot_point,
                    self.model_type, self.cvxopt_maxiter,
                    self.all_output, self.zero_crossings,
                    self.initial_params, self.constraints, self.new_basis)

                if self.all_output is True:
                    print(
                        '-----------------------------------------------' +
                        '-----')
                    print('Polynomial Order:', self.N)
                    if self.zero_crossings is not None:
                        print(
                            'Number of Constrained Derivatives:',
                            self.N-self.constraints-len(self.zero_crossings))
                    else:
                        print(
                            'Number of Constrained Derivatives:',
                            self.N-self.constraints)
                    print('Signs :', signs[j, :])
                    print('Objective Function Value:', fit.chi_squared)
                    print('Parameters:', (fit.parameters).T)
                    print('Method:', self.fit_type)
                    print('Model:', self.model_type)
                    print('Constraints: m >=', self.constraints)
                    if self.zero_crossings is None:
                        print(
                            'Zero Crossings Used?' +
                            ' (0 signifies Yes\n in derivative order "i"):',
                            fit.zc_dict)
                    if self.zero_crossings is not None:
                        print(
                            'Zero Crossing Derivatives:', self.zero_crossings)
                        print(
                            'Zero Crossings Used?' +
                            ' (0 signifies Yes\n in derivative order "i"):',
                            fit.zc_dict)
                    print(
                        '-----------------------------------------------' +
                        '-----')

                append_params(fit.parameters)
                append_chi(fit.chi_squared)
                append_zc_dict(fit.zc_dict)
                append_passed_signs(signs[j, :])
                if self.data_save is True:
                    save(
                        self.base_dir, fit.parameters, fit.chi_squared,
                        signs[j, :], self.N, self.fit_type)

            params, chi_squared, zc_dict, passed_signs = np.array(params), \
                np.array(chi_squared), np.array(zc_dict), \
                np.array(passed_signs)

            Optimum_chi_squared = chi_squared.min()
            for f in range(len(chi_squared)):
                if chi_squared[f] == chi_squared.min():
                    Optimum_params = params[f, :]
                    Optimum_sign_combination = passed_signs[f, :]

            y_fit = Models_class(
                Optimum_params, x, y, self.N, pivot_point,
                self.model_type, self.new_basis).y_sum
            der = derivative_class(
                x, y, Optimum_params, self.N,
                pivot_point, self.model_type, self.zero_crossings,
                self.constraints, self.new_basis)
            derivatives, Optimum_zc_dict = der.derivatives, der.zc_dict

            end = time.time()

            print(
                '######################################################' +
                '#######')
            print(
                '----------------------OPTIMUM RESULT--------------------' +
                '-----')
            print('Time:', end-start)
            print('Polynomial Order:', self.N)
            if self.zero_crossings is not None:
                print(
                    'Number of Constrained Derivatives:',
                    self.N-self.constraints-len(self.zero_crossings))
            else:
                print(
                    'Number of Constrained Derivatives:',
                    self.N-self.constraints)
            print('Signs :', Optimum_sign_combination)
            print('Objective Function Value:', Optimum_chi_squared)
            print('Parameters:', Optimum_params.T)
            print('Method:', self.fit_type)
            print('Model:', self.model_type)
            print('Constraints: m >=', self.constraints)
            if self.zero_crossings is None:
                print(
                    'Zero Crossings Used?' +
                    ' (0 signifies Yes\n in derivative order "i"):',
                    Optimum_zc_dict)
            if self.zero_crossings is not None:
                print('Zero Crossing Derivatives:', self.zero_crossings)
                print(
                    'Zero Crossings Used?' +
                    ' (0 signifies Yes\n in derivative order "i"):',
                    Optimum_zc_dict)
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
                self.zero_crossings, Optimum_zc_dict, self.constraints)

            return y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_zc_dict

        def qp_sign_flipping(x, y, pivot_point):  # Sign Sampling
            print(
                '######################################################' +
                '#######')
            start = time.time()

            if self.zero_crossings is not None:
                array_signs = signs_array([1]*(
                    self.N-self.constraints-len(self.zero_crossings)))
            else:
                array_signs = signs_array([1]*(self.N-self.constraints))

            r = np.random.randint(0, len(array_signs), 1)
            signs = array_signs[r][0]

            tested_indices = []
            chi_squared = []
            parameters = []
            tested_signs = []
            for i in range(len(array_signs)):
                if i == r:
                    tested_indices.append(i)
            fit = qp_class(
                x, y, self.N, signs, pivot_point,
                self.model_type, self.cvxopt_maxiter,
                self.all_output, self.zero_crossings,
                self.initial_params, self.constraints, self.new_basis)
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
                if self.zero_crossings is not None:
                    print(
                        'Number of Constrained Derivatives:',
                        self.N-self.constraints-len(self.zero_crossings))
                else:
                    print(
                        'Number of Constrained Derivatives:',
                        self.N-self.constraints)
                print('Signs :', signs)
                print('Objective Function Value:', chi_squared_old)
                print('Parameters:', (fit.parameters).T)
                print('Method:', self.fit_type)
                print('Model:', self.model_type)
                print('Constraints: m >=', self.constraints)
                if self.zero_crossings is None:
                    print(
                        'Zero Crossings Used?' +
                        ' (0 signifies Yes\n in derivative order "i"):',
                        fit.zc_dict)
                if self.zero_crossings is not None:
                    print(
                        'Zero Crossing Derivatives:',
                        self.zero_crossings)
                    print(
                        'Zero Crossings Used? (0 signifies' +
                        'Yes\n in derivative order "i"):', fit.zc_dict)
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
                            x, y, self.N, signs, pivot_point,
                            self.model_type, self.cvxopt_maxiter,
                            self.all_output,
                            self.zero_crossings, self.initial_params,
                            self.constraints, self.new_basis)
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
                                if self.zero_crossings is not None:
                                    print(
                                        'Number of Constrained Derivatives:',
                                        self.N - self.constraints -
                                        len(self.zero_crossings))
                                else:
                                    print(
                                        'Number of Constrained Derivatives:',
                                        self.N-self.constraints)
                                print('Signs :', signs)
                                print(
                                    'Objective Function Value:',
                                    fit.chi_squared)
                                print('Parameters:', fit.parameters.T)
                                print('Method:', self.fit_type)
                                print('Model:', self.model_type)
                                print('Constraints: m >=', self.constraints)
                                if self.zero_crossings is None:
                                    print(
                                        'Zero Crossings Used?' +
                                        ' (0 signifies Yes\n in derivative' +
                                        ' order "i"):',
                                        fit.zc_dict)
                                if self.zero_crossings is not None:
                                    print(
                                        'Zero Crossing Derivatives:',
                                        self.zero_crossings)
                                    print(
                                        'Zero Crossings Used?' +
                                        ' (0 signifies' +
                                        'Yes\n in derivative order "i"):',
                                        fit.zc_dict)
                                print(
                                    '--------------------------------------' +
                                    '--------------')
                            if self.data_save is True:
                                save(
                                    self.base_dir, fit.parameters,
                                    fit.chi_squared,
                                    signs, self.N, self.fit_type)

                            break
                    if h == sign_transform.shape[0] - 1:
                        chi_squared_new = chi_squared_old
                        break

            if self.data_save is True:
                np.save(self.base_dir + str(self.N) +
                '_'+self.fit_type+'_minimum_chi_post_descent.npy',
                min(chi_squared))

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
                        x, y, self.N, signs, pivot_point,
                        self.model_type, self.cvxopt_maxiter,
                        self.all_output,
                        self.zero_crossings, self.initial_params,
                        self.constraints, self.new_basis)
                    chi_down = fit.chi_squared
                    chi_squared.append(fit.chi_squared)
                    tested_signs.append(signs)
                    parameters.append(fit.parameters)

                    if self.all_output is True:
                        print(
                            '--------------------------------------' +
                            '--------------')
                        print('Polynomial Order:', self.N)
                        if self.zero_crossings is not None:
                            print(
                                'Number of Constrained Derivatives:',
                                self.N-self.constraints -
                                len(self.zero_crossings))
                        else:
                            print(
                                'Number of Constrained Derivatives:',
                                self.N-self.constraints)
                        print('Signs :', signs)
                        print('Objective Function Value:', fit.chi_squared)
                        print('Parameters:', fit.parameters.T)
                        print('Method:', self.fit_type)
                        print('Model:', self.model_type)
                        print('Constraints: m >=', self.constraints)
                        if self.zero_crossings is None:
                            print(
                                'Zero Crossings Used?' +
                                ' (0 signifies Yes\n in derivative' +
                                ' order "i"):',
                                fit.zc_dict)
                        if self.zero_crossings is not None:
                            print(
                                'Zero Crossing Derivatives:',
                                self.zero_crossings)
                            print(
                                'Zero Crossings Used? (0 signifies' +
                                'Yes\n in derivative order "i"):',
                                fit.zc_dict)
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
                        x, y, self.N, signs, pivot_point,
                        self.model_type, self.cvxopt_maxiter,
                        self.all_output,
                        self.zero_crossings, self.initial_params,
                        self.constraints, self.new_basis)
                    chi_up = fit.chi_squared
                    chi_squared.append(fit.chi_squared)
                    tested_signs.append(signs)
                    parameters.append(fit.parameters)

                    if self.all_output is True:
                        print(
                            '--------------------------------------' +
                            '--------------')
                        print('Polynomial Order:', self.N)
                        if self.zero_crossings is not None:
                            print(
                                'Number of Constrained Derivatives:',
                                self.N-self.constraints -
                                len(self.zero_crossings))
                        else:
                            print(
                                'Number of Constrained Derivatives:',
                                self.N-self.constraints)
                        print('Signs :', signs)
                        print('Objective Function Value:', fit.chi_squared)
                        print('Parameters:', fit.parameters.T)
                        print('Method:', self.fit_type)
                        print('Model:', self.model_type)
                        print('Constraints: m >=', self.constraints)
                        if self.zero_crossings is None:
                            print(
                                'Zero Crossings Used?' +
                                ' (0 signifies Yes\n in derivative' +
                                ' order "i"):',
                                fit.zc_dict)
                        if self.zero_crossings is not None:
                            print(
                                'Zero Crossing Derivatives:',
                                self.zero_crossings)
                            print(
                                'Zero Crossings Used? (0 signifies' +
                                'Yes\n in derivative order "i"):',
                                fit.zc_dict)
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

            y_fit = Models_class(
                Optimum_params, x, y, self.N, pivot_point,
                self.model_type, self.new_basis).y_sum
            der = derivative_class(
                x, y, Optimum_params, self.N,
                pivot_point, self.model_type, self.zero_crossings,
                self.constraints, self.new_basis)
            derivatives, Optimum_zc_dict = der.derivatives, der.zc_dict

            end = time.time()

            print(
                '######################################################' +
                '#######')
            print(
                '----------------------OPTIMUM RESULT--------------------' +
                '-----')
            print('Time:', end-start)
            print('Polynomial Order:', self.N)
            if self.zero_crossings is not None:
                print(
                    'Number of Constrained Derivatives:',
                    self.N-self.constraints-len(self.zero_crossings))
            else:
                print(
                    'Number of Constrained Derivatives:',
                    self.N-self.constraints)
            print('Signs :', Optimum_sign_combination)
            print('Objective Function Value:', Optimum_chi_squared)
            print('Parameters:', Optimum_params.T)
            print('Method:', self.fit_type)
            print('Model:', self.model_type)
            print('Constraints: m >=', self.constraints)
            if self.zero_crossings is None:
                print(
                    'Zero Crossings Used?' +
                    ' (0 signifies Yes\n in derivative order "i"):',
                    Optimum_zc_dict)
            if self.zero_crossings is not None:
                print('Zero Crossing Derivatives:', self.zero_crossings)
                print(
                    'Zero Crossings Used?' +
                    ' (0 signifies Yes\n in derivative order "i"):',
                    Optimum_zc_dict)
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
                self.zero_crossings, Optimum_zc_dict, self.constraints)

            return y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_zc_dict

        if self.fit_type == 'qp':
            y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_zc_dict = \
                qp(self.x, self.y, self.pivot_point)
        if self.fit_type == 'qp-sign_flipping':
            y_fit, derivatives, Optimum_chi_squared, Optimum_params, \
                Optimum_sign_combination, Optimum_zc_dict = \
                qp_sign_flipping(self.x, self.y, self.pivot_point)

        rms = (np.sqrt(np.sum((self.y-y_fit)**2)/len(self.y)))

        return y_fit, Optimum_sign_combination, Optimum_params, derivatives, \
            Optimum_chi_squared, rms, Optimum_zc_dict

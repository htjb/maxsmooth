from maxsmooth.derivatives import derivative_class
from maxsmooth.Models import Models_class
from cvxopt import matrix, solvers
import numpy as np
import sys
import warnings

warnings.simplefilter('always', UserWarning)


class qp_class(object):
    def __init__(
            self, x, y, N, signs, mid_point, model_type, cvxopt_maxiter,
            filtering, all_output, ifp, ifp_list, initial_params,
            basis_functions, data_matrix, derivative_pres, model,
            derivatives_function, args):
        self.x = x
        self.y = y
        self.N = N
        self.signs = signs
        self.mid_point = mid_point
        self.model_type = model_type
        self.cvxopt_maxiter = cvxopt_maxiter
        self.filtering = filtering
        self.all_output = all_output
        self.ifp = ifp
        self.ifp_list = ifp_list
        self.initial_params = initial_params
        self.basis_functions = basis_functions
        self.data_matrix = data_matrix
        self.derivative_pres = derivative_pres
        self.model = model
        self.derivatives_function = derivatives_function
        self.args = args
        self.parameters, self.chi_squared, self.pass_fail = self.fit()

    def fit(self):

        solvers.options['maxiters'] = self.cvxopt_maxiter
        solvers.options['show_progress'] = False

        def constraint_prefactors(m):
            # Derivative prefactors on parameters.
            derivatives = []
            for i in range(self.N):
                if i <= m - 1:
                    derivatives.append([0]*len(self.x))
            for i in range(self.N-m):
                if i < (self.N-m) or i == (self.N-m):
                    if self.derivative_pres is None:
                        if self.model_type == 'normalised_polynomial':
                            mth_order_derivative_term = (
                                self.y[self.mid_point] /
                                self.x[self.mid_point]) \
                                * np.math.factorial(m + i) \
                                / np.math.factorial(i) * \
                                (self.x)**i/(self.x[self.mid_point])**(i + 1)
                            derivatives.append(mth_order_derivative_term)
                        if self.model_type == 'polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i) * (self.x)**i
                            derivatives.append(mth_order_derivative_term)
                        if self.model_type == 'MSF_2017_polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i) * (
                                self.x - self.x[self.mid_point])**i
                            derivatives.append(mth_order_derivative_term)
                        if self.model_type == 'logarithmic_polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i)*(np.log10(self.x))**i
                            derivatives.append(mth_order_derivative_term)
                    if self.derivative_pres is not None:
                        if self.args is None:
                            mth_order_derivative_term = self.derivative_pres(
                                m, i, self.x, self.y, self.mid_point)
                        if self.args is not None:
                            mth_order_derivative_term = self.derivative_pres(
                                m, i, self.x, self.y, self.mid_point,
                                *self.args)
                        derivatives.append(mth_order_derivative_term)
            derivatives = np.array(derivatives).astype(np.double)
            derivatives = matrix(derivatives)
            derivatives = derivatives.T
            return derivatives

        m = np.arange(0, self.N, 1)
        derivatives = []
        signs = matrix(self.signs)
        for i in range(len(m)):
            if m[i] >= 2:
                derivative_prefactors = constraint_prefactors(m[i])
                if derivative_prefactors != []:
                    derivatives.append(signs[i - 2] * derivative_prefactors)
        G = matrix(derivatives)  # Array of derivative prefactors for all m>=2

        if self.basis_functions is None:
            A = np.empty([len(self.x), self.N])
            for h in range(len(self.x)):
                for i in range(self.N):
                    if self.model_type == 'normalised_polynomial':
                        A[h, i] = self.y[self.mid_point] * (
                            self.x[h] / self.x[self.mid_point])**i
                    if self.model_type == 'polynomial':
                        A[h, i] = (self.x[h])**i
                    if self.model_type == 'MSF_2017_polynomial':
                        A[h, i] = (self.x[h]-self.x[self.mid_point])**i
                    if self.model_type == 'logarithmic_polynomial':
                        A[h, i] = np.log10(self.x[h])**i
            A = matrix(A)
        if self.basis_functions is not None:
            if self.args is None:
                A = self.basis_functions(
                    self.x, self.y, self.mid_point, self.N)
            if self.args is not None:
                A = self.basis_functions(
                    self.x, self.y, self.mid_point, self.N, *self.args)

        if self.data_matrix is None:
            if self.model_type == 'logarithmic_polynomial':
                b = matrix(np.log10(self.y), (len(self.y), 1), 'd')
            else:
                b = matrix(self.y, (len(self.y), 1), 'd')
        if self.data_matrix is not None:
            b = self.data_matrix

        if self.ifp is False:
            h = matrix(-1e-7, ((self.N-2)*len(self.x), 1), 'd')
        if self.ifp is True:
            if self.ifp_list == 'None':
                print(
                    'ERROR: setting.ifp set to True but no derivatives' +
                    ' selected. Please state which derivatives you would' +
                    ' like to allow inflection points in by setting' +
                    ' ifp_list to a list of derivative orders(see' +
                    ' settings.py for more information). ')
                sys.exit(1)
            else:
                h_ifp = []
                ifp_list = np.array(self.ifp_list)
                for i in range(self.N-2):
                    if np.any(ifp_list-2 == i):
                        h_ifp.append([1e20]*(len(self.x)))
                    else:
                        h_ifp.append([-1e-7]*(len(self.x)))
                h_ifp = np.array(h_ifp)
                # Correcting array format to transform to cvxopt matrix.
                for i in range(h_ifp.shape[0]):
                    if i == 0:
                        hifp = np.array(h_ifp[i])
                    else:
                        hifp = np.concatenate([hifp, h_ifp[i]])
                h = matrix(hifp.T)

        P = A.T*A
        q = -A.T*b

        if self.initial_params is None:
            if self.model_type == 'logarithmic_polynomial':
                params0 = [(np.log10(self.y[-1])-np.log10(self.y[0]))/2] * \
                    (self.N)
            else:
                params0 = [(self.y[-1]-self.y[0])/2]*(self.N)
        if self.initial_params is not None:
            params0 = self.initial_params

        qpfit = solvers.coneqp(P, q, G, h, initvals=params0)

        parameters = qpfit['x']
        y = Models_class(
            parameters, self.x, self.y, self.N, self.mid_point,
            self.model_type, self.model, self.args).y_sum
        der = derivative_class(
            self.x, self.y, parameters, self.N, self.signs, self.mid_point,
            self.model_type, self.ifp, self.derivatives_function, self.args)
        pass_fail = der.pass_fail

        if 'unknown' in qpfit['status']:
            if qpfit['iterations'] == self.cvxopt_maxiter:
                print(
                    'ERROR: "Maximum number of iterations reached in' +
                    ' cvxopt routine." Increase value of' +
                    ' setting.cvxopt_maxiter')
                sys.exit(1)
            else:
                if self.filtering is False:
                    print(
                        'ERROR: "Terminated (singular KKT matrix)".' +
                        ' Problem is infeasible with this sign combination' +
                        ' set setting.filtering==True to filter out this ' +
                        ' and any other incidences.')
                    sys.exit(1)
                if self.filtering is True:
                    pass_fail = []
                    warnings.warn(
                        '"Terminated (singular KKT matrix)".' +
                        ' Problem infeasable with the following sign' +
                        ' combination, therefore sign combination will' +
                        ' be excluded when identifying the best solution.',
                        stacklevel=2)

        chi_squared = np.sum((self.y-y)**2)
        parameters = np.array(parameters)

        return parameters, chi_squared, pass_fail

from maxsmooth.derivatives import derivative_class
from maxsmooth.Models import Models_class
from cvxopt import matrix, solvers
import numpy as np
import warnings
from scipy.special import legendre, lpmv

warnings.simplefilter('always', UserWarning)


class qp_class(object):
    def __init__(
            self, x, y, N, signs, pivot_point, model_type, cvxopt_maxiter,
            all_output, zero_crossings, initial_params,
            constraints, new_basis):
        self.model_type = model_type
        self.pivot_point = pivot_point
        self.y = y
        self.x = x
        self.N = N
        self.signs = signs
        self.cvxopt_maxiter = cvxopt_maxiter
        self.all_output = all_output
        self.zero_crossings = zero_crossings
        self.initial_params = initial_params
        self.basis_functions = new_basis['basis_function']
        self.derivative_pres = new_basis['der_pres']
        self.model = new_basis['model']
        self.derivatives_function = new_basis['derivatives_function']
        self.args = new_basis['args']
        self.new_basis = new_basis
        self.constraints = constraints
        self.parameters, self.chi_squared, self.zc_dict = self.fit()

    def fit(self):

        solvers.options['maxiters'] = self.cvxopt_maxiter
        solvers.options['show_progress'] = False

        def constraint_prefactors(m):
            # Derivative prefactors on parameters
            if self.derivative_pres is None:
                if np.any(self.model_type != ['legendre', 'exponential']):
                    derivatives = []
                    for i in range(self.N):
                        if i <= m - 1:
                            derivatives.append([0]*len(self.x))
                    for i in range(self.N-m):
                        if self.model_type == 'normalised_polynomial':
                            mth_order_derivative_term = (
                                self.y[self.pivot_point] /
                                self.x[self.pivot_point]) \
                                * np.math.factorial(m + i) \
                                / np.math.factorial(i) * \
                                (self.x)**i/(self.x[self.pivot_point])**(i + 1)
                            derivatives.append(mth_order_derivative_term)
                        if self.model_type == 'polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i) * (self.x)**i
                            derivatives.append(mth_order_derivative_term)
                        if self.model_type == 'log_polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i) * \
                                np.log10(self.x/self.x[self.pivot_point])**i
                            derivatives.append(mth_order_derivative_term)
                        if self.model_type == 'loglog_polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i) * np.log10(self.x)**i
                            derivatives.append(mth_order_derivative_term)
                        if self.model_type == 'difference_polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i) * (
                                self.x - self.x[self.pivot_point])**i
                            derivatives.append(mth_order_derivative_term)

            if self.derivative_pres is not None:
                if self.args is None:
                    derivatives = self.derivative_pres(
                        m, self.x, self.y, self.N, self.pivot_point)
                if self.args is not None:
                    derivatives = self.derivative_pres(
                        m, self.x, self.y, self.N, self.pivot_point,
                        *self.args)

            if self.model_type == 'legendre':
                interval = np.linspace(-0.999, 0.999, len(self.x))
                alps = []
                for i in range(self.N):
                    alps.append(lpmv(m, i, interval))
                alps = np.array(alps)
                derivatives = []
                for h in range(len(alps)):
                    derivatives.append(
                        ((alps[h, :]*(-1)**(m))/(1-interval**2)**(m/2)))
                derivatives = np.array(derivatives)
            if self.model_type == 'exponential':
                derivatives = np.empty([self.N, len(self.x)])
                for i in range(self.N):
                    for h in range(len(self.x)):
                        derivatives[i, h] = \
                            self.y[self.pivot_point] * \
                            (np.exp(-i*self.x[h]/self.x[self.pivot_point])) * \
                            (-i/self.x[self.pivot_point])**m
                derivatives = np.array(derivatives)

            derivatives = np.array(derivatives).astype(np.double)
            derivatives = matrix(derivatives)
            if derivatives.size == (len(self.x), self.N):
                pass
            else:
                derivatives = derivatives.T
            return derivatives

        m = np.arange(0, self.N, 1)
        derivatives = []
        signs = matrix(self.signs)
        for i in range(len(m)):
            if m[i] >= self.constraints:
                if self.zero_crossings is not None:
                    if m[i] not in set(self.zero_crossings):
                        derivative_prefactors = constraint_prefactors(m[i])
                        if derivative_prefactors != []:
                            derivatives.append(derivative_prefactors)
                else:
                    derivative_prefactors = constraint_prefactors(m[i])
                    if derivative_prefactors != []:
                        derivatives.append(derivative_prefactors)

        for i in range(len(derivatives)):
            derivatives[i] *= signs[i]

        G = matrix(derivatives)

        if self.basis_functions is None:
            phi = np.empty([len(self.x), self.N])
            if self.model_type != 'legendre':
                for h in range(len(self.x)):
                    for i in range(self.N):
                        if self.model_type == 'normalised_polynomial':
                            phi[h, i] = self.y[self.pivot_point] * (
                                self.x[h] / self.x[self.pivot_point])**i
                        if self.model_type == 'polynomial':
                            phi[h, i] = (self.x[h])**i
                        if self.model_type == 'log_polynomial':
                            phi[h, i] = \
                                np.log10(self.x[h]/self.x[self.pivot_point])**i
                        if self.model_type == 'loglog_polynomial':
                            phi[h, i] = np.log10(self.x[h])**i
                        if self.model_type == 'difference_polynomial':
                            phi[h, i] = (self.x[h]-self.x[self.pivot_point])**i
                        if self.model_type == 'exponential':
                            phi[h, i] = self.y[self.pivot_point] * \
                                np.exp(-i*self.x[h]/self.x[self.pivot_point])
            if self.model_type == 'legendre':
                interval = np.linspace(-0.999, 0.999, len(self.x))
                phi = []
                for i in range(self.N):
                    P = legendre(i)
                    phi.append(P(interval))
                phi = np.array(phi).T
            phi = matrix(phi)
        if self.basis_functions is not None:
            if self.args is None:
                phi = self.basis_functions(
                    self.x, self.y, self.pivot_point, self.N)
                phi = matrix(phi)
            if self.args is not None:
                phi = self.basis_functions(
                    self.x, self.y, self.pivot_point, self.N, *self.args)
                phi = matrix(phi)

        if self.model_type == 'loglog_polynomial':
            data_matrix = matrix(
                np.log10(self.y).astype(np.double), (len(self.y), 1),
                'd')
        else:
            data_matrix = matrix(
                self.y.astype(np.double), (len(self.y), 1),
                'd')

        if self.zero_crossings is None:
            h = matrix(0.0, ((self.N-self.constraints)*len(self.x), 1), 'd')
        else:
            h = matrix(
                0.0, (
                    (self.N-self.constraints-len(self.zero_crossings))
                    * len(self.x), 1), 'd')

        Q = phi.T*phi

        q = -phi.T*data_matrix

        if self.initial_params is None:
            qpfit = solvers.qp(Q, q, G, h)
        if self.initial_params is not None:
            print(self.initial_params)
            initvals = {'x': matrix(
                self.initial_params, (1, self.N), 'd')}
            qpfit = solvers.qp(Q, q, G, h, initvals=initvals)

        parameters = qpfit['x']

        if 'unknown' in qpfit['status']:
            if qpfit['iterations'] == self.cvxopt_maxiter:
                raise ValueError(
                    'ERROR: "Maximum number of iterations reached in' +
                    ' cvxopt routine." Increase value of' +
                    ' setting.cvxopt_maxiter')
            else:
                parameters = np.array(matrix(0, (self.N, 1), 'd'))
                if self.model_type == 'loglog_polynomial':
                    chi_squared = np.sum((np.log10(self.y))**2)
                else:
                    chi_squared = np.sum((self.y)**2)
                zc_dict = {}
        else:
            y = Models_class(
                parameters, self.x, self.y, self.N, self.pivot_point,
                self.model_type, self.new_basis).y_sum
            der = derivative_class(
                self.x, self.y, parameters, self.N, self.pivot_point,
                self.model_type, self.zero_crossings,
                self.constraints, self.new_basis)
            zc_dict = der.zc_dict

            if self.model_type == 'loglog_polynomial':
                chi_squared = np.sum((np.log10(self.y)-np.log10(y))**2)
            else:
                chi_squared = np.sum((self.y-y)**2)
            parameters = np.array(parameters)

        return parameters, chi_squared, zc_dict

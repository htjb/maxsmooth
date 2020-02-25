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
            all_output, ifp_list, initial_params,
            basis_functions, data_matrix, derivative_pres, model,
            derivatives_function, args, warnings, constraints):
        self.x = x/x.max()
        self.true_x = x
        self.y = y/y.std()+y.mean()/y.std()
        self.true_y = y
        self.N = N
        self.signs = signs
        self.mid_point = mid_point
        self.model_type = model_type
        self.cvxopt_maxiter = cvxopt_maxiter
        self.all_output = all_output
        self.ifp_list = ifp_list
        self.initial_params = initial_params
        self.basis_functions = basis_functions
        self.data_matrix = data_matrix
        self.derivative_pres = derivative_pres
        self.model = model
        self.derivatives_function = derivatives_function
        self.args = args
        self.warnings = warnings
        self.constraints = constraints
        self.parameters, self.chi_squared, self.ifp_dict = self.fit()

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
                if i <= (self.N-m):
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
                        if self.model_type == 'log_MSF_polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i) * np.log10(self.x/ \
                                self.x[self.mid_point])**i
                            derivatives.append(mth_order_derivative_term)
                        if self.model_type == 'MSF_2017_polynomial':
                            mth_order_derivative_term = np.math.factorial(m+i)\
                                / np.math.factorial(i) * (
                                self.x - self.x[self.mid_point])**i
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
            if m[i] >= self.constraints:
                if self.ifp_list is not None:
                    if m[i] not in set(self.ifp_list):
                        derivative_prefactors = constraint_prefactors(m[i])
                        if derivative_prefactors != []:
                            derivatives.append(derivative_prefactors)
                else:
                    derivative_prefactors = constraint_prefactors(m[i])
                    if derivative_prefactors != []:
                        derivatives.append(derivative_prefactors)

        for i in range(len(derivatives)):
            derivatives[i] *= signs[i]

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
                    if self.model_type == 'log_MSF_polynomial':
                        A[h, i] = np.log10(self.x[h]/self.x[self.mid_point])**i
                    if self.model_type == 'MSF_2017_polynomial':
                        A[h, i] = (self.x[h]-self.x[self.mid_point])**i
            A = matrix(A)
        if self.basis_functions is not None:
            if self.args is None:
                A = self.basis_functions(
                    self.x, self.y, self.mid_point, self.N)
            if self.args is not None:
                A = self.basis_functions(
                    self.x, self.y, self.mid_point, self.N, *self.args)

        if self.data_matrix is None:
            b = matrix(self.y.astype(np.double), (len(self.y), 1), 'd')
        if self.data_matrix is not None:
            b = self.data_matrix

        if self.ifp_list is None:
            h = matrix(-1e-7, ((self.N-self.constraints)*len(self.x), 1), 'd')
        else:
            h = matrix(-1e-7, ((self.N-self.constraints-len(self.ifp_list))
                *len(self.x), 1), 'd')

        P = A.T*A
        q = -A.T*b

        if self.initial_params is None:
            params0 = [(self.y[-1]-self.y[0])/2]*(self.N)
        if self.initial_params is not None:
            params0 = self.initial_params

        qpfit = solvers.coneqp(P, q, G, h, initvals=params0)

        parameters = qpfit['x']

        for i in range(len(parameters)):
            if self.model_type == 'normalised_polynomial':
                if i == 0:
                    parameters[i] = parameters[i]*(1+self.true_y.mean()/self.true_y[self.mid_point]) \
                    -self.true_y.mean()/(self.true_y[self.mid_point])
                else:
                    parameters[i] = parameters[i]*(1+self.true_y.mean()/self.true_y[self.mid_point])
            if self.model_type == 'polynomial':
                if i == 0:
                    parameters[i] = parameters[i]*self.true_y.std() - self.true_y.mean()
                else:
                    parameters[i] = (parameters[i]/self.true_x.max()**(i))*self.true_y.std()
            if self.model_type == 'log_MSF_polynomial':
                if i == 0:
                    parameters[i] = parameters[i]*self.true_y.std() - self.true_y.mean()
                else:
                    parameters[i] = (parameters[i])*self.true_y.std()
            if self.model_type == 'MSF_2017_polynomial':
                if i == 0:
                    parameters[i] = parameters[i]*self.true_y.std() - self.true_y.mean()
                else:
                    parameters[i] = (parameters[i]/self.true_x.max()**(i))*self.true_y.std()

        if 'unknown' in qpfit['status']:
            if qpfit['iterations'] == self.cvxopt_maxiter:
                print(
                    'ERROR: "Maximum number of iterations reached in' +
                    ' cvxopt routine." Increase value of' +
                    ' setting.cvxopt_maxiter')
                sys.exit(1)
            else:
                parameters = np.array(matrix(0, (self.N, 1), 'd'))
                chi_squared = np.sum((self.true_y)**2)
                ifp_dict = {}
        else:
            y = Models_class(
                parameters, self.true_x, self.true_y, self.N, self.mid_point,
                self.model_type, self.model, self.args).y_sum
            der = derivative_class(
                self.true_x, self.true_y, parameters, self.N, self.signs, self.mid_point,
                self.model_type, self.ifp_list, self.derivatives_function, self.args,
                self.warnings, self.constraints)
            ifp_dict = der.ifp_dict

            chi_squared = np.sum((self.true_y-y)**2)
            parameters = np.array(parameters)


        return parameters, chi_squared, ifp_dict

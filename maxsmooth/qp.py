from maxsmooth.derivatives import derivative_class
from maxsmooth.Models import Models_class
from cvxopt import matrix, solvers
import numpy as np
import sys
import warnings
from scipy.special import legendre, lpmv

warnings.simplefilter('always', UserWarning)


class qp_class(object):
    def __init__(
            self, x, y, N, signs, mid_point, model_type, cvxopt_maxiter,
            all_output, ifp_list, initial_params,
            basis_functions, data_matrix, derivative_pres, model,
            derivatives_function, args, warnings, constraints, parameter_transforms):
        self.y = y
        self.x = x
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
        self.parameter_transforms = parameter_transforms
        self.parameters, self.chi_squared, self.ifp_dict = self.fit()

    def fit(self):

        solvers.options['maxiters'] = self.cvxopt_maxiter
        solvers.options['show_progress'] = False

        def constraint_prefactors(m):
            # Derivative prefactors on parameters
            if self.model_type != 'legendre':
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

            if self.model_type == 'legendre':
                interval = np.linspace(-0.999, 0.999, len(self.x))
                alps = []
                for l in range(self.N):
                    alps.append(lpmv(m, l, interval))
                alps = np.array(alps)
                derivatives = []
                for h in range(len(alps)):
                    derivatives.append(((alps[h,:]*(-1)**(m))/(1-interval**2)**(m/2)))
                derivatives = np.array(derivatives)
                #print(derivatives)
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
            if self.model_type != 'legendre':
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
            if self.model_type == 'legendre':
                interval = np.linspace(-0.999, 0.999, len(self.x))
                A = []
                for l in range(self.N):
                    P = legendre(l)
                    A.append(P(interval))
                A=np.array(A).T
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
            h = matrix(0.0, ((self.N-self.constraints)*len(self.x), 1), 'd')
        else:
            h = matrix(0.0, ((self.N-self.constraints-len(self.ifp_list))
                *len(self.x), 1), 'd')

        Q = A.T*A
        q = -A.T*b
        import matplotlib.pyplot as plt
        plt.imshow(Q, cmap='jet')
        cbar = plt.colorbar()
        cbar.set_label(r'$Q$')
        plt.savefig('EDGES_Q_basic_qp.pdf')
        plt.show()
        sys.exit(1)

        if self.initial_params is None:
            qpfit = solvers.qp(Q, q, G, h)
        if self.initial_params is not None:
            params0 = self.initial_params
            qpfit = solvers.qp(Q, q, G, h, initvals=params0)

        parameters = qpfit['x']

        if 'unknown' in qpfit['status']:
            if qpfit['iterations'] == self.cvxopt_maxiter:
                print(
                    'ERROR: "Maximum number of iterations reached in' +
                    ' cvxopt routine." Increase value of' +
                    ' setting.cvxopt_maxiter')
                sys.exit(1)
            else:
                parameters = np.array(matrix(0, (self.N, 1), 'd'))
                chi_squared = np.sum((self.y)**2)
                ifp_dict = {}
        else:
            y = Models_class(
                parameters, self.x, self.y, self.N, self.mid_point,
                self.model_type, self.model, self.args).y_sum
            der = derivative_class(
                self.x, self.y, parameters, self.N, self.mid_point,
                self.model_type, self.ifp_list, self.derivatives_function, self.args,
                self.warnings, self.constraints)
            ifp_dict = der.ifp_dict

            chi_squared = np.sum((self.y-y)**2)
            parameters = np.array(parameters)


        return parameters, chi_squared, ifp_dict

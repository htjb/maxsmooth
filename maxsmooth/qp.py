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
            all_output, ifp, ifp_list, initial_params,
            basis_functions, data_matrix, derivative_pres, model,
            derivatives_function, args, warnings, feastol):
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
        self.ifp = ifp
        self.ifp_list = ifp_list
        self.initial_params = initial_params
        self.basis_functions = basis_functions
        self.data_matrix = data_matrix
        self.derivative_pres = derivative_pres
        self.model = model
        self.derivatives_function = derivatives_function
        self.args = args
        self.warnings = warnings
        self.cvxopt_feastol = feastol
        self.parameters, self.chi_squared, self.pass_fail = self.fit()

    def fit(self):

        solvers.options['maxiters'] = self.cvxopt_maxiter
        solvers.options['show_progress'] = False
        if self.cvxopt_feastol != 'Default':
            solvers.options['feastol'] = self.cvxopt_feastol
        if self.cvxopt_feastol == 'Default':
            pass
        
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
            for i in range(len(self.ifp_list)):
                if self.ifp_list[i]-2 >= (self.N-2):
                    print(
                        'ERROR: ifp_list element exceeds the number of' +
                        ' derivatives')
                    sys.exit(1)
            else:
                h_ifp = []
                ifp_list = [None]*(self.N-2)
                for i in range(self.N-2):
                    for j in range(len(self.ifp_list)):
                        if i == self.ifp_list[j]-2:
                            if ifp_list == []:
                                ifp_list[i] = i
                            elif np.any(ifp_list) != i:
                                ifp_list[i] = (i)
                for i in range(self.N-2):
                    if ifp_list[i] == i:
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

        y = Models_class(
            parameters, self.true_x, self.true_y, self.N, self.mid_point,
            self.model_type, self.model, self.args).y_sum
        der = derivative_class(
            self.true_x, self.true_y, parameters, self.N, self.signs, self.mid_point,
            self.model_type, self.ifp, self.derivatives_function, self.args,
            self.warnings)
        pass_fail = der.pass_fail

        chi_squared = np.sum((self.true_y-y)**2)
        parameters = np.array(parameters)

        if 'unknown' in qpfit['status']:
            if qpfit['iterations'] == self.cvxopt_maxiter:
                print(
                    'ERROR: "Maximum number of iterations reached in' +
                    ' cvxopt routine." Increase value of' +
                    ' setting.cvxopt_maxiter')
                sys.exit(1)
            else:
                print(
                    'ERROR: "CVXOPT:Terminated (singular KKT matrix)".' +
                    ' The problem cannot be solved by CVXOPT to the required' +
                    ' accuracy. Increasing the maxsmooth parameter' +
                    ' setting.cvxopt_feastol above the default value of 1e-7' +
                    ' will prevent this error.')
                sys.exit(1)

        return parameters, chi_squared, pass_fail

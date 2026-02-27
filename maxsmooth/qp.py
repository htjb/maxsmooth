from maxsmooth.derivatives import derivative_class
from maxsmooth.Models import Models_class
from cvxopt import matrix, solvers
import numpy as np
import warnings
from scipy.special import legendre, lpmv
import math

warnings.simplefilter('always', UserWarning)


def _constraint_prefactors(m, x, y, N, pivot_point, model_type,
                            derivative_pres, args):
    """Compute the constraint matrix prefactors for derivative order m."""
    if derivative_pres is None:
        if model_type not in ['legendre', 'exponential']:
            derivatives = []
            for i in range(N):
                if i <= m - 1:
                    derivatives.append([0]*len(x))
            for i in range(N-m):
                if model_type == 'normalised_polynomial':
                    mth_order_derivative_term = (
                        y[pivot_point] /
                        x[pivot_point]) \
                        * math.factorial(m + i) \
                        / math.factorial(i) * \
                        x**i / x[pivot_point]**(i + 1)
                    derivatives.append(mth_order_derivative_term)
                if model_type == 'polynomial':
                    mth_order_derivative_term = math.factorial(m+i)\
                        / math.factorial(i) * x**i
                    derivatives.append(mth_order_derivative_term)
                if model_type == 'log_polynomial':
                    mth_order_derivative_term = math.factorial(m+i)\
                        / math.factorial(i) * \
                        np.log10(x/x[pivot_point])**i
                    derivatives.append(mth_order_derivative_term)
                if model_type == 'loglog_polynomial':
                    mth_order_derivative_term = math.factorial(m+i)\
                        / math.factorial(i) * np.log10(x)**i
                    derivatives.append(mth_order_derivative_term)
                if model_type == 'difference_polynomial':
                    mth_order_derivative_term = math.factorial(m+i)\
                        / math.factorial(i) * (
                        x - x[pivot_point])**i
                    derivatives.append(mth_order_derivative_term)

    if derivative_pres is not None:
        if args is None:
            derivatives = derivative_pres(m, x, y, N, pivot_point)
        if args is not None:
            derivatives = derivative_pres(m, x, y, N, pivot_point, *args)

    if model_type == 'legendre':
        interval = np.linspace(-0.999, 0.999, len(x))
        alps = []
        for i in range(N):
            alps.append(lpmv(m, i, interval))
        alps = np.array(alps)
        derivatives = []
        for row in range(len(alps)):
            derivatives.append(
                ((alps[row, :]*(-1)**(m))/(1-interval**2)**(m/2)))
        derivatives = np.array(derivatives)
    if model_type == 'exponential':
        derivatives = np.empty([N, len(x)])
        for i in range(N):
            for col in range(len(x)):
                derivatives[i, col] = \
                    y[pivot_point] * \
                    (np.exp(-i*x[col]/x[pivot_point])) * \
                    (-i/x[pivot_point])**m
        derivatives = np.array(derivatives)

    derivatives = np.array(derivatives).astype(np.double)
    derivatives = matrix(derivatives)
    if derivatives.size == (len(x), N):
        pass
    else:
        derivatives = derivatives.T
    return derivatives


def precompute_matrices(x, y, N, pivot_point, model_type, zero_crossings,
                        constraints, new_basis):
    """Precompute QP matrices that do not depend on the sign vector.

    Returns a dict with base_derivatives, phi, Q, q, h. Pass as
    precomputed=... to qp_class when calling it in a loop over sign
    combinations to avoid redundant computation.
    """
    derivative_pres = new_basis['der_pres']
    basis_functions = new_basis['basis_function']
    args = new_basis['args']

    m_vals = np.arange(0, N, 1)
    base_derivatives = []
    for i in range(len(m_vals)):
        if m_vals[i] >= constraints:
            if zero_crossings is not None:
                if m_vals[i] not in set(zero_crossings):
                    dp = _constraint_prefactors(
                        m_vals[i], x, y, N, pivot_point, model_type,
                        derivative_pres, args)
                    if dp != []:
                        base_derivatives.append(dp)
            else:
                dp = _constraint_prefactors(
                    m_vals[i], x, y, N, pivot_point, model_type,
                    derivative_pres, args)
                if dp != []:
                    base_derivatives.append(dp)

    if basis_functions is None:
        phi = np.empty([len(x), N])
        if model_type != 'legendre':
            for row in range(len(x)):
                for i in range(N):
                    if model_type == 'normalised_polynomial':
                        phi[row, i] = y[pivot_point] * (
                            x[row] / x[pivot_point])**i
                    if model_type == 'polynomial':
                        phi[row, i] = (x[row])**i
                    if model_type == 'log_polynomial':
                        phi[row, i] = \
                            np.log10(x[row]/x[pivot_point])**i
                    if model_type == 'loglog_polynomial':
                        phi[row, i] = np.log10(x[row])**i
                    if model_type == 'difference_polynomial':
                        phi[row, i] = (x[row]-x[pivot_point])**i
                    if model_type == 'exponential':
                        phi[row, i] = y[pivot_point] * \
                            np.exp(-i*x[row]/x[pivot_point])
        if model_type == 'legendre':
            interval = np.linspace(-0.999, 0.999, len(x))
            phi = []
            for i in range(N):
                P = legendre(i)
                phi.append(P(interval))
            phi = np.array(phi).T
        phi = matrix(phi)
    else:
        if args is None:
            phi = basis_functions(x, y, pivot_point, N)
            phi = matrix(phi)
        if args is not None:
            phi = basis_functions(x, y, pivot_point, N, *args)
            phi = matrix(phi)

    if model_type == 'loglog_polynomial':
        data_matrix = matrix(
            np.log10(y).astype(np.double), (len(y), 1), 'd')
    else:
        data_matrix = matrix(
            y.astype(np.double), (len(y), 1), 'd')

    Q = phi.T * phi
    q = -phi.T * data_matrix

    if zero_crossings is None:
        h = matrix(0.0, ((N - constraints) * len(x), 1), 'd')
    else:
        h = matrix(
            0.0, (
                (N - constraints - len(zero_crossings))
                * len(x), 1), 'd')

    return {
        'base_derivatives': base_derivatives,
        'phi': phi,
        'Q': Q,
        'q': q,
        'h': h,
    }


class qp_class(object):
    def __init__(
            self, x, y, N, signs, pivot_point, model_type, cvxopt_maxiter,
            zero_crossings, initial_params,
            constraints, new_basis, precomputed=None):
        self.model_type = model_type
        self.pivot_point = pivot_point
        self.y = y
        self.x = x
        self.N = N
        self.signs = signs
        self.cvxopt_maxiter = cvxopt_maxiter
        self.zero_crossings = zero_crossings
        self.initial_params = initial_params
        self.basis_functions = new_basis['basis_function']
        self.derivative_pres = new_basis['der_pres']
        self.model = new_basis['model']
        self.derivatives_function = new_basis['derivatives_function']
        self.args = new_basis['args']
        self.new_basis = new_basis
        self.constraints = constraints
        self.precomputed = precomputed
        self.parameters, self.chi_squared, self.zc_dict = self.fit()

    def fit(self):

        solvers.options['maxiters'] = self.cvxopt_maxiter
        solvers.options['show_progress'] = False

        if self.precomputed is not None:
            base_derivatives = self.precomputed['base_derivatives']
            Q = self.precomputed['Q']
            q = self.precomputed['q']
            h = self.precomputed['h']
        else:
            pc = precompute_matrices(
                self.x, self.y, self.N, self.pivot_point, self.model_type,
                self.zero_crossings, self.constraints, self.new_basis)
            base_derivatives = pc['base_derivatives']
            Q = pc['Q']
            q = pc['q']
            h = pc['h']

        signs = matrix(self.signs)
        G = matrix([base_derivatives[i] * signs[i]
                    for i in range(len(base_derivatives))])

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

import numpy as np
from scipy.special import lpmv


class derivative_class(object):
    def __init__(
                self, x, y, params, N, pivot_point, model_type, zero_crossings,
                constraints, new_basis, **kwargs):
        self.x = x
        self.y = y
        self.N = N
        self.params = params
        self.pivot_point = pivot_point
        self.model_type = model_type
        self.zero_crossings = zero_crossings
        self.derivatives_function = new_basis['derivatives_function']
        self.args = new_basis['args']
        self.constraints = constraints

        self.call_type = kwargs.pop('call_type', 'checking')

        self.derivatives, self.pass_fail, self.zc_dict = \
            self.derivatives_func()

    def derivatives_func(self):

        def mth_order_derivatives(m):
            if self.derivatives_function is None:
                if np.any(self.model_type != ['legendre', 'exponential']):
                    mth_order_derivative = []
                    for i in range(self.N-m):
                        if self.model_type == 'normalised_polynomial':
                            mth_order_derivative_term = (
                                self.y[self.pivot_point] /
                                self.x[self.pivot_point]) * \
                                np.math.factorial(m+i) / \
                                np.math.factorial(i) * \
                                self.params[int(m)+i]*(self.x)**i / \
                                (self.x[self.pivot_point])**(i+1)
                            mth_order_derivative.append(
                                mth_order_derivative_term)
                        if self.model_type == 'polynomial':
                            mth_order_derivative_term = \
                                np.math.factorial(m+i) / \
                                np.math.factorial(i) * \
                                self.params[int(m)+i]*(self.x)**i
                            mth_order_derivative.append(
                                mth_order_derivative_term)
                        if self.model_type == 'log_polynomial':
                            mth_order_derivative_term = \
                                np.math.factorial(m+i) / \
                                np.math.factorial(i) * \
                                self.params[int(m) + i] * \
                                np.log10(self.x/self.x[self.pivot_point])**i
                            mth_order_derivative.append(
                                mth_order_derivative_term)
                        if self.model_type == 'loglog_polynomial':
                            mth_order_derivative_term = \
                                np.math.factorial(m+i) / \
                                np.math.factorial(i) * \
                                self.params[int(m)+i]*np.log10(self.x)**i
                            mth_order_derivative.append(
                                mth_order_derivative_term)
                        if self.model_type == 'difference_polynomial':
                            mth_order_derivative_term = \
                                np.math.factorial(m+i) / \
                                np.math.factorial(i) * \
                                self.params[int(m)+i] * \
                                (self.x-self.x[self.pivot_point])**i
                            mth_order_derivative.append(
                                mth_order_derivative_term)

            if self.derivatives_function is not None:
                if self.args is None:
                    derivatives = \
                        self.derivatives_function(
                            m, self.x, self.y, self.N, self.pivot_point,
                            self.params)
                if self.args is not None:
                    derivatives = \
                        self.derivatives_function(
                            m, self.x, self.y, self.N, self.pivot_point,
                            self.params, *self.args)
                mth_order_derivative = derivatives

            if self.model_type == 'legendre':
                interval = np.linspace(-0.999, 0.999, len(self.x))
                alps = []
                for i in range(self.N):
                    alps.append(lpmv(m, i, interval))
                alps = np.array(alps)
                derivatives = []
                for h in range(len(alps)):
                    derivatives.append(
                        ((alps[h, :]*(-1)**(m))/(1-interval**2)**(m/2))
                        * self.params[h, 0])
                mth_order_derivative = np.array(derivatives)
            if self.model_type == 'exponential':
                derivatives = np.empty([self.N, len(self.x)])
                for i in range(self.N):
                    for h in range(len(self.x)):
                        derivatives[i, h] = \
                            self.y[self.pivot_point] * (
                            self.params[i] *
                            np.exp(-i * self.x[h]/self.x[self.pivot_point])) \
                            * (-i/self.x[self.pivot_point])**m
                mth_order_derivative = np.array(derivatives)

            if type(mth_order_derivative) == list:
                mth_order_derivative = np.array(mth_order_derivative)
            if mth_order_derivative.shape == (len(self.x), self.N):
                mth_order_derivative = mth_order_derivative.sum(axis=1)
            else:
                mth_order_derivative = mth_order_derivative.sum(axis=0)

            return mth_order_derivative

        m = np.arange(0, self.N, 1)
        derivatives = []
        zc_derivatives = []
        zc_orders = []
        for i in range(len(m)):
            if m[i] < self.constraints:
                zc_orders.append(m[i])
                zc_derivatives.append(mth_order_derivatives(m[i]))
            if m[i] >= self.constraints:
                if self.zero_crossings is not None:
                    if m[i] not in set(self.zero_crossings):
                        derivatives.append(mth_order_derivatives(m[i]))
                    if m[i] in set(self.zero_crossings):
                        zc_orders.append(m[i])
                        zc_derivatives.append(mth_order_derivatives(m[i]))
                else:
                    derivatives.append(mth_order_derivatives(m[i]))
        derivatives = np.array(derivatives)

        zc_derivatives = np.array(zc_derivatives)
        zc_orders = np.array(zc_orders)

        # Check constrained derivatives
        pass_fail = []
        for i in range(derivatives.shape[0]):
            if np.all(derivatives[i, :] >= -1e-6) or \
                    np.all(derivatives[i, :] <= 1e-6):
                pass_fail.append(1)
            else:
                pass_fail.append(0)
        pass_fail = np.array(pass_fail)

        zc_dict = {}
        for i in range(zc_derivatives.shape[0]):
            if np.all(zc_derivatives[i, :] >= -1e-6) or \
                    np.all(zc_derivatives[i, :] <= 1e-6):
                zc_dict[str(zc_orders[i])] = 1
            else:
                zc_dict[str(zc_orders[i])] = 0

        if self.call_type == 'checking':
            if np.any(pass_fail == 0):
                print('Pass or fail', pass_fail)
                raise Exception(
                    '"Condition Violated" Derivatives feature' +
                    ' crossing points.')

        return derivatives, pass_fail, zc_dict

import numpy as np
import sys
import warnings
from scipy.special import lpmv

warnings.simplefilter('always', UserWarning)


class derivative_class(object):
    def __init__(
                self, x, y, params, N, mid_point, model_type, ifp_list,
                derivatives_function, args, warnings, constraints):
        self.x = x
        self.y = y
        self.N = N
        self.params = params
        self.mid_point = mid_point
        self.model_type = model_type
        self.ifp_list = ifp_list
        self.derivatives_function = derivatives_function
        self.args = args
        self.warnings = warnings
        self.constraints = constraints
        self.derivatives, self.pass_fail, self.ifp_dict = \
            self.derivatives_func()

    def derivatives_func(self):

        def mth_order_derivatives(m):
            if self.model_type != 'legendre':
                mth_order_derivative = []
                for i in range(self.N-m):
                    if i < (self.N-m) or i == (self.N-m):
                        if self.derivatives_function is None:
                            if self.model_type == 'normalised_polynomial':
                                mth_order_derivative_term = (
                                    self.y[self.mid_point] /
                                    self.x[self.mid_point]) * \
                                    np.math.factorial(m+i) / \
                                    np.math.factorial(i) * \
                                    self.params[int(m)+i]*(self.x)**i / \
                                    (self.x[self.mid_point])**(i+1)
                                mth_order_derivative.append(
                                    mth_order_derivative_term)
                            if self.model_type == 'polynomial':
                                mth_order_derivative_term = \
                                    np.math.factorial(m+i) / \
                                    np.math.factorial(i) * \
                                    self.params[int(m)+i]*(self.x)**i
                                mth_order_derivative.append(
                                    mth_order_derivative_term)
                            if self.model_type == 'log_MSF_polynomial':
                                mth_order_derivative_term = \
                                    np.math.factorial(m+i) / \
                                    np.math.factorial(i) * \
                                    self.params[int(m)+i]*np.log10(self.x/ \
                                    self.x[self.mid_point])**i
                                mth_order_derivative.append(
                                    mth_order_derivative_term)
                            if self.model_type == 'MSF_2017_polynomial':
                                mth_order_derivative_term = \
                                    np.math.factorial(m+i) / \
                                    np.math.factorial(i) * \
                                    self.params[int(m)+i] * \
                                    (self.x-self.x[self.mid_point])**i
                                mth_order_derivative.append(
                                    mth_order_derivative_term)
                        if self.derivatives_function is not None:
                            if self.args is None:
                                mth_order_derivative_term = \
                                    self.derivatives_function(
                                        m, i, self.x, self.y, self.mid_point,
                                        self.params)
                            if self.args is not None:
                                mth_order_derivative_term = \
                                    self.derivatives_function(
                                        m, i, self.x, self.y, self.mid_point,
                                        self.params, *self.args)
                            mth_order_derivative.append(
                                mth_order_derivative_term)

            if self.model_type == 'legendre':
                interval = np.linspace(-0.999, 0.999, len(self.x))
                alps = []
                for l in range(self.N):
                    alps.append(lpmv(m, l, interval))
                alps = np.array(alps)
                derivatives = []
                for h in range(len(alps)):
                    derivatives.append(((alps[h,:]*(-1)**(m))/(1-interval**2)**(m/2))
                        *self.params[h, 0])
                mth_order_derivative = np.array(derivatives)
            mth_order_derivative = np.array(mth_order_derivative).sum(axis=0)
            return mth_order_derivative

        m = np.arange(0, self.N, 1)
        derivatives = []
        ifp_derivatives = []
        ifp_orders = []
        for i in range(len(m)):
            if m[i] < self.constraints:
                ifp_orders.append(m[i])
                ifp_derivatives.append(mth_order_derivatives(m[i]))
            if m[i] >= self.constraints:
                if self.ifp_list is not None:
                    if m[i] not in set(self.ifp_list):
                        derivatives.append(mth_order_derivatives(m[i]))
                    if m[i] in set(self.ifp_list):
                        ifp_orders.append(m[i])
                        ifp_derivatives.append(mth_order_derivatives(m[i]))
                else:
                    derivatives.append(mth_order_derivatives(m[i]))
        derivatives = np.array(derivatives)

        ifp_derivatives = np.array(ifp_derivatives)
        ifp_orders = np.array(ifp_orders)

        # Check constrained derivatives
        pass_fail = []
        for i in range(derivatives.shape[0]):
            if np.all(derivatives[i, :] >= -1e-6) or np.all(derivatives[i, :] <= 1e-6):
                pass_fail.append(1)
            else:
                pass_fail.append(0)
        pass_fail = np.array(pass_fail)

        # ifp dictionary for reporting back to user presence of inflection
        # points
        ifp_dict = {}
        for i in range(ifp_derivatives.shape[0]):
            if np.all(ifp_derivatives[i, :] >= -1e-6) or np.all(ifp_derivatives[i, :] <= 1e-6):
                ifp_dict[str(ifp_orders[i])] = 1
            else:
                ifp_dict[str(ifp_orders[i])] = 0

        if np.any(pass_fail == 0):
            print('Pass or fail', pass_fail)
            print(
                'ERROR: "Condition Violated" Derivatives feature' +
                ' crossing points.')
            sys.exit(1)

        return derivatives, pass_fail, ifp_dict

import numpy as np
import sys
import warnings

warnings.simplefilter('always', UserWarning)


class derivative_class(object):
    def __init__(
                self, x, y, params, N, signs, mid_point, model_type, ifp,
                derivatives_function, args, warnings):
        self.signs = signs
        self.x = x
        self.y = y
        self.N = N
        self.params = params
        self.mid_point = mid_point
        self.model_type = model_type
        self.ifp = ifp
        self.derivatives_function = derivatives_function
        self.args = args
        self.warnings = warnings
        self.derivatives, self.pass_fail = self.derivatives_func()

    def derivatives_func(self):

        def mth_order_derivatives(m):
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
                        if self.model_type == 'MSF_2017_polynomial':
                            mth_order_derivative_term = \
                                np.math.factorial(m+i) / \
                                np.math.factorial(i) * \
                                self.params[int(m)+i] * \
                                (self.x-self.x[self.mid_point])**i
                            mth_order_derivative.append(
                                mth_order_derivative_term)
                        if self.model_type == 'logarithmic_polynomial':
                            mth_order_derivative_term = \
                                np.math.factorial(m+i) / \
                                np.math.factorial(i) * \
                                self.params[int(m)+i] * \
                                np.log10(self.x)**i
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
            mth_order_derivative = np.array(mth_order_derivative).sum(axis=0)
            return mth_order_derivative

        m = np.arange(0, self.N, 1)
        derivatives = []
        for i in range(len(m)):
            if m[i] >= 2:
                derivatives.append(mth_order_derivatives(m[i]))
        derivatives = np.array(derivatives)

        pass_fail = []
        for i in range(derivatives.shape[0]):
            # In the array pass_fail a 0 signifies that the derivatives
            # features an inflection point. The position of the 0 in the list
            # informs the user which derivative that inflection point is in,
            # where position one in the list is a second order derivative.
            if np.all(derivatives[i, :] > 0) or np.all(derivatives[i, :] < 0):
                pass_fail.append(1)
            else:
                pass_fail.append(0)

        pass_fail = np.array(pass_fail)

        if np.any(pass_fail == 0):
            if self.ifp is True:
                if self.warnings is True:
                    warnings.warn(
                        'WARNING: setting.ipf = True has lead to derivatives' +
                        ' including inflection points.', stacklevel=2)
            if self.ifp is False:
                print('Pass or fail', pass_fail)
                print(
                    'ERROR: "Condition Violated" Derivatives feature' +
                    ' crossing points.')
                sys.exit(1)

        return derivatives, pass_fail

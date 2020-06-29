import numpy as np
from maxsmooth.DCF import smooth

x = np.load('Data/x.npy')
y = np.load('Data/y.npy')

N = 10

# This example is essentially the 'normalised_polynomial' model but
# with x and y normalised at different points than the built in model.

# Additional arguments used to define the new basis
arguments = [x[-1]*10, y[-1]*10]
#Initial parameters for cvxopt
params0 = [y[0]]*(N)

def basis_functions(x, y, mid_point, N, *args):

    phi = np.empty([len(x), N])
    for h in range(len(x)):
        for i in range(N):
            phi[h, i] = args[1]*(x[h]/args[0])**i

    return phi


def model(x, y, mid_point, N, params, *args):

    y_sum = args[1]*np.sum([
        params[i]*(x/args[0])**i
        for i in range(N)], axis=0)

    return y_sum


def derivative(m, x, y, mid_point, params, *args):

    mth_order_derivative = []
    for i in range(N):
        if i <= m - 1:
            mth_order_derivative.append([0]*len(x))
    for i in range(N - m):
            mth_order_derivative_term = args[1]*np.math.factorial(m+i) / \
                np.math.factorial(i) * \
                params[int(m)+i]*(x)**i / \
                (args[0])**(i + 1)
            mth_order_derivative.append(
                mth_order_derivative_term)

    return mth_order_derivative


def derivative_pre(m, x, y, mid_point, *args):
    # 'derivative prefactors' i.e. elements of G matrix
    mth_order_derivative = []
    for i in range(N):
        if i <= m - 1:
            mth_order_derivative.append([0]*len(x))
    for i in range(N - m):
            mth_order_derivative_term = args[1]*np.math.factorial(m+i) / \
                np.math.factorial(i) * \
                (x)**i / \
                (args[0])**(i + 1)
            mth_order_derivative.append(
                mth_order_derivative_term)

    return mth_order_derivative


result = smooth(x, y, N,
    basis_functions=basis_functions, model=model,
    derivatives=derivative, der_pres=derivative_pre, args=arguments)
print('Objective Funtion Evaluations:\n', result.Optimum_chi)
print('RMS:\n', result.rms)
# print('Parameters:\n', result.Optimum_params[2])
# print('Fitted y:\n', result.y_fit)
# print('Sign Combinations:\n', result.Optimum_signs)
# print('Derivatives:\n', result.derivatives)

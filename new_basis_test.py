import numpy as np
from maxsmooth.settings import setting
from maxsmooth.msf import smooth
from cvxopt import matrix

setting = setting()

x = np.load('Data/x.npy')
y = np.load('Data/y.npy')

N = [3, 4, 5, 6, 7, 8, 9, 10, 11]

arguments = [x[-1]*10, y[-1]*10]

params0 = [y[0]]*(N[0])

b = matrix(y, (len(y), 1), 'd')


def basis_functions(x, y, mid_point, N, *args):

    A = np.empty([len(x), N])
    for h in range(len(x)):
        for i in range(N):
            A[h, i] = args[1]*(x[h]/args[0])**i
    A = matrix(A)

    return A


def model(x, y, mid_point, N, params, *args):

    y_sum = args[1]*np.sum([
        params[i]*(x/args[0])**i
        for i in range(N)], axis=0)

    return y_sum


def derivative(m, i, x, y, mid_point, params, *args):

    mth_order_derivative_term = args[1]*np.math.factorial(m+i) / \
        np.math.factorial(i) * \
        params[int(m)+i]*(x)**i / \
        (args[0])**(int(m)+i)

    return mth_order_derivative_term


def derivative_pre(m, i, x, y, mid_point, *args):

    mth_order_derivative_term = args[1]*np.math.factorial(m+i) / \
        np.math.factorial(i)*(x)**i/(args[0])**(int(m)+i)

    return mth_order_derivative_term


result = smooth(
    x, y, N, setting, initial_params=params0,
    basis_functions=basis_functions, model=model, data_matrix=b,
    derivatives=derivative, der_pres=derivative_pre, args=arguments)
print('Objective Funtion Evaluations:\n', result.Optimum_chi)
print('RMS:\n', result.rms)
# print('Parameters:\n', result.Optimum_params[2])
# print('Fitted y:\n', result.y_fit)
# print('Sign Combinations:\n', result.Optimum_signs)
# print('Derivatives:\n', result.derivatives)

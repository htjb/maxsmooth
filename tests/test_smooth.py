import numpy as np
import pytest
import os
import shutil
from maxsmooth.DCF import smooth
from numpy.testing import assert_almost_equal

def test_api():
    # Check api
    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 4
    sol = smooth(x, y, N)
    # Check RMS/Chi/y_fit calculated/returned correctly
    assert_almost_equal(sol.rms, np.sqrt(np.sum((y - sol.y_fit)**2)/len(y)))
    assert_almost_equal( sol.optimum_chi , ((y - sol.y_fit)**2).sum())

    def model(x, N, params):
        y_sum = np.sum([
            params[i]*(x-x[len(x)//2])**i
            for i in range(N)], axis=0)
        return y_sum

    y_fit = model(x, N, sol.optimum_params)
    assert_almost_equal( sol.optimum_chi , ((y - y_fit)**2).sum())
    assert(len(sol.optimum_signs) == N-2)
    assert(sol.derivatives.shape == (N-2, len(y)))

def test_keywords():
    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0, 1, Ndat)
    y = 1 + x + x**2 + x**3

    N = 4

    with pytest.raises(Exception):
        sol = smooth(x, y, N, color='pink')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, base_dir='file')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, base_dir=5)
    with pytest.raises(Exception):
        sol = smooth(x, y, 5.5)
    with pytest.raises(Exception):
        sol = smooth(x, y, 'string')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, fit_type='banana')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, model_type='pink')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, pivot_point='string')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, pivot_point=5.5)
    with pytest.raises(Exception):
        sol = smooth(x, y, N, pivot_point=len(x)+10)
    with pytest.raises(Exception):
        sol = smooth(x, y, N, cvxopt_maxiter='string')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, cvxopt_maxiter=41.2)
    with pytest.raises(Exception):
        sol = smooth(x, y, N, all_output=9)
    with pytest.raises(Exception):
        sol = smooth(x, y, N, data_save='string')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, constraints='string')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, constraints= 20)
    with pytest.raises(Exception):
        sol = smooth(x, y, N, zero_crossings=[3.3])
    with pytest.raises(Exception):
        sol = smooth(x, y, N, zero_crossings=[1])
    with pytest.raises(Exception):
        sol = smooth(x, y, N, chi_squared_limit='string')
    with pytest.raises(Exception):
        sol = smooth(x, y, N, chi_squared_limit=[1, 2])
    with pytest.raises(Exception):
        sol = smooth(x, y, N, cap=5.5)
    with pytest.raises(Exception):
        sol = smooth(x, y, N, initial_params=[1]*(N+10))
    with pytest.raises(Exception):
        sol = smooth(x, y, N, initial_params=['string']*(N))

def test_smooth_fit():
    # Check parameters of smooth function are correct
    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0, 1, Ndat)
    y = 1 + x + x**2 + x**3

    N = 4
    sol = smooth(x, y, N, model_type='polynomial', all_output=True)

    assert_almost_equal(sol.optimum_params.T[0], [1, 1, 1, 1], decimal=3)

    for i in range(len(sol.derivatives)):
        assert(
            np.all(sol.derivatives[i] <= 1e-6) or
            np.all(sol.derivatives[i] >= -1e-6))

    sol = smooth(
        x, y, N, model_type='polynomial', fit_type='qp',
        all_output=True)

    assert(sol.cap is None)
    assert_almost_equal(sol.optimum_params.T[0], [1, 1, 1, 1], decimal=3)

    for i in range(len(sol.derivatives)):
        assert(
            np.all(sol.derivatives[i] <= 1e-6) or
            np.all(sol.derivatives[i] >= -1e-6))

    with pytest.raises(Exception):
        sol = smooth(
            x, y, N, model_type='polynomial', cvxopt_maxiter=1)

    sol = smooth(
        x, y, N, model_type='polynomial', initial_params=[1]*N,
        cap=100, chi_squared_limit=1e4)

    assert(sol.cap == 100)
    assert_almost_equal(sol.optimum_params.T[0], [1, 1, 1, 1], decimal=3)

def test_output_directional_exp():

    Ndat = 100
    x = np.linspace(50, 150, Ndat)
    y = 5e7*x**(-2.5)

    N = 10
    sol = smooth(x, y, N, model_type='legendre', all_output=True)

    assert_almost_equal( sol.optimum_chi , ((y - sol.y_fit)**2).sum())

def test_data_save():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 4

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    sol = smooth(x, y, N, data_save=True, base_dir='new_dir/')

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Output_Parameters/') is True)
    assert(os.path.exists('new_dir/Output_Signs/') is True)
    assert(os.path.exists('new_dir/Output_Evaluation/') is True)
    assert(
        os.path.isfile(
            'new_dir/Optimal_Results_qp-sign_flipping_4.txt') is True)
    assert(
        os.path.isfile(
            'new_dir/Output_Parameters/4_qp-sign_flipping.txt') is True)
    assert(
        os.path.isfile(
            'new_dir/Output_Signs/4_qp-sign_flipping.txt') is True)
    assert(
        os.path.isfile(
            'new_dir/Output_Evaluation/4_qp-sign_flipping.txt') is True)

    sol = smooth(x, y, N, data_save=True, base_dir='new_dir/')

def test_new_basis():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 4

    arguments = [x[-1]*10, y[-1]*10]

    def basis_functions(x, y, pivot_point, N, *args):

        phi = np.empty([len(x), N])
        for h in range(len(x)):
            for i in range(N):
                phi[h, i] = args[1]*(x[h]/args[0])**i

        return phi

    def model(x, y, pivot_point, N, params, *args):

        y_sum = args[1]*np.sum([
            params[i]*(x/args[0])**i
            for i in range(N)], axis=0)

        return y_sum

    def derivative(m, x, y, N, pivot_point, params, *args):
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
        mth_order_derivative = np.array(mth_order_derivative).T

        return mth_order_derivative

    def derivative_pre(m, x, y, pivot_point, *args):

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

    sol = smooth(x, y, N, basis_functions=basis_functions, model=model,
    derivatives=derivative, der_pres=derivative_pre, args=arguments)

    assert(sol.model_type == 'user_defined')
    assert_almost_equal(sol.rms, np.sqrt(np.sum((y - sol.y_fit)**2)/len(y)))
    assert_almost_equal( sol.optimum_chi , ((y - sol.y_fit)**2).sum())

    with pytest.raises(Exception):
        sol = smooth(x, y, N, basis_functions=basis_functions, model=None,
        derivatives=derivative, der_pres=derivative_pre, args=arguments)

def test_new_basis_without_args():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 4

    def basis_functions(x, y, pivot_point, N, *args):

        phi = np.empty([len(x), N])
        for h in range(len(x)):
            for i in range(N):
                phi[h, i] = (x[h])**i

        return phi

    def model(x, y, pivot_point, N, params, *args):

        y_sum = np.sum([
            params[i]*(x)**i
            for i in range(N)], axis=0)

        return y_sum

    def derivative(m, x, y, N, pivot_point, params, *args):
        mth_order_derivative = []
        for i in range(N):
            if i <= m - 1:
                mth_order_derivative.append([0]*len(x))
        for i in range(N - m):
                mth_order_derivative_term = np.math.factorial(m+i) / \
                    np.math.factorial(i) * \
                    params[int(m)+i]*(x)**i
                mth_order_derivative.append(
                    mth_order_derivative_term)

        return mth_order_derivative

    def derivative_pre(m, x, y, pivot_point, *args):

        mth_order_derivative = []
        for i in range(N):
            if i <= m - 1:
                mth_order_derivative.append([0]*len(x))
        for i in range(N - m):
                mth_order_derivative_term = np.math.factorial(m+i) / \
                    np.math.factorial(i) * \
                    (x)**i
                mth_order_derivative.append(
                    mth_order_derivative_term)

        return mth_order_derivative

    sol = smooth(x, y, N, basis_functions=basis_functions, model=model,
    derivatives=derivative, der_pres=derivative_pre)

    assert(sol.model_type == 'user_defined')
    assert_almost_equal(sol.rms, np.sqrt(np.sum((y - sol.y_fit)**2)/len(y)))
    assert_almost_equal( sol.optimum_chi , ((y - sol.y_fit)**2).sum())

    with pytest.raises(Exception):
        sol = smooth(x, y, N, basis_functions=basis_functions, model=None,
        derivatives=derivative, der_pres=derivative_pre)

def test_ifp():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 10

    sol = smooth(
        x, y, N, zero_crossings=[4, 5, 6], constraints=1, all_output=True)

    assert(len(sol.optimum_zc_dict) == 4)
    assert(type(sol.optimum_zc_dict) is dict)

    sol = smooth(
        x, y, N, zero_crossings=[4, 5, 6],
        constraints=1, all_output=True, fit_type='qp')

    assert(len(sol.optimum_zc_dict) == 4)
    assert(type(sol.optimum_zc_dict) is dict)
